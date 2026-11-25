"""Database operations for actions that target a persisted conversation turn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, func, literal, select, update
from sqlalchemy.dialects.sqlite import insert

from mai_gram.db.models import Chat, HistoryActionConfirmation, Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
    from sqlalchemy.sql import ColumnElement


def assistant_message_target(chat_id: str, target: int | str) -> ColumnElement[bool]:
    """Match an assistant row in the current conversation."""
    identity = Message.id == target if isinstance(target, int) else Message.history_ref == target
    return (Message.chat_id == chat_id) & (Message.role == "assistant") & identity


async def _claim_confirmation(
    connection: AsyncConnection,
    chat_id: str,
    confirmation: tuple[str, str],
    target_exists: ColumnElement[bool],
) -> bool:
    telegram_chat_id, telegram_message_id = confirmation
    claimed = await connection.execute(
        insert(HistoryActionConfirmation)
        .from_select(
            ["chat_id", "telegram_chat_id", "telegram_message_id"],
            select(Chat.id, literal(telegram_chat_id), literal(telegram_message_id)).where(
                Chat.id == chat_id, target_exists
            ),
        )
        .on_conflict_do_nothing()
    )
    return claimed.rowcount == 1


async def cut_history(
    session: AsyncSession,
    chat_id: str,
    target: int | str,
    *,
    confirmation: tuple[str, str] | None = None,
) -> int | None:
    """Claim the Telegram confirmation and move the boundary in one transaction."""
    target_id = (
        select(Message.id)
        .where(assistant_message_target(chat_id, target))
        .correlate(None)
        .scalar_subquery()
    )
    connection = await session.connection()
    if confirmation is not None and not await _claim_confirmation(
        connection, chat_id, confirmation, target_id.is_not(None)
    ):
        return None

    result = await connection.execute(
        update(Chat)
        .where(
            Chat.id == chat_id,
            target_id.is_not(None),
            Chat.cut_above_message_id.is_distinct_from(target_id),
        )
        .values(cut_above_message_id=target_id)
    )
    if result.rowcount != 1:
        return None

    count = await session.scalar(
        select(func.count(Message.id)).where(Message.chat_id == chat_id, Message.id <= target_id)
    )
    return int(count or 0)


async def rollback_history(session: AsyncSession, chat_id: str, target: int | str) -> bool:
    """Delete the exact target and its suffix within the caller's transaction."""
    connection = await session.connection()
    message_id = await connection.scalar(
        delete(Message)
        .where(assistant_message_target(chat_id, target), Message.chat.has())
        .returning(Message.id)
    )
    if message_id is None:
        return False
    await connection.execute(
        delete(Message).where(Message.chat_id == chat_id, Message.id > message_id)
    )
    return True


async def claim_retry(
    session: AsyncSession,
    chat_id: str,
    token: str,
    confirmation: tuple[str, str] | None,
) -> bool:
    """Start a write transaction only for a still-current pending turn."""
    latest_id = (
        select(Message.id)
        .where(Message.chat_id == chat_id)
        .order_by(Message.id.desc())
        .limit(1)
        .correlate(None)
        .scalar_subquery()
    )
    eligible = (
        select(Message.id)
        .where(
            Message.id == latest_id,
            Message.history_ref == token,
            Message.role.in_(("user", "tool")),
        )
        .exists()
    )
    connection = await session.connection()
    if confirmation is not None:
        return await _claim_confirmation(connection, chat_id, confirmation, eligible)
    result = await connection.execute(
        update(Chat).where(Chat.id == chat_id, eligible).values(id=Chat.id)
    )
    return result.rowcount == 1
