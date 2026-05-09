"""Handlers for chat-level setting commands (timezone, boolean toggles)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mai_gram.db.database import get_session
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.messenger.base import IncomingMessage, Messenger


async def handle_timezone(
    message: IncomingMessage,
    messenger: Messenger,
    get_chat: object,
    chat_id: str,
) -> None:
    """Handle /timezone command -- show or set the chat's timezone."""
    tz_arg = (message.command_args or "").strip()

    async with get_session() as session:
        from sqlalchemy import select

        from mai_gram.db.models import Chat

        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if not chat:
            await messenger.send_message(
                OutgoingMessage(
                    text="No chat exists yet. Use /start to create one.",
                    chat_id=message.chat_id,
                )
            )
            return

        if not tz_arg:
            await messenger.send_message(
                OutgoingMessage(
                    text=f"Current timezone: {chat.timezone}\n\nUsage: /timezone Europe/Moscow",
                    chat_id=message.chat_id,
                )
            )
            return

        from zoneinfo import available_timezones

        if tz_arg not in available_timezones():
            await messenger.send_message(
                OutgoingMessage(
                    text=(
                        f"Unknown timezone: {tz_arg}\n\n"
                        "Examples: Europe/Moscow, US/Eastern, Asia/Tokyo, UTC"
                    ),
                    chat_id=message.chat_id,
                )
            )
            return

        chat.timezone = tz_arg
        await session.commit()

    await messenger.send_message(
        OutgoingMessage(
            text=f"Timezone set to: {tz_arg}",
            chat_id=message.chat_id,
        )
    )
