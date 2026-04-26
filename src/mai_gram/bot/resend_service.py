"""Service for re-sending the last persisted assistant response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.messenger.base import Messenger


class ResendRenderer(Protocol):
    async def _send_response(
        self,
        chat_id: str,
        *,
        response_text: str | None,
        response_reasoning: str | None = None,
        show_reasoning: bool = False,
        keyboard: object = None,
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class ResendResult:
    """Outcome of the resend-last workflow."""

    sent_message_ids: list[str]
    replaced_previous: bool


class ResendService:
    """Load and re-send the most recent assistant message for a chat."""

    def __init__(
        self,
        messenger: Messenger,
        *,
        renderer: ResendRenderer,
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._renderer = renderer
        self._resolve_chat_id = resolve_chat_id

    async def handle_resend(
        self,
        message: IncomingMessage,
        *,
        previous_response_ids: list[str],
    ) -> ResendResult:
        chat_id = self._resolve_chat_id(message)
        tg_chat_id = message.chat_id

        async with get_session() as session:
            chat, last_msg = await self._load_resend_target(session, chat_id)
            if not chat or not last_msg:
                await self._notify_missing_resend_target(
                    tg_chat_id,
                    has_chat=chat is not None,
                )
                return ResendResult(sent_message_ids=[], replaced_previous=False)

        for old_id in previous_response_ids:
            await self._messenger.delete_message(tg_chat_id, old_id)

        from mai_gram.messenger.telegram import build_inline_keyboard

        kb_buttons = [[("🔄 Regenerate", "regen")]]
        kb_buttons[0].append(("✂ Cut this & above", f"cut:{last_msg.id}"))
        action_kb = build_inline_keyboard(kb_buttons)

        reasoning = last_msg.reasoning if last_msg.reasoning else None
        sent_ids = await self._renderer._send_response(
            tg_chat_id,
            response_text=last_msg.content,
            response_reasoning=reasoning,
            show_reasoning=chat.show_reasoning,
            keyboard=action_kb,
        )

        if sent_ids:
            await self._messenger.send_message(
                OutgoingMessage(
                    text=f"✅ Resent last AI message ({len(sent_ids)} part(s)).",
                    chat_id=tg_chat_id,
                )
            )

        return ResendResult(sent_message_ids=sent_ids, replaced_previous=True)

    async def _notify_missing_resend_target(self, tg_chat_id: str, *, has_chat: bool) -> None:
        text = (
            "No assistant message found to resend."
            if has_chat
            else "No chat configured. Use /start first."
        )
        await self._messenger.send_message(
            OutgoingMessage(
                text=text,
                chat_id=tg_chat_id,
            )
        )

    @staticmethod
    async def _load_resend_target(
        session: AsyncSession,
        chat_id: str,
    ) -> tuple[Chat | None, Message | None]:
        chat = await ResendService._get_chat(session, chat_id)
        if not chat:
            return None, None
        result = await session.execute(
            select(Message)
            .where(
                Message.chat_id == chat_id,
                Message.role == "assistant",
                Message.content.isnot(None),
                Message.content != "",
            )
            .order_by(Message.id.desc())
            .limit(1)
        )
        return chat, result.scalar_one_or_none()

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
