"""History-related actions such as previewing and cutting prompt context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.messenger.base import Messenger


class HistoryActions:
    """Own history preview and cut-above behavior."""

    def __init__(
        self,
        messenger: Messenger,
        *,
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._resolve_chat_id = resolve_chat_id

    async def get_message_preview(self, db_message_id: int, max_len: int = 80) -> str:
        """Fetch a truncated preview of a stored message by its DB id."""
        async with get_session() as session:
            result = await session.execute(select(Message).where(Message.id == db_message_id))
            message = result.scalar_one_or_none()
            if not message or not message.content:
                return ""

            text = message.content.replace("\n", " ").strip()
            if len(text) > max_len:
                text = text[:max_len] + "..."
            return text

    async def handle_cut_above(
        self,
        message: IncomingMessage,
        db_message_id: int,
        *,
        original_tg_msg_id: str = "",
        cached_original: tuple[str, str | None] | None = None,
    ) -> None:
        """Set the cut-above point so prior messages are excluded from prompts."""
        chat_id = self._resolve_chat_id(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)
            cut_count = 0
            for stored_message in await message_store.get_all(chat.id):
                if stored_message.id <= db_message_id:
                    cut_count += 1

            chat.cut_above_message_id = db_message_id
            await session.commit()

        if original_tg_msg_id and cached_original is not None:
            original_html, original_parse = cached_original
            badge = "\u2702\ufe0f <i>[this and above are hidden from the AI]</i>"
            if original_parse == "html":
                marked_text = f"{badge}\n\n{original_html}"
            else:
                import html as _html

                marked_text = f"{badge}\n\n{_html.escape(original_html)}"
            if len(marked_text) > 4000:
                marked_text = marked_text[:4000] + "..."
            await self._messenger.edit_message(
                message.chat_id,
                original_tg_msg_id,
                marked_text,
                parse_mode="html",
            )

        footer_lines = ["\u2500" * 20, "\u2702\ufe0f History cut applied"]
        if cut_count > 0:
            footer_lines.append(f"\U0001f4e6 {cut_count} message(s) hidden from AI")
        footer_lines.append("\u2139\ufe0f Hidden messages are still searchable via tools")
        footer_lines.append("\u2500" * 20)
        await self._messenger.send_message(
            OutgoingMessage(
                text="\n".join(footer_lines),
                chat_id=message.chat_id,
            )
        )

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
