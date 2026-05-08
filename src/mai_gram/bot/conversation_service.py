"""Ordinary conversation workflow built on the shared assistant-turn executor."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
    from mai_gram.bot.conversation_executor import ConversationExecutor
    from mai_gram.messenger.base import IncomingMessage, Messenger

logger = logging.getLogger(__name__)


class ConversationService:
    """Handle normal incoming user messages for an existing chat."""

    def __init__(
        self,
        messenger: Messenger,
        conversation_executor: ConversationExecutor,
        *,
        turn_builder: AssistantTurnBuilder,
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._conversation_executor = conversation_executor
        self._turn_builder = turn_builder
        self._resolve_chat_id = resolve_chat_id

    async def handle_message(self, message: IncomingMessage) -> list[str]:
        chat_id = self._resolve_chat_id(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No chat configured. Use /start to set one up.",
                        chat_id=message.chat_id,
                    )
                )
                return []

            image_urls = await self._download_photo(message)

            await self._messenger.send_typing_indicator(message.chat_id)
            user_text = message.text or ("What's in this image?" if image_urls else "")
            request = await self._turn_builder.save_user_message_and_build_request(
                session,
                chat=chat,
                user_text=user_text,
                telegram_chat_id=message.chat_id,
                failure_log_message="Failed to generate response",
                image_urls=image_urls,
            )
            result = await self._conversation_executor.execute(request)
            return result.sent_message_ids

    async def _download_photo(self, message: IncomingMessage) -> list[str] | None:
        """Download the attached photo and return a base64 data-URI list."""
        if not message.photo_file_id:
            return None
        try:
            photo_bytes = await self._messenger.download_file(message.photo_file_id)
            b64 = base64.b64encode(photo_bytes).decode("ascii")
            return [f"data:image/jpeg;base64,{b64}"]
        except Exception:
            logger.exception("Failed to download photo %s", message.photo_file_id)
            return None

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
