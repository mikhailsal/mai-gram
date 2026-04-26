"""Ordinary conversation workflow built on the shared assistant-turn executor."""

from __future__ import annotations

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

            await self._messenger.send_typing_indicator(message.chat_id)
            request = await self._turn_builder.save_user_message_and_build_request(
                session,
                chat=chat,
                user_text=message.text,
                telegram_chat_id=message.chat_id,
                failure_log_message="Failed to generate response",
            )
            result = await self._conversation_executor.execute(request)
            return result.sent_message_ids

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
