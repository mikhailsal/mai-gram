"""Regenerate workflow built on top of the shared conversation executor."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
    from mai_gram.bot.conversation_executor import ConversationExecutor
    from mai_gram.messenger.base import Messenger


class RegenerateService:
    """Own regenerate-specific persistence and execution steps."""

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

    async def handle_regenerate(
        self,
        message: IncomingMessage,
        *,
        previous_response_ids: list[str],
    ) -> list[str]:
        """Delete or preserve prior output as needed, then re-run the turn."""
        if not message.callback_data:
            return []

        chat_id = self._resolve_chat_id(message)
        has_tool_chain = await self._preserve_or_prune_trailing_turns(chat_id)

        if not has_tool_chain:
            for message_id in previous_response_ids:
                await self._messenger.delete_message(message.chat_id, message_id)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return []

            message_store = MessageStore(session)
            recent = await message_store.get_recent(chat.id, limit=1)
            last_role = recent[0].role if recent else None
            if last_role not in ("user", "tool"):
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="Cannot regenerate: no user message found.",
                        chat_id=message.chat_id,
                    )
                )
                return []

            await self._messenger.send_typing_indicator(message.chat_id)

            request = await self._turn_builder.build_request(
                session,
                chat=chat,
                telegram_chat_id=message.chat_id,
                failure_log_message="Failed to regenerate response",
                current_time=datetime.now(timezone.utc),
            )

            result = await self._conversation_executor.execute(request)
            return result.sent_message_ids

    async def _preserve_or_prune_trailing_turns(self, chat_id: str) -> bool:
        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return False

            message_store = MessageStore(session)
            recent = await message_store.get_recent(chat.id, limit=20)
            trailing = self._get_trailing_assistant_chain(recent)
            has_tool_chain = self._has_tool_chain(trailing)
            if has_tool_chain:
                return True

            for stored_message in trailing:
                await session.delete(stored_message)
            await session.flush()
            return False

    @staticmethod
    def _has_tool_chain(trailing: list[Message]) -> bool:
        has_tool_results = any(item.role == "tool" for item in trailing)
        has_tool_call_assistant = any(
            item.role == "assistant" and item.tool_calls for item in trailing
        )
        return has_tool_results and has_tool_call_assistant

    @staticmethod
    def _get_trailing_assistant_chain(recent: list[Message]) -> list[Message]:
        trailing: list[Message] = []
        for message in reversed(sorted(recent, key=lambda item: item.id)):
            if message.role in ("assistant", "tool"):
                trailing.append(message)
            else:
                break
        return trailing

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
