"""Regenerate workflow built on top of the shared conversation executor."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.history_actions import HISTORY_ACTION_UNAVAILABLE
from mai_gram.bot.history_buttons import parse_message_id
from mai_gram.db.database import get_session
from mai_gram.db.history import claim_retry, rollback_history
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
        self._active_targets: set[tuple[str, int | str]] = set()

    async def handle_regenerate(
        self,
        message: IncomingMessage,
        *,
        previous_response_ids: list[str],
        target_assistant_msg_id: int | None = None,
        target_ref: str | None = None,
        retry_ref: str | None = None,
    ) -> list[str] | None:
        """Regenerate a selected response or retry a selected pending turn."""
        if not message.callback_data:
            return []

        chat_id = self._resolve_chat_id(message)
        if retry_ref is not None:
            return await self._retry_pending(message, chat_id, retry_ref)

        target = target_ref if target_ref is not None else target_assistant_msg_id
        if target is not None:
            return await self._handle_targeted(message, chat_id, target)
        if not await self._preserve_or_prune_trailing_turns(chat_id):
            for message_id in previous_response_ids:
                await self._messenger.delete_message(message.chat_id, message_id)

        async with get_session() as session:
            return await self._execute_pending(session, message, chat_id)

    async def _retry_pending(
        self, message: IncomingMessage, chat_id: str, token: str
    ) -> list[str] | None:
        confirmation = None
        if message.platform == "telegram":
            source = self._messenger.get_callback_source_message(message)
            if source is None or parse_message_id(source.message_id) is None:
                await self._notify_unavailable(message)
                return None
            confirmation = (message.chat_id, source.message_id)

        async with get_session() as session:
            if await claim_retry(session, chat_id, token, confirmation):
                return await self._execute_pending(session, message, chat_id)
        await self._notify_unavailable(message)
        return None

    async def _notify_unavailable(self, message: IncomingMessage) -> None:
        await self._messenger.send_message(
            OutgoingMessage(text=HISTORY_ACTION_UNAVAILABLE, chat_id=message.chat_id)
        )

    async def _execute_pending(
        self, session: AsyncSession, message: IncomingMessage, chat_id: str
    ) -> list[str]:
        chat = await self._get_chat(session, chat_id)
        if chat is None:
            return []
        return await self._regenerate_turn(session, chat, message) or []

    async def _handle_targeted(
        self, message: IncomingMessage, chat_id: str, target_id: int | str
    ) -> list[str] | None:
        target = (chat_id, target_id)
        if target not in self._active_targets:
            self._active_targets.add(target)
            try:
                async with get_session() as session:
                    # Claim the exact target with the first write. The prompt and tools see
                    # the remaining prefix. Commit the rewrite after final reply delivery.
                    if await rollback_history(session, chat_id, target_id):
                        chat = await self._get_chat(session, chat_id)
                        if chat is not None:
                            return await self._regenerate_turn(
                                session, chat, message, target_id=target_id
                            )
                        await session.rollback()
            finally:
                self._active_targets.remove(target)

        await self._notify_unavailable(message)
        return None

    async def _regenerate_turn(
        self,
        session: AsyncSession,
        chat: Chat,
        message: IncomingMessage,
        *,
        target_id: int | str | None = None,
    ) -> list[str] | None:
        targeted = target_id is not None
        message_store = MessageStore(session)
        recent = await message_store.get_recent(
            chat.id, limit=1, after_message_id=chat.cut_above_message_id if targeted else None
        )
        if not recent or recent[0].role not in ("user", "tool"):
            if targeted:
                await session.rollback()
            await self._report_missing_input(message.chat_id)
            return None

        await self._messenger.send_typing_indicator(message.chat_id)
        request = await self._turn_builder.build_request(
            session,
            chat=chat,
            telegram_chat_id=message.chat_id,
            failure_log_message="Failed to regenerate response",
            current_time=datetime.now(timezone.utc),
        )
        if targeted:
            # A cut can also leave tool results without their preceding user input.
            if not any(item.role == "user" for item in request.llm_messages):
                await session.rollback()
                await self._report_missing_input(message.chat_id)
                return None
            request = replace(request, regenerate_target_id=target_id)

        result = await self._conversation_executor.execute(request)
        if targeted and result.saved_message_id is None:
            await session.rollback()
            return None
        return result.sent_message_ids

    async def _report_missing_input(self, telegram_chat_id: str) -> None:
        await self._messenger.send_message(
            OutgoingMessage(
                text="Cannot regenerate: no user message found.", chat_id=telegram_chat_id
            )
        )

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
        for message in sorted(recent, key=lambda item: item.id, reverse=True):
            if message.role in ("assistant", "tool"):
                trailing.append(message)
            else:
                break
        return trailing

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
