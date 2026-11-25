"""Regenerate workflow built on top of the shared conversation executor."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.history_actions import HISTORY_ACTION_UNAVAILABLE
from mai_gram.bot.history_buttons import parse_message_id
from mai_gram.db.database import get_session
from mai_gram.db.history import (
    assistant_message_target,
    claim_retry,
    latest_history_ref,
    save_generated_history,
)
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
    from mai_gram.bot.conversation_executor import AssistantTurnRequest, ConversationExecutor
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
        self._active_chats: set[str] = set()

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
        target = target_ref if target_ref is not None else target_assistant_msg_id
        if target is not None or retry_ref is not None:
            if chat_id in self._active_chats:
                await self._notify_unavailable(message)
                return None
            self._active_chats.add(chat_id)
            try:
                return await self._handle_bound(message, chat_id, target, retry_ref)
            finally:
                self._active_chats.remove(chat_id)

        if not await self._preserve_or_prune_trailing_turns(chat_id):
            for message_id in previous_response_ids:
                await self._messenger.delete_message(message.chat_id, message_id)

        async with get_session() as session:
            return await self._execute_pending(session, message, chat_id)

    async def _claim_retry(self, message: IncomingMessage, chat_id: str, token: str) -> bool:
        confirmation = None
        if message.platform == "telegram":
            source = self._messenger.get_callback_source_message(message)
            if source is None or parse_message_id(source.message_id) is None:
                return False
            confirmation = (message.chat_id, source.message_id)

        async with get_session() as session:
            return await claim_retry(session, chat_id, token, confirmation)

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

    async def _handle_bound(
        self,
        message: IncomingMessage,
        chat_id: str,
        target_id: int | str | None,
        retry_ref: str | None,
    ) -> list[str] | None:
        if retry_ref is not None and not await self._claim_retry(message, chat_id, retry_ref):
            await self._notify_unavailable(message)
            return None

        async with get_session() as session:
            anchor_match = (
                assistant_message_target(chat_id, target_id)
                if target_id is not None
                else (Message.chat_id == chat_id) & (Message.history_ref == retry_ref)
            )
            state = (
                await session.execute(
                    select(Chat, Message, latest_history_ref(chat_id))
                    .select_from(Chat)
                    .join(Message, anchor_match)
                    .where(Chat.id == chat_id)
                )
            ).one_or_none()
            if state is None or (retry_ref is not None and state[2] != retry_ref):
                await self._notify_unavailable(message)
                return None

            chat, anchor, expected_tail = state
            expected_cut = chat.cut_above_message_id
            through_id = anchor.id - 1 if target_id is not None else anchor.id
            request = await self._build_bound_request(
                session, chat, message, through_id, targeted=target_id is not None
            )
            if request is None:
                return None

            return await self._generate_bound(
                request, message, target_id, anchor, expected_tail, expected_cut
            )

    async def _generate_bound(
        self,
        request: AssistantTurnRequest,
        message: IncomingMessage,
        target_id: int | str | None,
        anchor: Message,
        expected_tail: str,
        expected_cut: int | None,
    ) -> list[str] | None:
        stale = False

        async def persist_generated() -> bool:
            nonlocal stale
            async with get_session() as session:
                stale = not await save_generated_history(
                    session,
                    request.chat.id,
                    expected_tail=expected_tail,
                    expected_cut=expected_cut,
                    rows=request.message_store.staged,
                    target=anchor if target_id is not None else None,
                )
            return not stale

        await self._messenger.send_typing_indicator(message.chat_id)
        result = await self._conversation_executor.execute(
            replace(request, regenerate_target_id=target_id, persist_generated=persist_generated)
        )
        if stale:
            for message_id in dict.fromkeys(result.sent_message_ids):
                await self._messenger.delete_message(message.chat_id, message_id)
            await self._notify_unavailable(message)
            return None
        if target_id is not None and result.saved_message_id is None:
            return None
        return result.sent_message_ids

    async def _build_bound_request(
        self,
        session: AsyncSession,
        chat: Chat,
        message: IncomingMessage,
        through_id: int,
        *,
        targeted: bool,
    ) -> AssistantTurnRequest | None:
        store = MessageStore(session, through_message_id=through_id)
        recent = await store.get_recent(
            chat.id, limit=1, after_message_id=chat.cut_above_message_id if targeted else None
        )
        if not recent or recent[0].role not in ("user", "tool"):
            await self._report_missing_input(message.chat_id)
            return None

        request = await self._turn_builder.build_request(
            session,
            chat=chat,
            telegram_chat_id=message.chat_id,
            failure_log_message="Failed to regenerate response",
            current_time=datetime.now(timezone.utc),
            through_message_id=through_id,
        )
        if targeted and not any(item.role == "user" for item in request.llm_messages):
            await self._report_missing_input(message.chat_id)
            return None
        return request

    async def _regenerate_turn(
        self, session: AsyncSession, chat: Chat, message: IncomingMessage
    ) -> list[str] | None:
        recent = await MessageStore(session).get_recent(chat.id, limit=1)
        if not recent or recent[0].role not in ("user", "tool"):
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
        result = await self._conversation_executor.execute(request)
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
