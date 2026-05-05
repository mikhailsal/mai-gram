"""Build typed assistant-turn requests for conversation workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mai_gram.bot.conversation_executor import AssistantTurnRequest
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import Settings
    from mai_gram.db.models import Chat
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.mcp_servers.manager import MCPManager


class AssistantTurnBuilder:
    """Build `AssistantTurnRequest` objects from persisted chat state."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        *,
        build_mcp_manager: Callable[[Chat, MessageStore, WikiStore], MCPManager],
        memory_data_dir: str,
        wiki_context_limit: int,
        short_term_limit: int,
        test_mode: bool,
    ) -> None:
        self._llm = llm_provider
        self._settings = settings
        self._build_mcp_manager = build_mcp_manager
        self._memory_data_dir = memory_data_dir
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._test_mode = test_mode

    async def save_user_message_and_build_request(
        self,
        session: AsyncSession,
        *,
        chat: Chat,
        user_text: str,
        telegram_chat_id: str,
        failure_log_message: str,
        current_time: datetime | None = None,
    ) -> AssistantTurnRequest:
        now = current_time or datetime.now(timezone.utc)
        message_store, wiki_store = self._make_stores(session)
        await message_store.save_message(
            chat.id,
            "user",
            user_text,
            timestamp=now,
            timezone_name=chat.timezone,
            show_datetime=chat.send_datetime,
        )
        return await self._build_request(
            chat=chat,
            message_store=message_store,
            wiki_store=wiki_store,
            telegram_chat_id=telegram_chat_id,
            failure_log_message=failure_log_message,
            current_time=now,
        )

    async def build_request(
        self,
        session: AsyncSession,
        *,
        chat: Chat,
        telegram_chat_id: str,
        failure_log_message: str,
        current_time: datetime | None = None,
    ) -> AssistantTurnRequest:
        now = current_time or datetime.now(timezone.utc)
        message_store, wiki_store = self._make_stores(session)
        return await self._build_request(
            chat=chat,
            message_store=message_store,
            wiki_store=wiki_store,
            telegram_chat_id=telegram_chat_id,
            failure_log_message=failure_log_message,
            current_time=now,
        )

    def _make_stores(self, session: AsyncSession) -> tuple[MessageStore, WikiStore]:
        return MessageStore(session), WikiStore(session, data_dir=self._memory_data_dir)

    async def _build_request(
        self,
        *,
        chat: Chat,
        message_store: MessageStore,
        wiki_store: WikiStore,
        telegram_chat_id: str,
        failure_log_message: str,
        current_time: datetime,
    ) -> AssistantTurnRequest:
        model_key = chat.llm_model
        prompt_builder = PromptBuilder(
            self._llm,
            message_store,
            wiki_store,
            wiki_context_limit=self._wiki_context_limit,
            short_term_limit=self._short_term_limit,
            max_context_tokens=self._settings.get_max_context_tokens(model_key),
            test_mode=self._test_mode,
        )
        await wiki_store.sync_from_disk(chat.id)
        mcp_manager = self._build_mcp_manager(chat, message_store, wiki_store)
        llm_messages = await prompt_builder.build_context(
            chat,
            current_time=current_time,
            send_datetime=chat.send_datetime,
            chat_timezone=chat.timezone,
            cut_above_message_id=chat.cut_above_message_id,
        )
        return AssistantTurnRequest(
            chat=chat,
            message_store=message_store,
            mcp_manager=mcp_manager,
            llm_messages=llm_messages,
            telegram_chat_id=telegram_chat_id,
            timezone_name=chat.timezone,
            show_datetime=chat.send_datetime,
            show_reasoning=chat.show_reasoning,
            show_tool_calls=chat.show_tool_calls,
            extra_params=self._settings.get_model_params(model_key),
            failure_log_message=failure_log_message,
            resolved_model=self._settings.get_model_id(model_key),
        )
