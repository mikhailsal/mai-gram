"""Regenerate workflow built on top of the shared conversation executor."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.conversation_executor import AssistantTurnRequest, ConversationExecutor
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import Settings
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.mcp_servers.manager import MCPManager
    from mai_gram.messenger.base import Messenger


class RegenerateService:
    """Own regenerate-specific persistence and execution steps."""

    def __init__(
        self,
        messenger: Messenger,
        llm_provider: LLMProvider,
        conversation_executor: ConversationExecutor,
        settings: Settings,
        *,
        resolve_chat_id: Callable[[IncomingMessage], str],
        build_mcp_manager: Callable[[Chat, MessageStore, WikiStore], MCPManager],
        memory_data_dir: str,
        wiki_context_limit: int,
        short_term_limit: int,
        test_mode: bool,
    ) -> None:
        self._messenger = messenger
        self._llm = llm_provider
        self._conversation_executor = conversation_executor
        self._settings = settings
        self._resolve_chat_id = resolve_chat_id
        self._build_mcp_manager = build_mcp_manager
        self._memory_data_dir = memory_data_dir
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._test_mode = test_mode

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
        has_tool_chain = False

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return []

            message_store = MessageStore(session)
            recent = await message_store.get_recent(chat.id, limit=20)
            trailing = self._get_trailing_assistant_chain(recent)
            has_tool_results = any(item.role == "tool" for item in trailing)
            has_tool_call_assistant = any(
                item.role == "assistant" and item.tool_calls for item in trailing
            )
            has_tool_chain = has_tool_results and has_tool_call_assistant

            if not has_tool_chain:
                for stored_message in trailing:
                    await session.delete(stored_message)
                await session.flush()

        if not has_tool_chain:
            for message_id in previous_response_ids:
                await self._messenger.delete_message(message.chat_id, message_id)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return []

            message_store = MessageStore(session)
            wiki_store = WikiStore(session, data_dir=self._memory_data_dir)
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

            prompt_builder = PromptBuilder(
                self._llm,
                message_store,
                wiki_store,
                wiki_context_limit=self._wiki_context_limit,
                short_term_limit=self._short_term_limit,
                test_mode=self._test_mode,
            )
            mcp_manager = self._build_mcp_manager(chat, message_store, wiki_store)
            llm_messages = await prompt_builder.build_context(
                chat,
                current_time=datetime.now(timezone.utc),
                send_datetime=chat.send_datetime,
                chat_timezone=chat.timezone,
                cut_above_message_id=chat.cut_above_message_id,
            )

            result = await self._conversation_executor.execute(
                AssistantTurnRequest(
                    chat=chat,
                    message_store=message_store,
                    mcp_manager=mcp_manager,
                    llm_messages=llm_messages,
                    telegram_chat_id=message.chat_id,
                    timezone_name=chat.timezone,
                    show_datetime=chat.send_datetime,
                    show_reasoning=chat.show_reasoning,
                    show_tool_calls=chat.show_tool_calls,
                    extra_params=self._settings.get_model_params(chat.llm_model),
                    failure_log_message="Failed to regenerate response",
                )
            )
            return result.sent_message_ids

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
