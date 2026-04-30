"""Build prompt preview data for CLI and other adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.mcp_manager_factory import MCPManagerFactory
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import Settings
    from mai_gram.llm.provider import ChatMessage, LLMProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.mcp_servers.manager import RegisteredTool


@dataclass(frozen=True, slots=True)
class PromptPreview:
    context: list[ChatMessage]
    tools: list[RegisteredTool]
    token_count: int


class PromptPreviewService:
    """Assemble prompt preview data with the same tool filtering used in bot flows."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        *,
        memory_data_dir: str,
        wiki_context_limit: int = 20,
        short_term_limit: int = 500,
        test_mode: bool = True,
        external_mcp_pool: ExternalMCPPool | None = None,
        mcp_manager_factory: MCPManagerFactory | None = None,
    ) -> None:
        self._llm = llm_provider
        self._memory_data_dir = memory_data_dir
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._test_mode = test_mode
        self._mcp_manager_factory = mcp_manager_factory or MCPManagerFactory(
            settings,
            external_mcp_pool=external_mcp_pool,
        )

    async def build_preview(
        self,
        session: AsyncSession,
        *,
        chat_id: str,
    ) -> PromptPreview:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if chat is None:
            raise LookupError(chat_id)

        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=self._memory_data_dir)
        prompt_builder = PromptBuilder(
            self._llm,
            message_store,
            wiki_store,
            wiki_context_limit=self._wiki_context_limit,
            short_term_limit=self._short_term_limit,
            test_mode=self._test_mode,
        )
        await wiki_store.sync_from_disk(chat.id)
        context = await prompt_builder.build_context(
            chat,
            send_datetime=chat.send_datetime,
            chat_timezone=chat.timezone,
            cut_above_message_id=chat.cut_above_message_id,
        )
        mcp_manager = self._mcp_manager_factory.build_manager(chat, message_store, wiki_store)
        tools = await mcp_manager.list_all_tools()
        token_count = await self._llm.count_tokens(context)
        return PromptPreview(
            context=context,
            tools=tools,
            token_count=token_count,
        )
