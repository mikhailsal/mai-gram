"""High-level memory orchestrator.

Simplified version: stores messages and wiki entries without summarization.
Memory compression modules are kept in the codebase but not invoked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from mai_gram.db.models import KnowledgeEntry, Message
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore


class MemoryManager:
    """Delegates memory operations to message/wiki subsystems."""

    def __init__(
        self,
        message_store: MessageStore,
        wiki_store: WikiStore,
    ) -> None:
        self._message_store = message_store
        self._wiki_store = wiki_store

    async def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime | None = None,
        tool_calls: str | None = None,
        tool_call_id: str | None = None,
        timezone_name: str = "UTC",
    ) -> Message:
        return await self._message_store.save_message(
            chat_id,
            role,
            content,
            timestamp=timestamp,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            timezone_name=timezone_name,
        )

    async def get_recent(
        self,
        chat_id: str,
        *,
        limit: int = 50,
    ) -> list[Message]:
        return await self._message_store.get_recent(chat_id, limit=limit)

    async def search_messages(self, chat_id: str, query: str, *, limit: int = 20) -> list[Message]:
        return await self._message_store.search(chat_id, query, limit=limit)

    async def get_wiki_top(self, chat_id: str, *, limit: int = 20) -> list[KnowledgeEntry]:
        return await self._wiki_store.get_top_entries(chat_id, limit=limit)
