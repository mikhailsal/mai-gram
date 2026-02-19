"""High-level memory orchestrator."""

from __future__ import annotations

from datetime import date, datetime

from mai_companion.db.models import KnowledgeEntry, Message
from mai_companion.memory.forgetting import ForgettingEngine
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import StoredSummary, SummaryStore
from mai_companion.memory.summarizer import MemorySummarizer


class MemoryManager:
    """Delegates memory operations to message/wiki/summary subsystems."""

    def __init__(
        self,
        message_store: MessageStore,
        summary_store: SummaryStore,
        wiki_store: WikiStore,
        summarizer: MemorySummarizer,
        forgetting_engine: ForgettingEngine,
    ) -> None:
        self._message_store = message_store
        self._summary_store = summary_store
        self._wiki_store = wiki_store
        self._summarizer = summarizer
        self._forgetting_engine = forgetting_engine

    async def save_message(
        self,
        companion_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime | None = None,
        is_proactive: bool = False,
        trigger_summary: bool = False,
    ) -> Message:
        message = await self._message_store.save_message(
            companion_id,
            role,
            content,
            timestamp=timestamp,
            is_proactive=is_proactive,
        )
        if trigger_summary:
            day = (timestamp or message.timestamp).date()
            await self._summarizer.trigger_daily_if_needed(companion_id, target_date=day)
        return message

    async def get_short_term(self, companion_id: str, *, limit: int = 30) -> list[Message]:
        return await self._message_store.get_short_term(companion_id, limit=limit)

    async def search_messages(self, companion_id: str, query: str, *, limit: int = 20) -> list[Message]:
        return await self._message_store.search(companion_id, query, limit=limit)

    def get_all_summaries(self, companion_id: str) -> list[StoredSummary]:
        return self._summary_store.get_all_summaries(companion_id)

    async def get_wiki_top(self, companion_id: str, *, limit: int = 20) -> list[KnowledgeEntry]:
        return await self._wiki_store.get_top_entries(companion_id, limit=limit)

    async def trigger_daily_summary(self, companion_id: str, target_date: date) -> str | None:
        return await self._summarizer.generate_daily_summary(companion_id, target_date)

    async def run_forgetting_cycle(self, companion_id: str, *, today: date | None = None) -> None:
        await self._forgetting_engine.run_forgetting_cycle(companion_id, today=today)
