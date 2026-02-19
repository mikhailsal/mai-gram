"""Tests for MemoryManager delegation."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

from mai_companion.memory.manager import MemoryManager


class TestMemoryManager:
    async def test_save_message_delegates(self) -> None:
        message_store = MagicMock()
        message_store.save_message = AsyncMock(return_value=MagicMock(timestamp=date.today()))
        summary_store = MagicMock()
        wiki_store = MagicMock()
        summarizer = MagicMock()
        summarizer.trigger_daily_if_needed = AsyncMock(return_value=False)
        forgetting = MagicMock()
        manager = MemoryManager(message_store, summary_store, wiki_store, summarizer, forgetting)

        await manager.save_message("comp", "user", "hello")

        message_store.save_message.assert_awaited_once()

    async def test_get_short_term_delegates(self) -> None:
        message_store = MagicMock()
        message_store.get_short_term = AsyncMock(return_value=[])
        manager = MemoryManager(message_store, MagicMock(), MagicMock(), MagicMock(), MagicMock())

        await manager.get_short_term("comp")

        message_store.get_short_term.assert_awaited_once()

    def test_get_all_summaries_delegates(self) -> None:
        summary_store = MagicMock()
        summary_store.get_all_summaries.return_value = ["x"]
        manager = MemoryManager(MagicMock(), summary_store, MagicMock(), MagicMock(), MagicMock())

        result = manager.get_all_summaries("comp")

        assert result == ["x"]
        summary_store.get_all_summaries.assert_called_once_with("comp")

    async def test_get_wiki_top_delegates(self) -> None:
        wiki_store = MagicMock()
        wiki_store.get_top_entries = AsyncMock(return_value=[])
        manager = MemoryManager(MagicMock(), MagicMock(), wiki_store, MagicMock(), MagicMock())

        await manager.get_wiki_top("comp")

        wiki_store.get_top_entries.assert_awaited_once()

    async def test_trigger_daily_summary(self) -> None:
        summarizer = MagicMock()
        summarizer.generate_daily_summary = AsyncMock(return_value="summary")
        manager = MemoryManager(MagicMock(), MagicMock(), MagicMock(), summarizer, MagicMock())

        result = await manager.trigger_daily_summary("comp", date(2026, 2, 14))

        assert result == "summary"
        summarizer.generate_daily_summary.assert_awaited_once()

    async def test_run_forgetting_cycle(self) -> None:
        forgetting = MagicMock()
        forgetting.run_forgetting_cycle = AsyncMock(return_value=None)
        manager = MemoryManager(MagicMock(), MagicMock(), MagicMock(), MagicMock(), forgetting)

        await manager.run_forgetting_cycle("comp")

        forgetting.run_forgetting_cycle.assert_awaited_once()
