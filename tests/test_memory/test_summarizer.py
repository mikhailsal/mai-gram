"""Tests for MemorySummarizer reconsolidation flows."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from mai_gram.memory.summaries import StoredSummary
from mai_gram.memory.summarizer import MemorySummarizer
from mai_gram.memory.summarizer_support import (
    backfill_missing_summaries,
    generate_monthly_for_period,
    generate_weekly_for_period,
    reconsolidate_periods,
    trigger_daily_if_needed,
)

if TYPE_CHECKING:
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore
    from mai_gram.memory.summaries import SummaryStore


class _RecordingSummarizer(MemorySummarizer):
    def __init__(
        self,
        message_store: MessageStore,
        summary_store: SummaryStore,
        llm_provider: LLMProvider,
        daily_results: list[str | None] | None = None,
        weekly_results: list[str | None] | None = None,
        *,
        summary_threshold: int = 20,
        model: str | None = None,
        wiki_store: WikiStore | None = None,
        companion_name: str | None = None,
        companion_model: str | None = None,
        wiki_context_limit: int = 10,
        recent_summary_days: int = 3,
    ) -> None:
        super().__init__(
            message_store,
            summary_store,
            llm_provider,
            summary_threshold=summary_threshold,
            model=model,
            wiki_store=wiki_store,
            companion_name=companion_name,
            companion_model=companion_model,
            wiki_context_limit=wiki_context_limit,
            recent_summary_days=recent_summary_days,
        )
        self._daily_results = iter(daily_results or [])
        self._weekly_results = iter(weekly_results or [])
        self.daily_calls: list[tuple[str, date]] = []
        self.weekly_calls: list[tuple[str, int, int]] = []

    async def generate_daily_summary(
        self,
        companion_id: str,
        target_date: date,
    ) -> str | None:
        self.daily_calls.append((companion_id, target_date))
        return next(self._daily_results)

    async def generate_weekly_summary(
        self,
        companion_id: str,
        year: int,
        week: int,
    ) -> str | None:
        self.weekly_calls.append((companion_id, year, week))
        return next(self._weekly_results)


@pytest.mark.asyncio
async def test_reconsolidate_daily_from_saves_versions_and_reports_progress() -> None:
    message_store = MagicMock()
    summary_store = MagicMock()
    llm = MagicMock()
    summarizer = _RecordingSummarizer(
        message_store,
        summary_store,
        llm,
        daily_results=["new day 1", None],
    )

    first_day = date(2024, 1, 1)
    second_day = date(2024, 1, 2)
    summary_store.list_dailies.return_value = [second_day, first_day]
    summary_store.get_daily.side_effect = [
        StoredSummary("daily", first_day.isoformat(), "old day 1"),
        StoredSummary("daily", second_day.isoformat(), "old day 2"),
    ]
    progress = MagicMock()

    result = await summarizer.reconsolidate_daily_from(
        "companion",
        first_day,
        until_date=second_day,
        on_progress=progress,
    )

    assert result == [(first_day, "new day 1")]
    assert summarizer.daily_calls == [
        ("companion", first_day),
        ("companion", second_day),
    ]
    assert summary_store.save_version.call_args_list == [
        call("companion", "daily", first_day.isoformat(), "old day 1"),
        call("companion", "daily", second_day.isoformat(), "old day 2"),
    ]
    assert progress.call_args_list == [
        call(first_day.isoformat(), "processing"),
        call(first_day.isoformat(), "done"),
        call(second_day.isoformat(), "processing"),
        call(second_day.isoformat(), "skipped (no messages)"),
    ]


@pytest.mark.asyncio
async def test_reconsolidate_weekly_from_parses_periods_in_order() -> None:
    message_store = MagicMock()
    summary_store = MagicMock()
    llm = MagicMock()
    summarizer = _RecordingSummarizer(
        message_store,
        summary_store,
        llm,
        weekly_results=["week 1", "week 2", "week 3"],
    )

    summary_store.list_weeklies.return_value = ["2024-W03", "2024-W01", "2024-W02"]
    summary_store.get_weekly.side_effect = [
        StoredSummary("weekly", "2024-W01", "old week 1"),
        StoredSummary("weekly", "2024-W02", "old week 2"),
        StoredSummary("weekly", "2024-W03", "old week 3"),
    ]

    result = await summarizer.reconsolidate_weekly_from(
        "companion",
        "2024-W01",
        until_period="2024-W03",
    )

    assert result == [
        ("2024-W01", "week 1"),
        ("2024-W02", "week 2"),
        ("2024-W03", "week 3"),
    ]
    assert summarizer.weekly_calls == [
        ("companion", 2024, 1),
        ("companion", 2024, 2),
        ("companion", 2024, 3),
    ]
    assert summary_store.save_version.call_args_list == [
        call("companion", "weekly", "2024-W01", "old week 1"),
        call("companion", "weekly", "2024-W02", "old week 2"),
        call("companion", "weekly", "2024-W03", "old week 3"),
    ]


@pytest.mark.asyncio
async def test_trigger_daily_if_needed_only_runs_on_threshold_multiple() -> None:
    message_store = MagicMock()
    message_store.get_messages_for_date = AsyncMock(side_effect=[[1, 2], [1, 2, 3, 4]])
    generate_daily_summary = AsyncMock(return_value="summary")

    skipped = await trigger_daily_if_needed(
        message_store,
        4,
        generate_daily_summary,
        "companion",
        target_date=date(2024, 1, 1),
    )
    triggered = await trigger_daily_if_needed(
        message_store,
        4,
        generate_daily_summary,
        "companion",
        target_date=date(2024, 1, 2),
    )

    assert skipped is False
    assert triggered is True
    generate_daily_summary.assert_awaited_once_with("companion", date(2024, 1, 2))


@pytest.mark.asyncio
async def test_backfill_missing_summaries_returns_only_generated_dates() -> None:
    message_store = MagicMock()
    message_store.get_dates_with_messages = AsyncMock(
        return_value=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    )
    summary_store = MagicMock()
    summary_store.list_dailies.return_value = [date(2024, 1, 2)]
    generate_daily_summary = AsyncMock(side_effect=["summary 1", None])

    result = await backfill_missing_summaries(
        message_store,
        summary_store,
        generate_daily_summary,
        "companion",
        today=date(2024, 1, 4),
    )

    assert result == [date(2024, 1, 1)]
    generate_daily_summary.assert_has_awaits(
        [call("companion", date(2024, 1, 1)), call("companion", date(2024, 1, 3))]
    )


@pytest.mark.asyncio
async def test_reconsolidate_periods_saves_versions_and_reports_progress() -> None:
    summary_store = MagicMock()
    progress = MagicMock()
    regenerate = AsyncMock(side_effect=["new 1", None])
    current_one = StoredSummary("weekly", "2024-W01", "old 1")
    current_two = StoredSummary("weekly", "2024-W02", "old 2")

    def get_current(_companion_id: str, period: str) -> StoredSummary:
        return current_one if period == "2024-W01" else current_two

    result = await reconsolidate_periods(
        summary_store,
        "companion",
        ["2024-W01", "2024-W02"],
        summary_type="weekly",
        get_current=get_current,
        regenerate=regenerate,
        format_period=lambda period: period,
        skipped_status="skipped",
        on_progress=progress,
    )

    assert result == [("2024-W01", "new 1")]
    assert summary_store.save_version.call_args_list == [
        call("companion", "weekly", "2024-W01", "old 1"),
        call("companion", "weekly", "2024-W02", "old 2"),
    ]
    assert progress.call_args_list == [
        call("2024-W01", "processing"),
        call("2024-W01", "done"),
        call("2024-W02", "processing"),
        call("2024-W02", "skipped"),
    ]


@pytest.mark.asyncio
async def test_period_helpers_delegate_to_summary_generators() -> None:
    generate_weekly_summary = AsyncMock(return_value="weekly")
    generate_monthly_summary = AsyncMock(return_value="monthly")

    weekly = await generate_weekly_for_period(generate_weekly_summary, "companion", "2024-W03")
    monthly = await generate_monthly_for_period(generate_monthly_summary, "companion", "2024-07")

    assert weekly == "weekly"
    assert monthly == "monthly"
    generate_weekly_summary.assert_awaited_once_with("companion", 2024, 3)
    generate_monthly_summary.assert_awaited_once_with("companion", 2024, 7)
