"""Tests for ForgettingEngine."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from mai_companion.memory.forgetting import ForgettingEngine
from mai_companion.memory.summaries import SummaryStore
from mai_companion.memory.summarizer import MemorySummarizer


class _MockSummarizer:
    def __init__(self) -> None:
        self.weekly_calls: list[tuple[str, int, int]] = []
        self.monthly_calls: list[tuple[str, int, int]] = []

    async def generate_weekly_summary(self, companion_id: str, year: int, week: int) -> str:
        self.weekly_calls.append((companion_id, year, week))
        return "weekly"

    async def generate_monthly_summary(self, companion_id: str, year: int, month: int) -> str:
        self.monthly_calls.append((companion_id, year, month))
        return "monthly"


class TestForgettingEngine:
    async def test_weekly_consolidation(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        today = date(2026, 2, 20)

        # Use older dates to ensure we cross the "end of week + 7 days" threshold.
        # Week ending Feb 8 is > 7 days ago.
        for day_offset in range(18, 11, -1):  # e.g. Feb 2 to Feb 9
            day = today - timedelta(days=day_offset)
            store.save_daily(companion_id, day, f"daily {day.isoformat()}")

        await engine.run_forgetting_cycle(companion_id, today=today)

        assert summarizer.weekly_calls
        remaining_dailies = store.list_dailies(companion_id)
        # Verify consolidation happened (dailies gone)
        assert not remaining_dailies

    async def test_weekly_consolidation_skips_recent(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        today = date(2026, 2, 20)
        store.save_daily(companion_id, today - timedelta(days=3), "recent")

        await engine.run_forgetting_cycle(companion_id, today=today)

        assert summarizer.weekly_calls == []
        assert store.list_dailies(companion_id) == [today - timedelta(days=3)]

    async def test_weekly_consolidation_regenerates_even_if_exists(self, tmp_path: Path) -> None:
        """Verify that existing weekly summaries are regenerated to include late-arriving dailies."""
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        # Use Feb 2, 2026 (Monday of ISO week 6, which ends Sunday Feb 8)
        # Run on Feb 16 so the week is fully stale (Feb 16 > Feb 8 + 7 = Feb 15)
        old_day = date(2026, 2, 2)
        today = date(2026, 2, 16)
        iso_year, iso_week, _ = old_day.isocalendar()
        store.save_daily(companion_id, old_day, "old daily content")
        store.save_weekly(companion_id, f"{iso_year}-W{iso_week:02d}", "pre-existing weekly")

        await engine.run_forgetting_cycle(companion_id, today=today)

        # The weekly should be regenerated (to include all dailies), even though one existed
        assert summarizer.weekly_calls == [(companion_id, iso_year, iso_week)]
        # The daily should be deleted after consolidation
        assert old_day not in store.list_dailies(companion_id)

    async def test_monthly_consolidation(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        today = date(2026, 3, 15)
        store.save_weekly(companion_id, "2026-W01", "w1")
        store.save_weekly(companion_id, "2026-W02", "w2")

        await engine.run_forgetting_cycle(companion_id, today=today)

        assert summarizer.monthly_calls
        assert not store.list_weeklies(companion_id)

    async def test_monthly_consolidation_skips_recent(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        today = date(2026, 3, 15)
        store.save_weekly(companion_id, "2026-W10", "recent")

        await engine.run_forgetting_cycle(companion_id, today=today)

        assert summarizer.monthly_calls == []
        assert store.list_weeklies(companion_id) == ["2026-W10"]

    async def test_full_forgetting_cycle(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-1"
        today = date(2026, 3, 15)

        for day_offset in range(15, 8, -1):
            d = today - timedelta(days=day_offset)
            store.save_daily(companion_id, d, f"daily {d}")
        store.save_weekly(companion_id, "2026-W01", "old week")

        await engine.run_forgetting_cycle(companion_id, today=today)

    async def test_regression_incremental_consolidation_data_loss(self, tmp_path: Path) -> None:
        """
        Regression test for bug where incremental daily consolidation caused data loss.
        Scenario:
        1. Week 1 has dailies: Mon, Tue.
        2. Friday of Week 2: Mon is > 7 days old. Tue is NOT.
        3. Forgetting runs: OLD BUG would consolidate Mon and delete it. NEW LOGIC waits.
        4. Saturday of Week 2: Tue is > 7 days old.
        5. Forgetting runs: OLD BUG would see existing summary and delete Tue without update.
        """
        store = SummaryStore(data_dir=tmp_path)
        summarizer = _MockSummarizer()
        engine = ForgettingEngine(store, summarizer)  # type: ignore[arg-type]
        companion_id = "comp-regress"

        # Dates
        monday = date(2026, 2, 2)
        tuesday = date(2026, 2, 3)
        
        store.save_daily(companion_id, monday, "Monday content")
        store.save_daily(companion_id, tuesday, "Tuesday content")
        
        # Run 1: 2026-02-10 (Monday + 8 days)
        # Week 2026-W06 ends Sunday Feb 8.
        # Threshold: today > week_end + 7 => Feb 10 > Feb 15? False.
        await engine.run_forgetting_cycle(companion_id, today=date(2026, 2, 10))
        
        # Verify NO consolidation happened yet (safer logic)
        assert not summarizer.weekly_calls
        assert monday in store.list_dailies(companion_id)
        assert tuesday in store.list_dailies(companion_id)
        
        # Run 2: 2026-02-16 (Monday + 14 days)
        # Today > Feb 15. Ready!
        await engine.run_forgetting_cycle(companion_id, today=date(2026, 2, 16))
        
        # Verify consolidation happened for the WHOLE week
        assert len(summarizer.weekly_calls) == 1
        assert monday not in store.list_dailies(companion_id)
        assert tuesday not in store.list_dailies(companion_id)

