"""Tests for SummaryStore."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from mai_companion.memory.summaries import SummaryStore


class TestSummaryStore:
    """SummaryStore behavior."""

    def test_save_daily(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        path = store.save_daily("comp-sum", date(2026, 2, 14), "Summary text")

        assert path == tmp_path / "comp-sum" / "summaries" / "daily" / "2026-02-14.md"
        assert path.read_text(encoding="utf-8") == "Summary text"

    def test_save_weekly(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        path = store.save_weekly("comp-sum", "2026-W07", "Weekly summary")

        assert path == tmp_path / "comp-sum" / "summaries" / "weekly" / "2026-W07.md"
        assert path.read_text(encoding="utf-8") == "Weekly summary"

    def test_save_monthly(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        path = store.save_monthly("comp-sum", "2026-01", "Monthly summary")

        assert path == tmp_path / "comp-sum" / "summaries" / "monthly" / "2026-01.md"
        assert path.read_text(encoding="utf-8") == "Monthly summary"

    def test_get_all_summaries_ordering(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_daily("comp-sum", date(2026, 2, 14), "Daily A")
        store.save_daily("comp-sum", date(2026, 2, 15), "Daily B")
        store.save_weekly("comp-sum", "2026-W06", "Weekly A")
        store.save_monthly("comp-sum", "2026-01", "Monthly A")

        summaries = store.get_all_summaries("comp-sum")
        assert [(s.summary_type, s.period) for s in summaries] == [
            ("monthly", "2026-01"),
            ("weekly", "2026-W06"),
            ("daily", "2026-02-14"),
            ("daily", "2026-02-15"),
        ]

    def test_get_all_summaries_empty(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        assert store.get_all_summaries("comp-sum") == []

    def test_delete_daily(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_daily("comp-sum", date(2026, 2, 14), "Daily summary")

        deleted = store.delete_daily("comp-sum", date(2026, 2, 14))
        assert deleted is True
        assert store.get_all_summaries("comp-sum") == []

    def test_delete_weekly(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_weekly("comp-sum", "2026-W07", "Weekly summary")

        deleted = store.delete_weekly("comp-sum", "2026-W07")
        assert deleted is True
        assert store.get_all_summaries("comp-sum") == []

    def test_list_dailies(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_daily("comp-sum", date(2026, 2, 15), "B")
        store.save_daily("comp-sum", date(2026, 2, 14), "A")

        assert store.list_dailies("comp-sum") == [date(2026, 2, 14), date(2026, 2, 15)]

    def test_list_weeklies(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_weekly("comp-sum", "2026-W07", "W7")
        store.save_weekly("comp-sum", "2026-W06", "W6")

        assert store.list_weeklies("comp-sum") == ["2026-W06", "2026-W07"]

    def test_list_monthlies(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        store.save_monthly("comp-sum", "2026-02", "M2")
        store.save_monthly("comp-sum", "2026-01", "M1")

        assert store.list_monthlies("comp-sum") == ["2026-01", "2026-02"]

    def test_auto_creates_directories(self, tmp_path: Path) -> None:
        store = SummaryStore(data_dir=tmp_path)
        assert not (tmp_path / "comp-sum" / "summaries").exists()

        store.save_monthly("comp-sum", "2026-01", "Monthly summary")
        assert (tmp_path / "comp-sum" / "summaries" / "monthly").exists()
