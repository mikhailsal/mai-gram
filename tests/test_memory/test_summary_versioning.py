"""Tests for summary versioning and reconsolidation."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from mai_companion.memory.summaries import SummaryStore, SummaryVersion


class TestSummaryVersioning:
    """Tests for version history functionality."""

    def test_save_version_creates_version_file(self, tmp_path: Path) -> None:
        """Test that save_version creates a version file."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        # First save a regular summary
        store.save_daily(companion_id, date(2024, 1, 15), "Original content")
        
        # Now save a version
        version_path = store.save_version(
            companion_id,
            "daily",
            "2024-01-15",
            "Original content",
        )
        
        assert version_path.exists()
        assert ".versions" in str(version_path)
        assert "2024-01-15_v1_" in version_path.name
        assert version_path.read_text() == "Original content"

    def test_save_version_increments_version_number(self, tmp_path: Path) -> None:
        """Test that multiple versions get incrementing numbers."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        store.save_daily(companion_id, date(2024, 1, 15), "Content v1")
        
        # Save multiple versions
        v1_path = store.save_version(companion_id, "daily", "2024-01-15", "Content v1")
        v2_path = store.save_version(companion_id, "daily", "2024-01-15", "Content v2")
        v3_path = store.save_version(companion_id, "daily", "2024-01-15", "Content v3")
        
        assert "_v1_" in v1_path.name
        assert "_v2_" in v2_path.name
        assert "_v3_" in v3_path.name

    def test_list_versions_returns_all_versions(self, tmp_path: Path) -> None:
        """Test that list_versions returns all saved versions."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        store.save_daily(companion_id, date(2024, 1, 15), "Current")
        store.save_version(companion_id, "daily", "2024-01-15", "Version 1")
        store.save_version(companion_id, "daily", "2024-01-15", "Version 2")
        
        versions = store.list_versions(companion_id, "daily", "2024-01-15")
        
        assert len(versions) == 2
        assert all(isinstance(v, SummaryVersion) for v in versions)
        assert versions[0].content == "Version 1"
        assert versions[1].content == "Version 2"
        assert "v1" in versions[0].version_id
        assert "v2" in versions[1].version_id

    def test_list_versions_empty_for_no_versions(self, tmp_path: Path) -> None:
        """Test that list_versions returns empty list when no versions exist."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        store.save_daily(companion_id, date(2024, 1, 15), "Current")
        
        versions = store.list_versions(companion_id, "daily", "2024-01-15")
        
        assert versions == []

    def test_list_versions_only_returns_matching_period(self, tmp_path: Path) -> None:
        """Test that list_versions only returns versions for the specified period."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        store.save_daily(companion_id, date(2024, 1, 15), "Day 15")
        store.save_daily(companion_id, date(2024, 1, 16), "Day 16")
        store.save_version(companion_id, "daily", "2024-01-15", "Day 15 v1")
        store.save_version(companion_id, "daily", "2024-01-16", "Day 16 v1")
        
        versions_15 = store.list_versions(companion_id, "daily", "2024-01-15")
        versions_16 = store.list_versions(companion_id, "daily", "2024-01-16")
        
        assert len(versions_15) == 1
        assert len(versions_16) == 1
        assert versions_15[0].content == "Day 15 v1"
        assert versions_16[0].content == "Day 16 v1"

    def test_get_all_summaries_with_versions(self, tmp_path: Path) -> None:
        """Test get_all_summaries_with_versions returns complete data."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-versioning"
        
        # Create some summaries with versions
        store.save_daily(companion_id, date(2024, 1, 15), "Day 15 current")
        store.save_version(companion_id, "daily", "2024-01-15", "Day 15 v1")
        
        store.save_daily(companion_id, date(2024, 1, 16), "Day 16 current")
        # No versions for day 16
        
        store.save_weekly(companion_id, "2024-W03", "Week 3 current")
        store.save_version(companion_id, "weekly", "2024-W03", "Week 3 v1")
        store.save_version(companion_id, "weekly", "2024-W03", "Week 3 v2")
        
        result = store.get_all_summaries_with_versions(companion_id)
        
        assert "daily" in result
        assert "weekly" in result
        assert "monthly" in result
        
        # Check daily
        assert len(result["daily"]) == 2
        day_15 = next((s, v) for s, v in result["daily"] if s.period == "2024-01-15")
        day_16 = next((s, v) for s, v in result["daily"] if s.period == "2024-01-16")
        assert len(day_15[1]) == 1  # 1 version
        assert len(day_16[1]) == 0  # no versions
        
        # Check weekly
        assert len(result["weekly"]) == 1
        week_3 = result["weekly"][0]
        assert week_3[0].period == "2024-W03"
        assert len(week_3[1]) == 2  # 2 versions


class TestSummaryGetters:
    """Tests for individual summary getters."""

    def test_get_daily_returns_summary(self, tmp_path: Path) -> None:
        """Test get_daily returns the correct summary."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-getters"
        
        store.save_daily(companion_id, date(2024, 1, 15), "Day 15 content")
        
        summary = store.get_daily(companion_id, date(2024, 1, 15))
        
        assert summary is not None
        assert summary.content == "Day 15 content"
        assert summary.period == "2024-01-15"
        assert summary.summary_type == "daily"

    def test_get_daily_returns_none_for_missing(self, tmp_path: Path) -> None:
        """Test get_daily returns None when summary doesn't exist."""
        store = SummaryStore(data_dir=tmp_path)
        
        summary = store.get_daily("test-getters", date(2024, 1, 15))
        
        assert summary is None

    def test_get_weekly_returns_summary(self, tmp_path: Path) -> None:
        """Test get_weekly returns the correct summary."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-getters"
        
        store.save_weekly(companion_id, "2024-W03", "Week 3 content")
        
        summary = store.get_weekly(companion_id, "2024-W03")
        
        assert summary is not None
        assert summary.content == "Week 3 content"
        assert summary.period == "2024-W03"

    def test_get_monthly_returns_summary(self, tmp_path: Path) -> None:
        """Test get_monthly returns the correct summary."""
        store = SummaryStore(data_dir=tmp_path)
        companion_id = "test-getters"
        
        store.save_monthly(companion_id, "2024-01", "January content")
        
        summary = store.get_monthly(companion_id, "2024-01")
        
        assert summary is not None
        assert summary.content == "January content"
        assert summary.period == "2024-01"
