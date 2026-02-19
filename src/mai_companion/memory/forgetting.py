"""Forgetting engine that consolidates old summaries into higher-level summaries."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from mai_companion.memory.summaries import SummaryStore
from mai_companion.memory.summarizer import MemorySummarizer


class ForgettingEngine:
    """Consolidates stale daily/weekly summaries and removes lower-level files."""

    def __init__(self, summary_store: SummaryStore, summarizer: MemorySummarizer) -> None:
        self._summary_store = summary_store
        self._summarizer = summarizer

    async def run_forgetting_cycle(self, companion_id: str, *, today: date | None = None) -> None:
        """Run both weekly and monthly consolidation cycles."""
        current_day = today or datetime.now(timezone.utc).date()
        await self._consolidate_old_dailies(companion_id, current_day=current_day)
        await self._consolidate_old_weeklies(companion_id, current_day=current_day)

    async def _consolidate_old_dailies(self, companion_id: str, *, current_day: date) -> None:
        # We only consolidate a week if the ENTIRE week is older than the threshold.
        # This prevents partial consolidation where late-arriving dailies (or just the later days of the week)
        # get deleted without being added to the summary because the summary already exists.
        
        # Threshold: allow 7 days grace period after the week ends.
        # Week ends on Sunday. If today is > Sunday + 7, we consolidate.
        
        all_dailies = self._summary_store.list_dailies(companion_id)
        
        # Group all dailies by week
        groups: dict[tuple[int, int], list[date]] = defaultdict(list)
        for daily in all_dailies:
            iso_year, iso_week, _ = daily.isocalendar()
            groups[(iso_year, iso_week)].append(daily)

        weekly_periods = set(self._summary_store.list_weeklies(companion_id))
        
        for (iso_year, iso_week), daily_dates in groups.items():
            # Calculate the end of this ISO week (Sunday)
            week_start = date.fromisocalendar(iso_year, iso_week, 1)
            week_end = week_start + timedelta(days=6)
            
            # Check if the week is fully stale (e.g. ended more than 7 days ago)
            if current_day > week_end + timedelta(days=7):
                # Always regenerate/update the summary to ensure it includes ALL dailies
                # (even if it already exists, we might have new dailies that appeared later)
                await self._summarizer.generate_weekly_summary(companion_id, iso_year, iso_week)
                
                # Safe to delete all dailies for this week now
                for daily in daily_dates:
                    self._summary_store.delete_daily(companion_id, daily)

    async def _consolidate_old_weeklies(self, companion_id: str, *, current_day: date) -> None:
        # Similar logic for monthly consolidation: only consolidate if the month is fully past.
        
        weekly_periods = self._summary_store.list_weeklies(companion_id)
        
        # Group weeklies by month
        groups: dict[tuple[int, int], list[str]] = defaultdict(list)
        for period in weekly_periods:
            week_start = _period_to_week_start(period)
            # We assign a week to the month of its start date (simplification, but consistent)
            groups[(week_start.year, week_start.month)].append(period)

        monthly_periods = set(self._summary_store.list_monthlies(companion_id))
        
        for (year, month), periods in groups.items():
            # Calculate end of month
            # (start of next month - 1 day)
            if month == 12:
                next_month_start = date(year + 1, 1, 1)
            else:
                next_month_start = date(year, month + 1, 1)
            month_end = next_month_start - timedelta(days=1)
            
            # Check if month is fully stale (ended more than 28 days ago)
            if current_day > month_end + timedelta(days=28):
                await self._summarizer.generate_monthly_summary(companion_id, year, month)
                
                for period in periods:
                    self._summary_store.delete_weekly(companion_id, period)


def _period_to_week_start(period: str) -> date:
    year_raw, week_raw = period.split("-W")
    return date.fromisocalendar(int(year_raw), int(week_raw), 1)
