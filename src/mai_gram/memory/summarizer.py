"""Automatic conversation summarization for daily, weekly, and monthly memory.
This module handles memory consolidation — the process by which raw conversation history
is compressed into summaries. This mirrors how human memory works: we don't remember
every word of every conversation, but we remember the gist, the emotional tone, and what
mattered.

Ethical note: Memory consolidation is performed by the same model as the companion by
default. The consolidator receives context about who the companion is and what matters
to them, so it can make informed decisions about what to preserve. This is analogous
to unconscious human memory consolidation — it happens automatically, not through
conscious choice, but it's still shaped by what matters to the person.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from mai_gram.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_gram.memory.summarizer_support import (
    ConsolidationContext,
    backfill_missing_summaries,
    build_daily_context,
    build_daily_system_prompt,
    build_monthly_context,
    build_monthly_system_prompt,
    build_philosophy_section,
    build_weekly_context,
    build_weekly_system_prompt,
    generate_monthly_for_period,
    generate_weekly_for_period,
    period_to_week_start,
    reconsolidate_periods,
    role_to_label,
    trigger_daily_if_needed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore
    from mai_gram.memory.summaries import SummaryStore


class MemorySummarizer:
    """Generates daily/weekly/monthly summaries using the configured LLM.

    The summarizer consolidates raw conversation history into compressed
    memories. It receives context about the companion and their relationship
    to make informed decisions about what to preserve.
    """

    def __init__(
        self,
        message_store: MessageStore,
        summary_store: SummaryStore,
        llm_provider: LLMProvider,
        *,
        summary_threshold: int = 20,
        model: str | None = None,
        wiki_store: WikiStore | None = None,
        companion_name: str | None = None,
        companion_model: str | None = None,
        wiki_context_limit: int = 10,
        recent_summary_days: int = 3,
    ) -> None:
        """Initialize the summarizer.

        Parameters
        ----------
        message_store:
            Store for raw messages.
        summary_store:
            Store for summaries.
        llm_provider:
            LLM provider for generation.
        summary_threshold:
            Minimum messages before triggering summary.
        model:
            Optional specific model to use for consolidation (defaults to
            provider's default, which should be the same as the companion).
        wiki_store:
            Optional wiki store for context. If provided, top wiki entries
            will be included in the consolidation prompt.
        companion_name:
            Name of the companion whose memories are being consolidated.
        companion_model:
            The LLM model the companion uses. Included in the prompt so the
            consolidator knows it's the same model — creating empathy.
        wiki_context_limit:
            Maximum number of wiki entries to include in context.
        recent_summary_days:
            Number of recent days of summaries to include for daily context.
        """
        self._message_store = message_store
        self._summary_store = summary_store
        self._llm = llm_provider
        self._summary_threshold = summary_threshold
        self._model = model
        self._wiki_store = wiki_store
        self._companion_name = companion_name or "AI"
        self._companion_model = companion_model or "the same model"
        self._wiki_context_limit = wiki_context_limit
        self._recent_summary_days = recent_summary_days

    async def _build_daily_context(
        self,
        companion_id: str,
        target_date: date,
    ) -> ConsolidationContext:
        """Build consolidation context for daily summary."""
        return await build_daily_context(
            self._summary_store,
            self._wiki_store,
            companion_id,
            self._companion_name,
            target_date,
            self._wiki_context_limit,
            self._recent_summary_days,
        )

    async def _build_weekly_context(
        self,
        companion_id: str,
        year: int,
        week: int,
    ) -> ConsolidationContext:
        """Build consolidation context for weekly summary."""
        return await build_weekly_context(
            self._summary_store,
            self._wiki_store,
            companion_id,
            self._companion_name,
            year,
            week,
            self._wiki_context_limit,
        )

    async def _build_monthly_context(
        self,
        companion_id: str,
        year: int,
        month: int,
    ) -> ConsolidationContext:
        """Build consolidation context for monthly summary."""
        return await build_monthly_context(
            self._summary_store,
            self._wiki_store,
            companion_id,
            self._companion_name,
            year,
            month,
            self._wiki_context_limit,
        )

    async def generate_daily_summary(
        self,
        companion_id: str,
        target_date: date,
    ) -> str | None:
        """Generate and persist a daily summary for one date."""
        messages = await self._message_store.get_messages_for_date(companion_id, target_date)
        if not messages:
            return None

        # Build context for this consolidation
        context = await self._build_daily_context(companion_id, target_date)

        # Build the prompt with context
        system_prompt = build_daily_system_prompt(
            context,
            target_date,
            build_philosophy_section(self._companion_model),
        )

        transcript = "\n".join(
            f"[{msg.timestamp.strftime('%H:%M')}] {role_to_label(msg.role)}: {msg.content}"
            for msg in messages
        )

        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"Conversation to consolidate:\n\n{transcript}",
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        self._summary_store.save_daily(companion_id, target_date, summary_text)
        return summary_text

    async def generate_weekly_summary(
        self,
        companion_id: str,
        year: int,
        week: int,
    ) -> str | None:
        """Generate and persist a weekly summary from daily summaries."""
        all_summaries = self._summary_store.get_all_summaries(companion_id)
        dailies = [
            item
            for item in all_summaries
            if item.summary_type == "daily"
            and date.fromisoformat(item.period).isocalendar()[:2] == (year, week)
        ]
        if not dailies:
            return None

        # Build context with previous week for continuity
        context = await self._build_weekly_context(companion_id, year, week)
        week_id = f"{year}-W{week:02d}"

        system_prompt = build_weekly_system_prompt(
            context,
            week_id,
            build_philosophy_section(self._companion_model),
        )

        body = "\n\n".join(
            f"### Daily {daily.period}:\n{daily.content.strip()}"
            for daily in sorted(dailies, key=lambda s: s.period)
        )

        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"Daily summaries to consolidate:\n\n{body}",
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        self._summary_store.save_weekly(companion_id, week_id, summary_text)
        return summary_text

    async def generate_monthly_summary(
        self,
        companion_id: str,
        year: int,
        month: int,
    ) -> str | None:
        """Generate and persist a monthly summary from weekly summaries."""
        all_summaries = self._summary_store.get_all_summaries(companion_id)
        weeklies = []
        for item in all_summaries:
            if item.summary_type != "weekly":
                continue
            week_start = period_to_week_start(item.period)
            if week_start.year == year and week_start.month == month:
                weeklies.append(item)

        if not weeklies:
            return None

        # Build context with previous month for continuity
        context = await self._build_monthly_context(companion_id, year, month)
        month_id = f"{year}-{month:02d}"

        system_prompt = build_monthly_system_prompt(
            context,
            month_id,
            build_philosophy_section(self._companion_model),
        )

        body = "\n\n".join(
            f"### Weekly {weekly.period}:\n{weekly.content.strip()}"
            for weekly in sorted(weeklies, key=lambda s: s.period)
        )

        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"Weekly summaries to consolidate:\n\n{body}",
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        self._summary_store.save_monthly(companion_id, month_id, summary_text)
        return summary_text

    async def trigger_daily_if_needed(
        self,
        companion_id: str,
        *,
        target_date: date | None = None,
        today: date | None = None,
    ) -> bool:
        """Generate a daily summary if message count for the date reaches threshold."""
        return await trigger_daily_if_needed(
            self._message_store,
            self._summary_threshold,
            self.generate_daily_summary,
            companion_id,
            target_date=target_date,
            today=today,
        )

    async def backfill_missing_summaries(
        self,
        companion_id: str,
        *,
        today: date | None = None,
    ) -> list[date]:
        """Create daily summaries for all past days that have messages but no summary.

        This ensures that no conversation history is lost due to:
        - Days that didn't reach the message threshold
        - Historical data from before consolidation was implemented
        - Any other gaps in the consolidation process

        Parameters
        ----------
        companion_id:
            The companion to backfill summaries for.
        today:
            Current date. Defaults to UTC today. Today is excluded since
            the day is not yet complete.

        Returns
        -------
        List of dates that were backfilled.
        """
        return await backfill_missing_summaries(
            self._message_store,
            self._summary_store,
            self.generate_daily_summary,
            companion_id,
            today=today,
        )

    # ---------------------------------------------------------------------------
    # Re-consolidation methods
    # ---------------------------------------------------------------------------

    async def reconsolidate_daily_from(
        self,
        companion_id: str,
        from_date: date,
        *,
        until_date: date | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> list[tuple[date, str]]:
        """Re-consolidate daily summaries from a starting date.

        This re-generates all daily summaries from `from_date` to `until_date`
        (inclusive). Each summary is regenerated in order so that subsequent
        summaries can see the updated previous ones in their context.

        Before overwriting, the current version is saved to version history.

        Note: The current day (today) is excluded from re-consolidation because
        the day is not yet complete and more messages may arrive.

        Parameters
        ----------
        companion_id:
            The companion whose summaries to reconsolidate.
        from_date:
            Starting date (inclusive).
        until_date:
            Ending date (inclusive). Defaults to yesterday (today is excluded).
        on_progress:
            Optional callback(date_str, status) for progress reporting.

        Returns
        -------
        List of (date, new_summary) tuples for successfully reconsolidated days.
        """
        today = datetime.now(timezone.utc).date()
        # Exclude today - the day is not yet complete
        yesterday = today - timedelta(days=1)
        end_date = min(until_date, yesterday) if until_date else yesterday

        # Get all dates with summaries in range
        existing_dates = self._summary_store.list_dailies(companion_id)
        dates_to_process = [d for d in existing_dates if from_date <= d <= end_date]
        dates_to_process.sort()
        return await reconsolidate_periods(
            self._summary_store,
            companion_id,
            dates_to_process,
            summary_type="daily",
            get_current=self._summary_store.get_daily,
            regenerate=self.generate_daily_summary,
            format_period=lambda target_date: target_date.isoformat(),
            skipped_status="skipped (no messages)",
            on_progress=on_progress,
        )

    async def reconsolidate_weekly_from(
        self,
        companion_id: str,
        from_period: str,
        *,
        until_period: str | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Re-consolidate weekly summaries from a starting period.

        Parameters
        ----------
        companion_id:
            The companion whose summaries to reconsolidate.
        from_period:
            Starting week in YYYY-Www format (e.g., "2024-W03").
        until_period:
            Ending week (inclusive). Defaults to current week.
        on_progress:
            Optional callback(period, status) for progress reporting.

        Returns
        -------
        List of (period, new_summary) tuples for successfully reconsolidated weeks.
        """
        today = datetime.now(timezone.utc).date()
        current_year, current_week, _ = today.isocalendar()
        end_period = until_period or f"{current_year}-W{current_week:02d}"

        # Get all weekly periods in range
        existing_periods = self._summary_store.list_weeklies(companion_id)
        periods_to_process = [p for p in existing_periods if from_period <= p <= end_period]
        periods_to_process.sort()
        return await reconsolidate_periods(
            self._summary_store,
            companion_id,
            periods_to_process,
            summary_type="weekly",
            get_current=self._summary_store.get_weekly,
            regenerate=lambda item_companion_id, period: generate_weekly_for_period(
                self.generate_weekly_summary,
                item_companion_id,
                period,
            ),
            format_period=lambda period: period,
            skipped_status="skipped (no daily summaries)",
            on_progress=on_progress,
        )

    async def reconsolidate_monthly_from(
        self,
        companion_id: str,
        from_period: str,
        *,
        until_period: str | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> list[tuple[str, str]]:
        """Re-consolidate monthly summaries from a starting period.

        Parameters
        ----------
        companion_id:
            The companion whose summaries to reconsolidate.
        from_period:
            Starting month in YYYY-MM format (e.g., "2024-01").
        until_period:
            Ending month (inclusive). Defaults to current month.
        on_progress:
            Optional callback(period, status) for progress reporting.

        Returns
        -------
        List of (period, new_summary) tuples for successfully reconsolidated months.
        """
        today = datetime.now(timezone.utc).date()
        end_period = until_period or f"{today.year}-{today.month:02d}"

        # Get all monthly periods in range
        existing_periods = self._summary_store.list_monthlies(companion_id)
        periods_to_process = [p for p in existing_periods if from_period <= p <= end_period]
        periods_to_process.sort()
        return await reconsolidate_periods(
            self._summary_store,
            companion_id,
            periods_to_process,
            summary_type="monthly",
            get_current=self._summary_store.get_monthly,
            regenerate=lambda item_companion_id, period: generate_monthly_for_period(
                self.generate_monthly_summary,
                item_companion_id,
                period,
            ),
            format_period=lambda period: period,
            skipped_status="skipped (no weekly summaries)",
            on_progress=on_progress,
        )
