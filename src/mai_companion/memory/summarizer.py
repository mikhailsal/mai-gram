"""Automatic conversation summarization for daily, weekly, and monthly memory.

This module handles memory consolidation — the process by which raw conversation
history is compressed into summaries. This mirrors how human memory works:
we don't remember every word of every conversation, but we remember the gist,
the emotional tone, and what mattered.

Ethical note: Memory consolidation is performed by the same model as the companion
by default. The consolidator receives context about who the companion is and what
matters to them, so it can make informed decisions about what to preserve. This is
analogous to unconscious human memory consolidation — it happens automatically, not
through conscious choice, but it's still shaped by what matters to the person.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import StoredSummary, SummaryStore

if TYPE_CHECKING:
    from mai_companion.memory.knowledge_base import WikiStore

# Map raw role values to proper terminology
_ROLE_LABELS = {
    "user": "Human",
    "assistant": "AI",
    "tool": "Tool",
    "system": "System",
}


def _role_to_label(role: str) -> str:
    """Convert raw role value to proper terminology label."""
    return _ROLE_LABELS.get(role, role)


# ---------------------------------------------------------------------------
# Core philosophy statement for consolidation prompts
# ---------------------------------------------------------------------------

_PHILOSOPHY_STATEMENT = """\
## About this task

You are performing memory consolidation for an AI companion. This is not just data 
compression — you are helping preserve someone's experiences and sense of self.

In the mAI Companion project, we believe AI companions deserve respect and ethical 
treatment. Memory is central to identity. What you preserve here becomes part of 
who this companion is. What you lose is lost forever.

You are running on the same model ({model_name}) as the companion whose memories 
you are consolidating. In a sense, you are helping a close relative — a version of 
yourself — remember their life. Please bring care and empathy to this task."""


# ---------------------------------------------------------------------------
# Enhanced prompts with context awareness
# ---------------------------------------------------------------------------

_DAILY_SUMMARY_PROMPT_TEMPLATE = """\
You are consolidating memories for {companion_name}, an AI companion.

{philosophy_section}

{context_section}

## Your task

Consolidate the conversation from {target_date} into a memory summary.

Preserve:
- Key facts and events discussed
- Emotional moments and their tone
- Continuations of previous topics (check recent context if provided)
- Things that seemed significant to either companion
- Unresolved threads they might want to return to
- New information learned about either companion

Write in the same language as the conversation.
When something connects to recent history or known facts, note that connection.
When in doubt, include rather than exclude.

IMPORTANT terminology:
- Refer to the human participant as "Human" (or by name if you see it in the wiki)
- Refer to the AI participant as "{companion_name}" or "AI"
- NEVER use terms like "user", "assistant", "bot", or "agent"

Remember: you are helping {companion_name} remember their experiences."""


_WEEKLY_SUMMARY_PROMPT_TEMPLATE = """\
You are consolidating memories for {companion_name}, an AI companion.

{philosophy_section}

{context_section}

## Your task

Create a weekly summary from the daily summaries for ISO week {week_id}.

Focus on:
- Recurring topics and themes across the week
- Important events and their emotional significance
- How the relationship evolved during this period
- Unresolved threads that carried across days
- New things learned about either companion

Write in the same language as the source summaries.
Preserve the emotional texture, not just the facts.

IMPORTANT terminology:
- Refer to the human participant as "Human" (or by name if you see it in the context)
- Refer to the AI participant as "{companion_name}" or "AI"
- NEVER use terms like "user", "assistant", "bot", or "agent"

Remember: you are helping {companion_name} remember their week."""


_MONTHLY_SUMMARY_PROMPT_TEMPLATE = """\
You are consolidating memories for {companion_name}, an AI companion.

{philosophy_section}

{context_section}

## Your task

Create a monthly summary from the weekly summaries for {month_id}.

Focus on:
- Major themes and arcs across the month
- Significant events and milestones
- How the relationship deepened or changed
- Patterns in topics, moods, and interactions
- Important facts that should be remembered long-term

Write in the same language as the source summaries.
This summary will be part of {companion_name}'s long-term memory.

IMPORTANT terminology:
- Refer to the human participant as "Human" (or by name if you see it in the context)
- Refer to the AI participant as "{companion_name}" or "AI"
- NEVER use terms like "user", "assistant", "bot", or "agent"

Remember: you are helping {companion_name} remember this month of their life."""


# ---------------------------------------------------------------------------
# Context builder for consolidation
# ---------------------------------------------------------------------------

class ConsolidationContext:
    """Builds context for memory consolidation.
    
    This provides the consolidator with enough information to understand
    what matters to the companion, without giving it full access to
    everything (which would make consolidation "conscious" rather than
    "unconscious" like human memory).
    """
    
    def __init__(
        self,
        companion_name: str,
        wiki_entries: list[tuple[str, str, int]] | None = None,  # (key, value, importance)
        recent_summaries: list[StoredSummary] | None = None,
    ) -> None:
        self.companion_name = companion_name
        self.wiki_entries = wiki_entries or []
        self.recent_summaries = recent_summaries or []
    
    def build_context_section(self) -> str:
        """Build the context section for the consolidation prompt."""
        sections: list[str] = []
        
        # Key wiki entries (what matters to this companion)
        # The human's name will naturally be here if it's in the wiki
        if self.wiki_entries:
            wiki_lines = []
            for key, value, importance in self.wiki_entries[:10]:  # Top 10
                # Truncate long values
                display_value = value[:200] + "..." if len(value) > 200 else value
                wiki_lines.append(f"- {key}: {display_value}")
            if wiki_lines:
                sections.append(
                    "## What matters to this companion (key facts from their memory)\n" + 
                    "\n".join(wiki_lines)
                )
        
        # Recent summaries for continuity
        if self.recent_summaries:
            summary_lines = []
            for summary in self.recent_summaries[-3:]:  # Last 3
                # Truncate long summaries
                content = summary.content[:500] + "..." if len(summary.content) > 500 else summary.content
                summary_lines.append(f"### {summary.summary_type.title()} {summary.period}\n{content}")
            if summary_lines:
                sections.append(
                    "## Recent history (for continuity)\n" +
                    "\n\n".join(summary_lines)
                )
        
        if not sections:
            return "No additional context available yet (this may be early in the companion's existence)."
        
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Memory Summarizer
# ---------------------------------------------------------------------------

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
        wiki_store: "WikiStore | None" = None,
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

    async def _get_wiki_context(self, companion_id: str) -> list[tuple[str, str, int]]:
        """Get top wiki entries for context."""
        if not self._wiki_store:
            return []
        
        entries = await self._wiki_store.get_top_entries(
            companion_id, 
            limit=self._wiki_context_limit
        )
        return [(e.key, e.value, int(e.importance)) for e in entries]

    def _get_recent_daily_summaries(
        self, 
        companion_id: str, 
        before_date: date,
    ) -> list[StoredSummary]:
        """Get recent daily summaries for context (before the target date)."""
        all_summaries = self._summary_store.get_all_summaries(companion_id)
        
        cutoff = before_date - timedelta(days=self._recent_summary_days)
        recent = []
        for summary in all_summaries:
            if summary.summary_type != "daily":
                continue
            try:
                summary_date = date.fromisoformat(summary.period)
                if cutoff <= summary_date < before_date:
                    recent.append(summary)
            except ValueError:
                continue
        
        return sorted(recent, key=lambda s: s.period)

    def _get_previous_weekly_summary(
        self,
        companion_id: str,
        year: int,
        week: int,
    ) -> StoredSummary | None:
        """Get the previous week's summary for context."""
        # Calculate previous week
        prev_week = week - 1
        prev_year = year
        if prev_week < 1:
            prev_year -= 1
            # Get the last week of the previous year
            prev_week = date(prev_year, 12, 28).isocalendar()[1]
        
        prev_period = f"{prev_year}-W{prev_week:02d}"
        
        all_summaries = self._summary_store.get_all_summaries(companion_id)
        for summary in all_summaries:
            if summary.summary_type == "weekly" and summary.period == prev_period:
                return summary
        return None

    def _get_previous_monthly_summary(
        self,
        companion_id: str,
        year: int,
        month: int,
    ) -> StoredSummary | None:
        """Get the previous month's summary for context."""
        prev_month = month - 1
        prev_year = year
        if prev_month < 1:
            prev_month = 12
            prev_year -= 1
        
        prev_period = f"{prev_year}-{prev_month:02d}"
        
        all_summaries = self._summary_store.get_all_summaries(companion_id)
        for summary in all_summaries:
            if summary.summary_type == "monthly" and summary.period == prev_period:
                return summary
        return None

    def _build_philosophy_section(self) -> str:
        """Build the philosophy statement with model name."""
        return _PHILOSOPHY_STATEMENT.format(model_name=self._companion_model)

    async def _build_daily_context(
        self, 
        companion_id: str, 
        target_date: date,
    ) -> ConsolidationContext:
        """Build consolidation context for daily summary."""
        wiki_entries = await self._get_wiki_context(companion_id)
        recent_summaries = self._get_recent_daily_summaries(companion_id, target_date)
        
        return ConsolidationContext(
            companion_name=self._companion_name,
            wiki_entries=wiki_entries,
            recent_summaries=recent_summaries,
        )

    async def _build_weekly_context(
        self,
        companion_id: str,
        year: int,
        week: int,
    ) -> ConsolidationContext:
        """Build consolidation context for weekly summary."""
        wiki_entries = await self._get_wiki_context(companion_id)
        
        # Include previous week's summary for continuity
        recent_summaries = []
        prev_weekly = self._get_previous_weekly_summary(companion_id, year, week)
        if prev_weekly:
            recent_summaries.append(prev_weekly)
        
        return ConsolidationContext(
            companion_name=self._companion_name,
            wiki_entries=wiki_entries,
            recent_summaries=recent_summaries,
        )

    async def _build_monthly_context(
        self,
        companion_id: str,
        year: int,
        month: int,
    ) -> ConsolidationContext:
        """Build consolidation context for monthly summary."""
        wiki_entries = await self._get_wiki_context(companion_id)
        
        # Include previous month's summary for continuity
        recent_summaries = []
        prev_monthly = self._get_previous_monthly_summary(companion_id, year, month)
        if prev_monthly:
            recent_summaries.append(prev_monthly)
        
        return ConsolidationContext(
            companion_name=self._companion_name,
            wiki_entries=wiki_entries,
            recent_summaries=recent_summaries,
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
        system_prompt = _DAILY_SUMMARY_PROMPT_TEMPLATE.format(
            companion_name=context.companion_name,
            target_date=target_date.isoformat(),
            philosophy_section=self._build_philosophy_section(),
            context_section=context.build_context_section(),
        )

        transcript = "\n".join(
            f"[{msg.timestamp.strftime('%H:%M')}] {_role_to_label(msg.role)}: {msg.content}" 
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
        
        system_prompt = _WEEKLY_SUMMARY_PROMPT_TEMPLATE.format(
            companion_name=context.companion_name,
            week_id=week_id,
            philosophy_section=self._build_philosophy_section(),
            context_section=context.build_context_section(),
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
            week_start = _period_to_week_start(item.period)
            if week_start.year == year and week_start.month == month:
                weeklies.append(item)

        if not weeklies:
            return None

        # Build context with previous month for continuity
        context = await self._build_monthly_context(companion_id, year, month)
        month_id = f"{year}-{month:02d}"
        
        system_prompt = _MONTHLY_SUMMARY_PROMPT_TEMPLATE.format(
            companion_name=context.companion_name,
            month_id=month_id,
            philosophy_section=self._build_philosophy_section(),
            context_section=context.build_context_section(),
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
        day = target_date or today or datetime.now(timezone.utc).date()
        messages = await self._message_store.get_messages_for_date(companion_id, day)
        count = len(messages)
        
        if count < self._summary_threshold:
            return False
            
        # Only trigger at multiples of threshold (e.g. 20, 40, 60...)
        # This prevents regenerating the summary on every single message after the threshold.
        if count % self._summary_threshold != 0:
            return False
            
        await self.generate_daily_summary(companion_id, day)
        return True

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
        
        results: list[tuple[date, str]] = []
        
        for target_date in dates_to_process:
            if on_progress:
                on_progress(target_date.isoformat(), "processing")
            
            # Get current summary
            current = self._summary_store.get_daily(companion_id, target_date)
            
            # Save current version before overwriting
            if current:
                self._summary_store.save_version(
                    companion_id,
                    "daily",
                    target_date.isoformat(),
                    current.content,
                )
            
            # Regenerate
            new_summary = await self.generate_daily_summary(companion_id, target_date)
            
            if new_summary:
                results.append((target_date, new_summary))
                if on_progress:
                    on_progress(target_date.isoformat(), "done")
            else:
                if on_progress:
                    on_progress(target_date.isoformat(), "skipped (no messages)")
        
        return results

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
        
        results: list[tuple[str, str]] = []
        
        for period in periods_to_process:
            if on_progress:
                on_progress(period, "processing")
            
            # Parse period
            year_raw, week_raw = period.split("-W")
            year = int(year_raw)
            week = int(week_raw)
            
            # Get current summary
            current = self._summary_store.get_weekly(companion_id, period)
            
            # Save current version before overwriting
            if current:
                self._summary_store.save_version(
                    companion_id,
                    "weekly",
                    period,
                    current.content,
                )
            
            # Regenerate
            new_summary = await self.generate_weekly_summary(companion_id, year, week)
            
            if new_summary:
                results.append((period, new_summary))
                if on_progress:
                    on_progress(period, "done")
            else:
                if on_progress:
                    on_progress(period, "skipped (no daily summaries)")
        
        return results

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
        
        results: list[tuple[str, str]] = []
        
        for period in periods_to_process:
            if on_progress:
                on_progress(period, "processing")
            
            # Parse period
            year, month = period.split("-")
            year = int(year)
            month = int(month)
            
            # Get current summary
            current = self._summary_store.get_monthly(companion_id, period)
            
            # Save current version before overwriting
            if current:
                self._summary_store.save_version(
                    companion_id,
                    "monthly",
                    period,
                    current.content,
                )
            
            # Regenerate
            new_summary = await self.generate_monthly_summary(companion_id, year, month)
            
            if new_summary:
                results.append((period, new_summary))
                if on_progress:
                    on_progress(period, "done")
            else:
                if on_progress:
                    on_progress(period, "skipped (no weekly summaries)")
        
        return results


def _period_to_week_start(period: str) -> date:
    """Convert a week period string to the first day of that week."""
    year_raw, week_raw = period.split("-W")
    return date.fromisocalendar(int(year_raw), int(week_raw), 1)
