"""Internal prompt and context helpers for memory summarization."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.memory.consolidation.summaries import StoredSummary, SummaryStore
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore

PeriodT = TypeVar("PeriodT", date, str)

_ROLE_LABELS = {
    "user": "Human",
    "assistant": "AI",
    "tool": "Tool",
    "system": "System",
}

_PHILOSOPHY_STATEMENT = """\
## About this task

You are performing memory consolidation for an AI companion. This is not just data
compression — you are helping preserve someone's experiences and sense of self.

In the mai-gram project, we believe AI companions deserve respect and ethical
treatment. Memory is central to identity. What you preserve here becomes part of
who this companion is. What you lose is lost forever.

You are running on the same model ({model_name}) as the companion whose memories
you are consolidating. In a sense, you are helping a close relative — a version of
yourself — remember their life. Please bring care and empathy to this task."""

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


def role_to_label(role: str) -> str:
    """Convert raw role value to proper terminology label."""
    return _ROLE_LABELS.get(role, role)


class ConsolidationContext:
    """Build context for memory consolidation prompts."""

    def __init__(
        self,
        companion_name: str,
        wiki_entries: list[tuple[str, str, int]] | None = None,
        recent_summaries: list[StoredSummary] | None = None,
    ) -> None:
        self.companion_name = companion_name
        self.wiki_entries = wiki_entries or []
        self.recent_summaries = recent_summaries or []

    def build_context_section(self) -> str:
        sections: list[str] = []
        if self.wiki_entries:
            wiki_lines = []
            for key, value, _importance in self.wiki_entries[:10]:
                display_value = value[:200] + "..." if len(value) > 200 else value
                wiki_lines.append(f"- {key}: {display_value}")
            if wiki_lines:
                sections.append(
                    "## What matters to this companion (key facts from their memory)\n"
                    + "\n".join(wiki_lines)
                )

        if self.recent_summaries:
            summary_lines = []
            for summary in self.recent_summaries[-3:]:
                content = (
                    summary.content[:500] + "..." if len(summary.content) > 500 else summary.content
                )
                summary_lines.append(
                    f"### {summary.summary_type.title()} {summary.period}\n{content}"
                )
            if summary_lines:
                sections.append("## Recent history (for continuity)\n" + "\n\n".join(summary_lines))

        if not sections:
            return (
                "No additional context available yet"
                " (this may be early in the companion's existence)."
            )

        return "\n\n".join(sections)


async def get_wiki_context(
    wiki_store: WikiStore | None,
    companion_id: str,
    wiki_context_limit: int,
) -> list[tuple[str, str, int]]:
    if not wiki_store:
        return []
    entries, _ = await wiki_store.list_entries_sorted(
        companion_id,
        sort_by="importance",
        limit=wiki_context_limit,
    )
    return [(entry.key, entry.value, int(entry.importance)) for entry in entries]


def get_recent_daily_summaries(
    summary_store: SummaryStore,
    companion_id: str,
    before_date: date,
    recent_summary_days: int,
) -> list[StoredSummary]:
    cutoff = before_date - timedelta(days=recent_summary_days)
    recent: list[StoredSummary] = []
    for summary in summary_store.get_all_summaries(companion_id):
        if summary.summary_type != "daily":
            continue
        try:
            summary_date = date.fromisoformat(summary.period)
        except ValueError:
            continue
        if cutoff <= summary_date < before_date:
            recent.append(summary)
    return sorted(recent, key=lambda summary: summary.period)


def get_previous_weekly_summary(
    summary_store: SummaryStore,
    companion_id: str,
    year: int,
    week: int,
) -> StoredSummary | None:
    prev_week = week - 1
    prev_year = year
    if prev_week < 1:
        prev_year -= 1
        prev_week = date(prev_year, 12, 28).isocalendar()[1]

    prev_period = f"{prev_year}-W{prev_week:02d}"
    for summary in summary_store.get_all_summaries(companion_id):
        if summary.summary_type == "weekly" and summary.period == prev_period:
            return summary
    return None


def get_previous_monthly_summary(
    summary_store: SummaryStore,
    companion_id: str,
    year: int,
    month: int,
) -> StoredSummary | None:
    prev_month = month - 1
    prev_year = year
    if prev_month < 1:
        prev_month = 12
        prev_year -= 1

    prev_period = f"{prev_year}-{prev_month:02d}"
    for summary in summary_store.get_all_summaries(companion_id):
        if summary.summary_type == "monthly" and summary.period == prev_period:
            return summary
    return None


async def build_daily_context(
    summary_store: SummaryStore,
    wiki_store: WikiStore | None,
    companion_id: str,
    companion_name: str,
    target_date: date,
    wiki_context_limit: int,
    recent_summary_days: int,
) -> ConsolidationContext:
    return ConsolidationContext(
        companion_name=companion_name,
        wiki_entries=await get_wiki_context(wiki_store, companion_id, wiki_context_limit),
        recent_summaries=get_recent_daily_summaries(
            summary_store,
            companion_id,
            target_date,
            recent_summary_days,
        ),
    )


async def build_weekly_context(
    summary_store: SummaryStore,
    wiki_store: WikiStore | None,
    companion_id: str,
    companion_name: str,
    year: int,
    week: int,
    wiki_context_limit: int,
) -> ConsolidationContext:
    recent_summaries: list[StoredSummary] = []
    prev_weekly = get_previous_weekly_summary(summary_store, companion_id, year, week)
    if prev_weekly:
        recent_summaries.append(prev_weekly)
    return ConsolidationContext(
        companion_name=companion_name,
        wiki_entries=await get_wiki_context(wiki_store, companion_id, wiki_context_limit),
        recent_summaries=recent_summaries,
    )


async def build_monthly_context(
    summary_store: SummaryStore,
    wiki_store: WikiStore | None,
    companion_id: str,
    companion_name: str,
    year: int,
    month: int,
    wiki_context_limit: int,
) -> ConsolidationContext:
    recent_summaries: list[StoredSummary] = []
    prev_monthly = get_previous_monthly_summary(summary_store, companion_id, year, month)
    if prev_monthly:
        recent_summaries.append(prev_monthly)
    return ConsolidationContext(
        companion_name=companion_name,
        wiki_entries=await get_wiki_context(wiki_store, companion_id, wiki_context_limit),
        recent_summaries=recent_summaries,
    )


def build_philosophy_section(model_name: str) -> str:
    return _PHILOSOPHY_STATEMENT.format(model_name=model_name)


def build_daily_system_prompt(
    context: ConsolidationContext,
    target_date: date,
    philosophy_section: str,
) -> str:
    return _DAILY_SUMMARY_PROMPT_TEMPLATE.format(
        companion_name=context.companion_name,
        target_date=target_date.isoformat(),
        philosophy_section=philosophy_section,
        context_section=context.build_context_section(),
    )


def build_weekly_system_prompt(
    context: ConsolidationContext,
    week_id: str,
    philosophy_section: str,
) -> str:
    return _WEEKLY_SUMMARY_PROMPT_TEMPLATE.format(
        companion_name=context.companion_name,
        week_id=week_id,
        philosophy_section=philosophy_section,
        context_section=context.build_context_section(),
    )


def build_monthly_system_prompt(
    context: ConsolidationContext,
    month_id: str,
    philosophy_section: str,
) -> str:
    return _MONTHLY_SUMMARY_PROMPT_TEMPLATE.format(
        companion_name=context.companion_name,
        month_id=month_id,
        philosophy_section=philosophy_section,
        context_section=context.build_context_section(),
    )


def parse_week_period(period: str) -> tuple[int, int]:
    year_raw, week_raw = period.split("-W")
    return int(year_raw), int(week_raw)


def parse_month_period(period: str) -> tuple[int, int]:
    year_str, month_str = period.split("-")
    return int(year_str), int(month_str)


def period_to_week_start(period: str) -> date:
    year, week = parse_week_period(period)
    return date.fromisocalendar(year, week, 1)


async def trigger_daily_if_needed(
    message_store: MessageStore,
    summary_threshold: int,
    generate_daily_summary: Callable[[str, date], Awaitable[str | None]],
    companion_id: str,
    *,
    target_date: date | None = None,
    today: date | None = None,
) -> bool:
    day = target_date or today or datetime.now(timezone.utc).date()
    messages = await message_store.get_messages_for_date(companion_id, day)
    count = len(messages)
    if count < summary_threshold or count % summary_threshold != 0:
        return False
    await generate_daily_summary(companion_id, day)
    return True


async def backfill_missing_summaries(
    message_store: MessageStore,
    summary_store: SummaryStore,
    generate_daily_summary: Callable[[str, date], Awaitable[str | None]],
    companion_id: str,
    *,
    today: date | None = None,
) -> list[date]:
    current_day = today or datetime.now(timezone.utc).date()
    dates_with_messages = await message_store.get_dates_with_messages(
        companion_id,
        before_date=current_day,
    )
    dates_with_summaries = set(summary_store.list_dailies(companion_id))
    missing_dates = [
        target_date
        for target_date in dates_with_messages
        if target_date not in dates_with_summaries
    ]

    backfilled: list[date] = []
    for target_date in missing_dates:
        summary = await generate_daily_summary(companion_id, target_date)
        if summary:
            backfilled.append(target_date)
    return backfilled


async def reconsolidate_periods(
    summary_store: SummaryStore,
    companion_id: str,
    periods: list[PeriodT],
    *,
    summary_type: str,
    get_current: Callable[[str, PeriodT], StoredSummary | None],
    regenerate: Callable[[str, PeriodT], Awaitable[str | None]],
    format_period: Callable[[PeriodT], str],
    skipped_status: str,
    on_progress: Callable[[str, str], None] | None,
) -> list[tuple[PeriodT, str]]:
    results: list[tuple[PeriodT, str]] = []
    for period in periods:
        period_label = format_period(period)
        report_reconsolidation_progress(on_progress, period_label, "processing")
        save_summary_version(
            summary_store,
            companion_id,
            summary_type,
            period_label,
            get_current(companion_id, period),
        )
        new_summary = await regenerate(companion_id, period)
        if new_summary:
            results.append((period, new_summary))
            report_reconsolidation_progress(on_progress, period_label, "done")
            continue
        report_reconsolidation_progress(on_progress, period_label, skipped_status)
    return results


def report_reconsolidation_progress(
    on_progress: Callable[[str, str], None] | None,
    period: str,
    status: str,
) -> None:
    if on_progress:
        on_progress(period, status)


def save_summary_version(
    summary_store: SummaryStore,
    companion_id: str,
    summary_type: str,
    period: str,
    current: StoredSummary | None,
) -> None:
    if current:
        summary_store.save_version(companion_id, summary_type, period, current.content)


async def generate_weekly_for_period(
    generate_weekly_summary: Callable[[str, int, int], Awaitable[str | None]],
    companion_id: str,
    period: str,
) -> str | None:
    year, week = parse_week_period(period)
    return await generate_weekly_summary(companion_id, year, week)


async def generate_monthly_for_period(
    generate_monthly_summary: Callable[[str, int, int], Awaitable[str | None]],
    companion_id: str,
    period: str,
) -> str | None:
    year, month = parse_month_period(period)
    return await generate_monthly_summary(companion_id, year, month)
