"""Automatic conversation summarization for daily, weekly, and monthly memory."""

from __future__ import annotations

from datetime import date, datetime, timezone

from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore

_DAILY_SUMMARY_PROMPT = """You are a neutral memory summarizer.

Summarize the conversation snippets into concise factual memory notes.
Do not roleplay. Do not use personality.
Preserve key facts, events, preferences, plans, and unresolved questions.
Write in the same language as the conversation."""

_WEEKLY_SUMMARY_PROMPT = """You are a neutral memory summarizer.

You are given several daily summaries from the same ISO week.
Create one concise weekly summary with recurring topics and important facts.
Write in the same language as the source summaries."""

_MONTHLY_SUMMARY_PROMPT = """You are a neutral memory summarizer.

You are given several weekly summaries from the same month.
Create one concise monthly summary that keeps durable memory signal.
Write in the same language as the source summaries."""


class MemorySummarizer:
    """Generates daily/weekly/monthly summaries using the configured LLM."""

    def __init__(
        self,
        message_store: MessageStore,
        summary_store: SummaryStore,
        llm_provider: LLMProvider,
        *,
        summary_threshold: int = 20,
        model: str | None = None,
    ) -> None:
        self._message_store = message_store
        self._summary_store = summary_store
        self._llm = llm_provider
        self._summary_threshold = summary_threshold
        self._model = model

    async def generate_daily_summary(self, companion_id: str, target_date: date) -> str | None:
        """Generate and persist a daily summary for one date."""
        messages = await self._message_store.get_messages_for_date(companion_id, target_date)
        if not messages:
            return None

        transcript = "\n".join(
            f"[{msg.timestamp.strftime('%H:%M')}] {msg.role}: {msg.content}" for msg in messages
        )
        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=_DAILY_SUMMARY_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        f"Create a daily summary for {target_date.isoformat()}.\n\n"
                        f"Conversation:\n{transcript}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        self._summary_store.save_daily(companion_id, target_date, summary_text)
        return summary_text

    async def generate_weekly_summary(self, companion_id: str, year: int, week: int) -> str | None:
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

        body = "\n\n".join(
            f"Daily {daily.period}:\n{daily.content.strip()}" for daily in sorted(dailies, key=lambda s: s.period)
        )
        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=_WEEKLY_SUMMARY_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        f"Create a weekly summary for ISO week {year}-W{week:02d}.\n\n"
                        f"Daily summaries:\n{body}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        period = f"{year}-W{week:02d}"
        self._summary_store.save_weekly(companion_id, period, summary_text)
        return summary_text

    async def generate_monthly_summary(self, companion_id: str, year: int, month: int) -> str | None:
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

        body = "\n\n".join(
            f"Weekly {weekly.period}:\n{weekly.content.strip()}"
            for weekly in sorted(weeklies, key=lambda s: s.period)
        )
        response = await self._llm.generate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=_MONTHLY_SUMMARY_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        f"Create a monthly summary for {year}-{month:02d}.\n\n"
                        f"Weekly summaries:\n{body}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=800,
            model=self._model,
        )
        summary_text = response.content.strip()
        period = f"{year}-{month:02d}"
        self._summary_store.save_monthly(companion_id, period, summary_text)
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


def _period_to_week_start(period: str) -> date:
    year_raw, week_raw = period.split("-W")
    return date.fromisocalendar(int(year_raw), int(week_raw), 1)
