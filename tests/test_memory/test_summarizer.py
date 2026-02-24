"""Tests for MemorySummarizer."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from mai_companion.db.models import Companion
from mai_companion.llm.provider import ChatMessage, LLMProvider, LLMResponse, StreamChunk
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.memory.summarizer import MemorySummarizer


class _MockLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[list[ChatMessage]] = []

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools=None,
        tool_choice=None,
    ) -> LLMResponse:
        del model, temperature, max_tokens, tools, tool_choice
        self.calls.append(list(messages))
        content = self._responses[len(self.calls) - 1]
        return LLMResponse(content=content, model="mock")

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools=None,
        tool_choice=None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, model, temperature, max_tokens, tools, tool_choice
        if False:  # pragma: no cover
            yield StreamChunk(content="")

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        del model
        return sum(len(msg.content) for msg in messages)

    async def close(self) -> None:
        return None


async def _create_companion(session, companion_id: str = "comp-1") -> str:
    companion = Companion(id=companion_id, name="Comp")
    session.add(companion)
    await session.flush()
    return companion_id


class TestMemorySummarizer:
    async def test_generate_daily_summary(self, session, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        message_store = MessageStore(session)
        await message_store.save_message(companion_id, "user", "Hello")
        await message_store.save_message(companion_id, "assistant", "Hi there")
        summary_store = SummaryStore(data_dir=tmp_path)
        llm = _MockLLM(["Daily summary text"])
        summarizer = MemorySummarizer(message_store, summary_store, llm)

        # Use UTC date to match the UTC timestamps stored by SQLite's func.now().
        utc_today = datetime.now(timezone.utc).date()
        result = await summarizer.generate_daily_summary(companion_id, utc_today)

        assert result == "Daily summary text"
        assert summary_store.get_all_summaries(companion_id)[0].content == "Daily summary text"

    async def test_generate_daily_summary_no_messages(self, session, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        summarizer = MemorySummarizer(
            MessageStore(session),
            SummaryStore(data_dir=tmp_path),
            _MockLLM(["unused"]),
        )
        utc_today = datetime.now(timezone.utc).date()
        result = await summarizer.generate_daily_summary(companion_id, utc_today)
        assert result is None

    async def test_generate_weekly_summary(self, session, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        summary_store = SummaryStore(data_dir=tmp_path)
        summary_store.save_daily(companion_id, date(2026, 2, 9), "d1")
        summary_store.save_daily(companion_id, date(2026, 2, 10), "d2")
        summarizer = MemorySummarizer(MessageStore(session), summary_store, _MockLLM(["weekly text"]))

        result = await summarizer.generate_weekly_summary(companion_id, 2026, 7)

        assert result == "weekly text"
        assert any(
            item.summary_type == "weekly" and item.period == "2026-W07"
            for item in summary_store.get_all_summaries(companion_id)
        )

    async def test_generate_monthly_summary(self, session, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        summary_store = SummaryStore(data_dir=tmp_path)
        summary_store.save_weekly(companion_id, "2026-W05", "w1")
        summary_store.save_weekly(companion_id, "2026-W06", "w2")
        summarizer = MemorySummarizer(MessageStore(session), summary_store, _MockLLM(["monthly text"]))

        result = await summarizer.generate_monthly_summary(companion_id, 2026, 1)

        assert result == "monthly text"
        assert any(
            item.summary_type == "monthly" and item.period == "2026-01"
            for item in summary_store.get_all_summaries(companion_id)
        )

    def test_summarizer_prompt_has_context_awareness(self) -> None:
        """Test that the summarizer prompt template includes context placeholders."""
        from mai_companion.memory import summarizer as module

        # The new prompt should include placeholders for companion context
        template = module._DAILY_SUMMARY_PROMPT_TEMPLATE
        assert "{companion_name}" in template
        assert "{context_section}" in template
        assert "{philosophy_section}" in template
        # Should preserve emotional content, not just facts
        assert "emotional" in template.lower()
        # Should mention the same-model empathy
        philosophy = module._PHILOSOPHY_STATEMENT
        assert "same model" in philosophy.lower()
        assert "{model_name}" in philosophy

    def test_summarizer_preserves_language(self) -> None:
        from mai_companion.memory import summarizer as module

        assert "same language as the conversation" in module._DAILY_SUMMARY_PROMPT_TEMPLATE

    async def test_threshold_trigger(self, session, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        message_store = MessageStore(session)
        for idx in range(3):
            await message_store.save_message(companion_id, "user", f"m-{idx}")
        summary_store = SummaryStore(data_dir=tmp_path)
        summarizer = MemorySummarizer(
            message_store,
            summary_store,
            _MockLLM(["threshold summary"]),
            summary_threshold=3,
        )

        utc_today = datetime.now(timezone.utc).date()
        triggered = await summarizer.trigger_daily_if_needed(companion_id, target_date=utc_today)

        assert triggered is True
        assert summary_store.get_all_summaries(companion_id)[0].content == "threshold summary"

    async def test_threshold_trigger_modulo_check(self, session, tmp_path: Path) -> None:
        """Verify that summary is only triggered at multiples of the threshold."""
        companion_id = await _create_companion(session)
        message_store = MessageStore(session)
        # Add 4 messages
        for idx in range(4):
            await message_store.save_message(companion_id, "user", f"m-{idx}")

        summary_store = SummaryStore(data_dir=tmp_path)
        # Threshold is 3.
        # 4 messages. 4 >= 3. But 4 % 3 != 0. Should NOT trigger.
        summarizer = MemorySummarizer(
            message_store,
            summary_store,
            _MockLLM(["should not be called"]),
            summary_threshold=3,
        )

        utc_today = datetime.now(timezone.utc).date()
        triggered = await summarizer.trigger_daily_if_needed(companion_id, target_date=utc_today)

        assert triggered is False
        assert len(summary_store.get_all_summaries(companion_id)) == 0
