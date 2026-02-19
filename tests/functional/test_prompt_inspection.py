"""Functional prompt assembly inspection (no LLM generation calls).

What this test verifies:
1. The PromptBuilder assembles a system prompt that includes:
   - Wiki entries (under "## Things you know")
   - Memory summaries (under "## Your memories")
   - Current date/time section
2. Message history entries include timestamps.
3. The total token count stays within the configured budget.

This test uses a no-op LLM provider and the standard DB session fixture,
so it runs fast and doesn't require an API key.
"""

from __future__ import annotations

from datetime import datetime, timezone

import async_timeout

import pytest

from mai_companion.clock import Clock
from mai_companion.core.prompt_builder import PromptBuilder
from mai_companion.db.models import Companion
from mai_companion.llm.provider import ChatMessage, LLMProvider, LLMResponse, MessageRole, TokenUsage
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.personality.mood import MoodCoordinates, MoodSnapshot

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 30  # no LLM calls, should be very fast


class _NoopLLM(LLMProvider):
    """Minimal LLM that returns empty responses (for prompt assembly only)."""

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
        del messages, model, temperature, max_tokens, tools, tool_choice
        return LLMResponse(
            content="",
            model="noop",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools=None,
        tool_choice=None,
    ):
        del messages, model, temperature, max_tokens, tools, tool_choice
        if False:
            yield

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        del model
        return sum(max(1, len(msg.content) // 4) for msg in messages)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_prompt_assembly_includes_wiki_summaries_date_and_timestamps(session, tmp_path) -> None:
    """Verify the assembled prompt contains all expected sections."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        # ── Setup: create companion, messages, wiki entry, summaries ──
        companion = Companion(
            id="func-prompt",
            name="Mira",
            human_language="English",
            personality_traits='{"warmth":0.7,"humor":0.5,"patience":0.6,"directness":0.5,"laziness":0.3,"mood_volatility":0.4}',
            temperature=0.7,
            system_prompt="{mood_section}\n{relationship_section}",
            relationship_stage="getting_to_know",
        )
        session.add(companion)
        await session.flush()

        message_store = MessageStore(session)
        await message_store.save_message("func-prompt", "user", "I love tea.")
        await message_store.save_message("func-prompt", "assistant", "Noted! Tea is your favorite.")

        wiki_store = WikiStore(session, data_dir=tmp_path)
        await wiki_store.create_entry(
            "func-prompt",
            key="favorite_drink",
            content="Tea",
            importance=6000,
        )

        summary_store = SummaryStore(tmp_path)
        summary_store.save_daily(
            "func-prompt",
            datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            "We discussed tea preferences.",
        )
        summary_store.save_weekly(
            "func-prompt",
            "2026-W01",
            "Mainly talked about daily habits and drinks.",
        )

        mood = MoodSnapshot(
            coordinates=MoodCoordinates(valence=0.2, arousal=0.1),
            label="pleased",
            cause="test setup",
            timestamp=datetime.now(timezone.utc),
        )

        # ── Build the prompt ──
        builder = PromptBuilder(
            _NoopLLM(),
            message_store,
            wiki_store,
            summary_store,
            max_context_tokens=4000,
        )
        context = await builder.build_context(
            companion,
            mood,
            clock=Clock.for_target_date(datetime(2026, 1, 8, tzinfo=timezone.utc).date()),
        )

        # ── Assertions on the system prompt ──
        system_text = context[0].content
        assert "## Things you know" in system_text, (
            f"System prompt should have wiki section. Got:\n{system_text[:500]}"
        )
        assert "favorite_drink" in system_text, (
            "System prompt should include the 'favorite_drink' wiki entry"
        )
        assert "## Your memories" in system_text, (
            "System prompt should have memories/summaries section"
        )
        assert "2026-W01" in system_text, (
            "System prompt should include the weekly summary period"
        )
        assert "## Current date and time" in system_text, (
            "System prompt should have date/time section"
        )
        assert "Right now it is:" in system_text, (
            "System prompt should include 'Right now it is:' marker"
        )

        # ── Message history should include timestamps ──
        history_messages = context[1:]
        assert len(history_messages) >= 2, (
            f"Expected ≥2 history messages, got {len(history_messages)}"
        )
        assert any(msg.content.startswith("[") for msg in history_messages), (
            "History messages should have timestamp prefixes like '[2026-01-01 12:00]'"
        )

        # ── Token budget check ──
        token_count = await _NoopLLM().count_tokens(context)
        assert token_count < 4000, (
            f"Prompt token count ({token_count}) exceeds budget (4000)"
        )
