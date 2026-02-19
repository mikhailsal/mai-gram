"""Tests for PromptBuilder."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from mai_companion.core.prompt_builder import PromptBuilder
from mai_companion.db.models import Companion
from mai_companion.llm.provider import ChatMessage, LLMProvider, LLMResponse, StreamChunk, TokenUsage
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.personality.mood import MoodCoordinates, MoodSnapshot


class _MockLLM(LLMProvider):
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
        return LLMResponse(content="unused", model="mock")

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
        return sum(len(message.content) for message in messages)

    async def close(self) -> None:
        return None


async def _create_companion(session, companion_id: str = "comp-1") -> Companion:
    companion = Companion(
        id=companion_id,
        name="Comp",
        system_prompt=(
            "You are a companion.\n\n{mood_section}\n\n{relationship_section}"
        ),
    )
    session.add(companion)
    await session.flush()
    return companion


def _mood() -> MoodSnapshot:
    return MoodSnapshot(
        coordinates=MoodCoordinates(valence=0.2, arousal=0.1),
        label="pleased",
        cause=None,
        timestamp=datetime.now(timezone.utc),
    )


class TestPromptBuilder:
    async def test_build_context_structure(self, session, tmp_path: Path) -> None:
        companion = await _create_companion(session)
        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=tmp_path)
        summary_store = SummaryStore(data_dir=tmp_path)
        await message_store.save_message(companion.id, "user", "Hello")
        summary_store.save_daily(companion.id, datetime.now(timezone.utc).date(), "memory")
        await wiki_store.create_entry(companion.id, "human_name", "Alex", 9999)
        builder = PromptBuilder(_MockLLM(), message_store, wiki_store, summary_store)

        fixed_now = datetime(2026, 2, 19, 14, 35, tzinfo=timezone.utc)
        context = await builder.build_context(companion, _mood(), current_time=fixed_now)

        assert context[0].role.value == "system"
        assert "Things you know" in context[0].content
        assert "Your memories" in context[0].content
        assert "## Current date and time" in context[0].content
        assert len(context) >= 2

    async def test_build_context_short_term_messages(self, session, tmp_path: Path) -> None:
        companion = await _create_companion(session)
        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=tmp_path)
        summary_store = SummaryStore(data_dir=tmp_path)
        await message_store.save_message(companion.id, "user", "first")
        await message_store.save_message(companion.id, "assistant", "second")
        builder = PromptBuilder(_MockLLM(), message_store, wiki_store, summary_store)

        fixed_now = datetime(2026, 2, 19, 14, 35, tzinfo=timezone.utc)
        context = await builder.build_context(companion, _mood(), current_time=fixed_now)

        history_contents = [m.content for m in context[1:]]
        assert len(history_contents) == 2
        assert history_contents[0].endswith(" first")
        assert history_contents[1].endswith(" second")
        assert history_contents[0].startswith("[")
        assert history_contents[1].startswith("[")

    async def test_token_budget_truncation(self, session, tmp_path: Path, caplog) -> None:
        companion = await _create_companion(session)
        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=tmp_path)
        summary_store = SummaryStore(data_dir=tmp_path)
        summary_store.save_monthly(companion.id, "2026-01", "x" * 500)
        summary_store.save_monthly(companion.id, "2026-02", "y" * 500)
        builder = PromptBuilder(
            _MockLLM(),
            message_store,
            wiki_store,
            summary_store,
            max_context_tokens=300,
        )

        context = await builder.build_context(companion, _mood())

        assert "truncating summaries" in caplog.text.lower()
        assert "2026-01" not in context[0].content

    async def test_empty_wiki_and_summaries(self, session, tmp_path: Path) -> None:
        companion = await _create_companion(session)
        builder = PromptBuilder(
            _MockLLM(),
            MessageStore(session),
            WikiStore(session, data_dir=tmp_path),
            SummaryStore(data_dir=tmp_path),
        )

        context = await builder.build_context(companion, _mood())

        assert "No persistent knowledge yet." in context[0].content
        assert "No memory summaries yet." in context[0].content
