"""Tests for structured LLM logging and session cost tracking."""

from __future__ import annotations

from pathlib import Path

from mai_companion.debug.llm_logger import LLMLoggerProvider
from mai_companion.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    TokenUsage,
)


class _FakeProvider(LLMProvider):
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
        return LLMResponse(
            content="ok",
            model="openai/gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
            finish_reason="stop",
            tool_calls=[],
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
        if False:
            yield None

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        return 42

    async def close(self) -> None:
        return None


class TestLLMLoggerProvider:
    async def test_session_stats_include_cost_tracking(self, tmp_path: Path) -> None:
        provider = LLMLoggerProvider(_FakeProvider(), chat_id="chat-debug", base_dir=tmp_path)

        await provider.generate([ChatMessage(role=MessageRole.USER, content="hello")])
        await provider.generate([ChatMessage(role=MessageRole.USER, content="again")])

        stats = provider.get_session_stats()
        assert stats["llm_calls"] == 2
        assert stats["prompt_tokens"] == 2000
        assert stats["completion_tokens"] == 400
        assert stats["total_tokens"] == 2400
        assert stats["last_call_total_tokens"] == 1200
        assert stats["last_call_cost_usd"] > 0
        assert stats["session_cost_usd"] > stats["last_call_cost_usd"]
