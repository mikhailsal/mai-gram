"""Tests for LLMLoggerProvider."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from mai_gram.debug.llm_logger import LLMLoggerProvider
from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class _FakeProvider(LLMProvider):
    def __init__(
        self,
        *,
        response: LLMResponse | None = None,
        stream_chunks: list[StreamChunk] | None = None,
    ) -> None:
        self._response = response or LLMResponse(content="ok", model="openai/gpt-4o")
        self._stream_chunks = stream_chunks or []

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return self._response

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self._stream_chunks:
            yield chunk

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        return len(messages)

    async def close(self) -> None:
        return None


def _read_log_entries(log_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_generate_logs_request_response_and_stats(tmp_path: Path) -> None:
    provider = _FakeProvider(
        response=LLMResponse(
            content="hello",
            model="openai/gpt-4o",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            tool_calls=[ToolCall(id="call_1", name="wiki_read", arguments="{}")],
        )
    )
    logger_provider = LLMLoggerProvider(provider, chat_id="chat-1", base_dir=tmp_path)

    response = await logger_provider.generate(
        [ChatMessage(role=MessageRole.USER, content="Hi")],
        model="openai/gpt-4o",
    )

    assert response.content == "hello"
    stats = logger_provider.get_session_stats()
    assert stats["llm_calls"] == 1
    assert stats["calls_with_tool_calls"] == 1
    assert stats["tools_used"] == ["wiki_read"]
    assert logger_provider.latest_log_path is not None
    entries = _read_log_entries(logger_provider.latest_log_path)
    assert entries[0]["entry_type"] == "llm_call"
    assert entries[0]["response"]["tool_calls"][0]["name"] == "wiki_read"


@pytest.mark.asyncio
async def test_generate_stream_logs_aggregated_stream_response(tmp_path: Path) -> None:
    provider = _FakeProvider(
        stream_chunks=[
            StreamChunk(content="Hel", reasoning="Think"),
            StreamChunk(
                content="lo",
                tool_calls_delta=[
                    {"index": 0, "function": {"name": "wiki_"}},
                    {"index": 0, "function": {"name": "search"}},
                ],
            ),
            StreamChunk(
                content="",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=8, completion_tokens=2, total_tokens=10),
            ),
        ]
    )
    logger_provider = LLMLoggerProvider(provider, chat_id="chat-1", base_dir=tmp_path)

    chunks = [
        chunk
        async for chunk in logger_provider.generate_stream(
            [ChatMessage(role=MessageRole.USER, content="Hi")],
            model="openai/gpt-4o",
        )
    ]

    assert [chunk.content for chunk in chunks] == ["Hel", "lo", ""]
    stats = logger_provider.get_session_stats()
    assert stats["llm_calls"] == 1
    assert stats["calls_with_tool_calls"] == 1
    assert stats["tools_used"] == ["wiki_search"]
    assert logger_provider.latest_log_path is not None
    entries = _read_log_entries(logger_provider.latest_log_path)
    assert entries[0]["entry_type"] == "llm_stream_call"
    assert entries[0]["response"]["content"] == "Hello"
    assert entries[0]["response"]["reasoning"] == "Think"
    assert entries[0]["response"]["has_tool_calls"] is True
