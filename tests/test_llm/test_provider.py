"""Tests for the abstract LLM provider and data classes."""

from __future__ import annotations

import pytest

from mai_gram.llm.provider import (
    ChatMessage,
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class TestMessageRole:
    """MessageRole enum works as expected."""

    def test_values(self) -> None:
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_is_str_subclass(self) -> None:
        assert isinstance(MessageRole.USER, str)


class TestChatMessage:
    """ChatMessage dataclass basics."""

    def test_creation(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="Hello!")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_with_tool_calls(self) -> None:
        calls = [ToolCall(id="call_1", name="search_messages", arguments='{"query":"paris"}')]
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=calls)
        assert msg.tool_calls == calls

    def test_with_tool_call_id(self) -> None:
        msg = ChatMessage(role=MessageRole.TOOL, content="result", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"

    def test_is_frozen(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="Hi")
        with pytest.raises(AttributeError):
            msg.content = "Modified"  # type: ignore[misc]


class TestTokenUsage:
    """TokenUsage dataclass."""

    def test_defaults_to_zero(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30


class TestLLMResponse:
    """LLMResponse dataclass."""

    def test_creation(self) -> None:
        calls = [ToolCall(id="call_1", name="search_messages", arguments='{"query":"paris"}')]
        resp = LLMResponse(
            content="Hello, world!",
            model="openai/gpt-4o",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            finish_reason="stop",
            tool_calls=calls,
        )
        assert resp.content == "Hello, world!"
        assert resp.model == "openai/gpt-4o"
        assert resp.usage.total_tokens == 8
        assert resp.finish_reason == "stop"
        assert resp.tool_calls == calls

    def test_defaults(self) -> None:
        resp = LLMResponse(content="Hi", model="test")
        assert resp.usage.total_tokens == 0
        assert resp.finish_reason == ""
        assert resp.tool_calls == []


class TestToolStructures:
    """ToolDefinition and ToolCall dataclasses."""

    def test_tool_definition_creation(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search messages",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        assert tool.name == "search"
        assert tool.description == "Search messages"
        assert tool.parameters["type"] == "object"

    def test_tool_call_creation(self) -> None:
        call = ToolCall(id="call_abc", name="search_messages", arguments='{"query":"hello"}')
        assert call.id == "call_abc"
        assert call.name == "search_messages"
        assert call.arguments == '{"query":"hello"}'


class TestStreamChunk:
    """StreamChunk dataclass."""

    def test_content_only(self) -> None:
        chunk = StreamChunk(content="Hel")
        assert chunk.content == "Hel"
        assert chunk.finish_reason is None

    def test_with_finish_reason(self) -> None:
        chunk = StreamChunk(content="", finish_reason="stop")
        assert chunk.finish_reason == "stop"


class TestErrorHierarchy:
    """All LLM errors inherit from LLMError."""

    def test_base_error(self) -> None:
        with pytest.raises(LLMError):
            raise LLMError("generic")

    def test_auth_error(self) -> None:
        with pytest.raises(LLMError):
            raise LLMAuthenticationError("bad key")

    def test_rate_limit_with_retry_after(self) -> None:
        exc = LLMRateLimitError("slow down", retry_after=30.0)
        assert exc.retry_after == 30.0
        assert isinstance(exc, LLMError)

    def test_model_not_found(self) -> None:
        assert issubclass(LLMModelNotFoundError, LLMError)

    def test_context_length(self) -> None:
        assert issubclass(LLMContextLengthError, LLMError)

    def test_provider_error_status_code(self) -> None:
        exc = LLMProviderError("boom", status_code=500)
        assert exc.status_code == 500
        assert isinstance(exc, LLMError)
