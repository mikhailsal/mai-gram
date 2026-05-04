"""Tests for OpenRouter support helpers (payload building and stream parsing)."""

from __future__ import annotations

import pytest

from mai_gram.llm.openrouter_support import (
    _error_message,
    _parse_stream_usage,
    decode_sse_json,
    parse_inline_stream_error,
    parse_response,
    parse_stream_chunk,
    parse_tool_calls,
    serialize_message,
    serialize_tool_definition,
)
from mai_gram.llm.provider import (
    ChatMessage,
    LLMProviderError,
    MessageRole,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class TestSerializeMessage:
    def test_basic_user_message(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        result = serialize_message(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_message_with_reasoning(self) -> None:
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Answer", reasoning="thought")
        result = serialize_message(msg)
        assert result["reasoning"] == "thought"

    def test_message_with_tool_call_id(self) -> None:
        msg = ChatMessage(role=MessageRole.TOOL, content="result", tool_call_id="tc_1")
        result = serialize_message(msg)
        assert result["tool_call_id"] == "tc_1"

    def test_message_with_tool_calls(self) -> None:
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[ToolCall(id="tc_1", name="search", arguments='{"q":"test"}')],
        )
        result = serialize_message(msg)
        assert result["tool_calls"] == [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"test"}'},
            }
        ]


class TestSerializeToolDefinition:
    def test_serializes_tool(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        result = serialize_tool_definition(tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }


class TestParseToolCalls:
    def test_none_input(self) -> None:
        assert parse_tool_calls(None) == []

    def test_non_list_input(self) -> None:
        assert parse_tool_calls("not a list") == []

    def test_non_dict_elements(self) -> None:
        assert parse_tool_calls(["not_a_dict", 42]) == []

    def test_missing_function_data(self) -> None:
        assert parse_tool_calls([{"id": "1"}]) == []

    def test_non_dict_function_data(self) -> None:
        assert parse_tool_calls([{"id": "1", "function": "not_dict"}]) == []

    def test_non_string_id(self) -> None:
        result = parse_tool_calls([{"id": 123, "function": {"name": "foo", "arguments": ""}}])
        assert result == []

    def test_non_string_name(self) -> None:
        result = parse_tool_calls([{"id": "1", "function": {"name": None, "arguments": ""}}])
        assert result == []

    def test_non_string_arguments_coerced(self) -> None:
        result = parse_tool_calls([{"id": "1", "function": {"name": "foo", "arguments": 42}}])
        assert len(result) == 1
        assert result[0].arguments == ""

    def test_valid_tool_call(self) -> None:
        raw = [{"id": "tc_1", "function": {"name": "search", "arguments": '{"q":"hi"}'}}]
        result = parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0].id == "tc_1"
        assert result[0].name == "search"
        assert result[0].arguments == '{"q":"hi"}'


class TestParseInlineStreamError:
    def test_non_json_prefix(self) -> None:
        assert parse_inline_stream_error("not json") is None

    def test_invalid_json(self) -> None:
        assert parse_inline_stream_error("{bad json") is None

    def test_no_error_key(self) -> None:
        assert parse_inline_stream_error('{"status":"ok"}') is None

    def test_non_dict_json(self) -> None:
        assert parse_inline_stream_error("[1, 2, 3]") is None

    def test_error_dict(self) -> None:
        result = parse_inline_stream_error('{"error":{"message":"rate limited"}}')
        assert result == "rate limited"

    def test_error_string(self) -> None:
        result = parse_inline_stream_error('{"error":"something failed"}')
        assert result == "something failed"


class TestDecodeSSEJson:
    def test_valid_json(self) -> None:
        assert decode_sse_json('{"key":"value"}') == {"key": "value"}

    def test_invalid_json(self) -> None:
        assert decode_sse_json("not json") is None


class TestParseStreamChunk:
    def test_non_dict_data(self) -> None:
        assert parse_stream_chunk("not a dict") is None

    def test_error_in_data(self) -> None:
        with pytest.raises(LLMProviderError, match="Stream error"):
            parse_stream_chunk({"error": {"message": "failed"}})

    def test_empty_choices(self) -> None:
        assert parse_stream_chunk({"choices": []}) is None

    def test_empty_delta(self) -> None:
        assert parse_stream_chunk({"choices": [{"delta": {}}]}) is None

    def test_content_chunk(self) -> None:
        data = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        chunk = parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.content == "hello"

    def test_reasoning_chunk(self) -> None:
        data = {"choices": [{"delta": {"reasoning": "thinking..."}, "finish_reason": None}]}
        chunk = parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.reasoning == "thinking..."

    def test_finish_reason_chunk(self) -> None:
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        chunk = parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.finish_reason == "stop"

    def test_tool_calls_delta(self) -> None:
        data = {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "id": "tc_1"}]},
                    "finish_reason": None,
                }
            ]
        }
        chunk = parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.tool_calls_delta is not None

    def test_tool_calls_empty_list_ignored(self) -> None:
        data = {"choices": [{"delta": {"tool_calls": []}, "finish_reason": None}]}
        assert parse_stream_chunk(data) is None

    def test_usage_in_chunk(self) -> None:
        data = {
            "choices": [{"delta": {}, "finish_reason": None}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        chunk = parse_stream_chunk(data)
        assert chunk is not None
        assert chunk.usage == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class TestParseStreamUsage:
    def test_non_dict_returns_none(self) -> None:
        usage, cost, is_byok = _parse_stream_usage(None)
        assert usage is None
        assert cost is None
        assert is_byok is False

    def test_normal_usage(self) -> None:
        data = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.005}
        usage, cost, is_byok = _parse_stream_usage(data)
        assert usage == TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert cost == 0.005
        assert is_byok is False

    def test_byok_usage_with_cost_details(self) -> None:
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.003,
            "is_byok": True,
            "cost_details": {"upstream_inference_cost": 0.002},
        }
        _usage, cost, is_byok = _parse_stream_usage(data)
        assert is_byok is True
        assert cost == 0.005  # 0.003 + 0.002

    def test_byok_usage_native_tokens_cost_fallback(self) -> None:
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
            "is_byok": True,
            "native_tokens_cost": 0.004,
        }
        _usage, cost, is_byok = _parse_stream_usage(data)
        assert is_byok is True
        assert cost == 0.005  # 0.001 + 0.004

    def test_byok_usage_inference_cost_fallback(self) -> None:
        data = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
            "is_byok": True,
            "inference_cost": 0.006,
        }
        _usage, cost, is_byok = _parse_stream_usage(data)
        assert is_byok is True
        assert cost == 0.007  # 0.001 + 0.006

    def test_zero_cost(self) -> None:
        data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage, cost, is_byok = _parse_stream_usage(data)
        assert usage is not None
        assert cost is None
        assert is_byok is False


class TestParseResponse:
    def test_error_response(self) -> None:
        with pytest.raises(LLMProviderError, match="API error"):
            parse_response({"error": {"message": "invalid key"}})

    def test_empty_choices(self) -> None:
        with pytest.raises(LLMProviderError, match="No choices"):
            parse_response({"choices": []})

    def test_successful_response(self) -> None:
        data = {
            "choices": [
                {"message": {"content": "Hello!", "reasoning": "thought"}, "finish_reason": "stop"}
            ],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = parse_response(data)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.reasoning == "thought"

    def test_response_with_tool_calls(self) -> None:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {"name": "search", "arguments": '{"q":"test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = parse_response(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"


class TestErrorMessage:
    def test_dict_with_message(self) -> None:
        assert _error_message({"message": "something broke"}) == "something broke"

    def test_dict_without_message(self) -> None:
        result = _error_message({"code": 500})
        assert "500" in result

    def test_dict_with_none_message(self) -> None:
        result = _error_message({"message": None})
        assert "None" not in result or result == str({"message": None})

    def test_string_error(self) -> None:
        assert _error_message("plain error") == "plain error"
