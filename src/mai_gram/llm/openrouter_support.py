"""Internal payload and stream parsing helpers for OpenRouter."""

from __future__ import annotations

import json
from typing import Any

from mai_gram.llm.provider import (
    ChatMessage,
    LLMProviderError,
    LLMResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

EMPTY_STREAM_ERROR = (
    "Stream completed without any data — the provider likely returned "
    "an error in an unsupported format"
)


def serialize_tool_definition(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def serialize_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.reasoning is not None:
        payload["reasoning"] = message.reasoning
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def parse_response(data: dict[str, Any]) -> LLMResponse:
    if "error" in data:
        raise LLMProviderError(f"API error: {_error_message(data['error'])}")

    choices = data.get("choices", [])
    if not choices:
        raise LLMProviderError("No choices in API response")

    message = choices[0].get("message", {})
    usage_data = data.get("usage", {})
    usage = TokenUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )
    return LLMResponse(
        content=message.get("content") or "",
        model=data.get("model", ""),
        usage=usage,
        finish_reason=choices[0].get("finish_reason", ""),
        tool_calls=parse_tool_calls(message.get("tool_calls")),
        reasoning=message.get("reasoning") or None,
    )


def parse_tool_calls(raw_tool_calls: Any) -> list[ToolCall]:
    if not isinstance(raw_tool_calls, list):
        return []

    parsed_calls: list[ToolCall] = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_data = tool_call.get("function")
        if not isinstance(function_data, dict):
            continue

        tool_call_id = tool_call.get("id")
        name = function_data.get("name")
        arguments = function_data.get("arguments")
        if not isinstance(tool_call_id, str) or not isinstance(name, str):
            continue

        parsed_calls.append(
            ToolCall(
                id=tool_call_id,
                name=name,
                arguments=arguments if isinstance(arguments, str) else "",
            )
        )

    return parsed_calls


def parse_inline_stream_error(line: str) -> str | None:
    if not line.startswith("{"):
        return None

    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw, dict) or "error" not in raw:
        return None
    return _error_message(raw["error"])


def decode_sse_json(data_str: str) -> Any | None:
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def parse_stream_chunk(data: Any) -> StreamChunk | None:
    if not isinstance(data, dict):
        return None
    if "error" in data:
        raise LLMProviderError(f"Stream error: {_error_message(data['error'])}")

    choices = data.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta", {})
    content = delta.get("content", "")
    reasoning = delta.get("reasoning", "")
    finish_reason = choices[0].get("finish_reason")
    raw_tool_calls = delta.get("tool_calls")
    tool_calls_delta = (
        raw_tool_calls if isinstance(raw_tool_calls, list) and raw_tool_calls else None
    )

    usage, cost, is_byok = _parse_stream_usage(data.get("usage"))
    if not (content or reasoning or finish_reason or tool_calls_delta or usage):
        return None

    return StreamChunk(
        content=content or "",
        finish_reason=finish_reason,
        reasoning=reasoning or None,
        tool_calls_delta=tool_calls_delta,
        usage=usage,
        cost=cost,
        is_byok=is_byok,
    )


def _parse_stream_usage(usage_data: Any) -> tuple[TokenUsage | None, float | None, bool]:
    if not isinstance(usage_data, dict):
        return None, None, False

    usage = TokenUsage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    )
    is_byok = bool(usage_data.get("is_byok", False))
    raw_cost = usage_data.get("cost") or 0.0
    if not is_byok:
        return usage, float(raw_cost) if raw_cost else None, False

    cost_details = usage_data.get("cost_details") or {}
    inference_cost = (
        cost_details.get("upstream_inference_cost")
        or usage_data.get("native_tokens_cost")
        or usage_data.get("inference_cost")
        or 0.0
    )
    return usage, float(raw_cost) + float(inference_cost), True


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        return str(message) if message is not None else str(error)
    return str(error)
