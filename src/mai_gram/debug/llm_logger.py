"""Provider proxy that logs structured LLM interactions to JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mai_gram.debug.cost_tracker import SessionCostTracker
from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mai_gram.llm.provider import TokenUsage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _serialize_tool_call(tool_call: ToolCall) -> dict[str, str]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            _serialize_tool_call(tool_call) for tool_call in message.tool_calls
        ]
    return payload


def _serialize_tool_def(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


@dataclass(slots=True)
class _StreamLogState:
    content_parts: list[str]
    reasoning_parts: list[str]
    tool_name_parts: dict[int, list[str]]
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    has_tool_calls: bool = False

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning(self) -> str | None:
        joined = "".join(self.reasoning_parts)
        return joined or None


class LLMLoggerProvider(LLMProvider):
    """LLM provider proxy that captures full request/response details per call."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        chat_id: str,
        base_dir: Path | str = Path("data/debug_logs"),
    ) -> None:
        self._provider = provider
        self._chat_id = chat_id
        self._base_dir = Path(base_dir)
        self._sequence = 0
        self._tool_call_sequence: dict[str, int] = {}
        self._calls_total = 0
        self._calls_with_tools = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._total_tokens_total = 0
        self._tools_used: list[str] = []
        self._last_log_path: Path | None = None
        self._cost_tracker = SessionCostTracker()
        self._last_call_cost_usd = 0.0
        self._last_call_prompt_tokens = 0
        self._last_call_completion_tokens = 0
        self._last_call_total_tokens = 0

    def _timestamp_iso(self) -> str:
        return _utc_now_iso()

    def _log_file_for_timestamp(self, timestamp: str) -> Path:
        date_part = timestamp.split("T", maxsplit=1)[0]
        path = self._base_dir / self._chat_id / f"{date_part}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._last_log_path = path
        return path

    def _append_entry(self, timestamp: str, entry: dict[str, Any]) -> None:
        log_file = self._log_file_for_timestamp(timestamp)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _begin_call(self) -> tuple[int, str]:
        self._sequence += 1
        return self._sequence, self._timestamp_iso()

    @staticmethod
    def _build_request_payload(
        messages: list[ChatMessage],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        tools: list[ToolDefinition] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tool_choice": tool_choice,
            "messages": [_serialize_message(msg) for msg in messages],
            "tools": [_serialize_tool_def(tool) for tool in tools or []],
        }

    def _record_call_stats(
        self,
        *,
        has_tool_calls: bool,
        usage: TokenUsage | None,
        model_name: str | None,
    ) -> None:
        self._calls_total += 1
        if has_tool_calls:
            self._calls_with_tools += 1
        if usage is None:
            return
        self._prompt_tokens_total += usage.prompt_tokens
        self._completion_tokens_total += usage.completion_tokens
        self._total_tokens_total += usage.total_tokens
        self._last_call_prompt_tokens = usage.prompt_tokens
        self._last_call_completion_tokens = usage.completion_tokens
        self._last_call_total_tokens = usage.total_tokens
        call_cost = self._cost_tracker.record(usage, model_name=model_name)
        self._last_call_cost_usd = call_cost.estimated_cost_usd

    def _record_response_tool_calls(
        self,
        tool_calls: list[ToolCall],
        sequence: int,
    ) -> list[dict[str, str]]:
        response_tools = [_serialize_tool_call(tool_call) for tool_call in tool_calls]
        for tool_call in tool_calls:
            self._tool_call_sequence[tool_call.id] = sequence
            if tool_call.name not in self._tools_used:
                self._tools_used.append(tool_call.name)
        return response_tools

    def _record_stream_tool_names(self, state: _StreamLogState) -> None:
        for index in sorted(state.tool_name_parts):
            tool_name = "".join(state.tool_name_parts[index])
            if tool_name and tool_name not in self._tools_used:
                self._tools_used.append(tool_name)

    def _append_llm_call_entry(
        self,
        *,
        entry_type: str,
        sequence: int,
        timestamp: str,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        self._append_entry(
            timestamp,
            {
                "entry_type": entry_type,
                "sequence": sequence,
                "timestamp": timestamp,
                "chat_id": self._chat_id,
                "request": request,
                "response": response,
            },
        )

    @staticmethod
    def _usage_payload(usage: TokenUsage | None) -> dict[str, int]:
        return {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

    @staticmethod
    def _record_stream_chunk(state: _StreamLogState, chunk: StreamChunk) -> None:
        if chunk.content:
            state.content_parts.append(chunk.content)
        if chunk.reasoning:
            state.reasoning_parts.append(chunk.reasoning)
        if chunk.tool_calls_delta:
            state.has_tool_calls = True
            for delta in chunk.tool_calls_delta:
                if not isinstance(delta, dict):
                    continue
                function = delta.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(name, str) or not name:
                    continue
                index = delta.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                state.tool_name_parts.setdefault(index, []).append(name)
        if chunk.finish_reason:
            state.finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            state.usage = chunk.usage

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        sequence, timestamp = self._begin_call()
        request = self._build_request_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = await self._provider.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            extra_params=extra_params,
        )
        self._record_call_stats(
            has_tool_calls=bool(response.tool_calls),
            usage=response.usage,
            model_name=response.model,
        )
        response_tools = self._record_response_tool_calls(response.tool_calls, sequence)
        self._append_llm_call_entry(
            entry_type="llm_call",
            sequence=sequence,
            timestamp=timestamp,
            request=request,
            response={
                "model": response.model,
                "content": response.content,
                "finish_reason": response.finish_reason,
                "tool_calls": response_tools,
                "usage": self._usage_payload(response.usage),
            },
        )
        return response

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        sequence, timestamp = self._begin_call()
        request = self._build_request_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        stream_state = _StreamLogState(
            content_parts=[],
            reasoning_parts=[],
            tool_name_parts={},
        )

        async for chunk in self._provider.generate_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            extra_params=extra_params,
        ):
            self._record_stream_chunk(stream_state, chunk)
            yield chunk

        self._record_call_stats(
            has_tool_calls=stream_state.has_tool_calls,
            usage=stream_state.usage,
            model_name=model,
        )
        self._record_stream_tool_names(stream_state)
        self._append_llm_call_entry(
            entry_type="llm_stream_call",
            sequence=sequence,
            timestamp=timestamp,
            request=request,
            response={
                "content": stream_state.content,
                "reasoning": stream_state.reasoning,
                "finish_reason": stream_state.finish_reason,
                "has_tool_calls": stream_state.has_tool_calls,
                "usage": self._usage_payload(stream_state.usage),
            },
        )

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        return await self._provider.count_tokens(messages, model=model)

    async def close(self) -> None:
        await self._provider.close()

    def record_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: str | dict[str, Any] | None,
        result: Any,
        error: str | None = None,
        server_name: str | None = None,
    ) -> None:
        """Append one tool execution event, linked to the originating LLM call."""
        timestamp = self._timestamp_iso()
        sequence = self._tool_call_sequence.get(tool_call_id)
        entry = {
            "entry_type": "tool_result",
            "timestamp": timestamp,
            "chat_id": self._chat_id,
            "llm_sequence": sequence,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "server_name": server_name,
            "arguments": arguments,
            "result": result,
            "error": error,
        }
        self._append_entry(timestamp, entry)

    @property
    def latest_log_path(self) -> Path | None:
        return self._last_log_path

    def get_session_stats(self) -> dict[str, Any]:
        cost_stats = self._cost_tracker.stats()
        return {
            "llm_calls": self._calls_total,
            "calls_with_tool_calls": self._calls_with_tools,
            "tools_used": list(self._tools_used),
            "prompt_tokens": self._prompt_tokens_total,
            "completion_tokens": self._completion_tokens_total,
            "total_tokens": self._total_tokens_total,
            "last_call_prompt_tokens": self._last_call_prompt_tokens,
            "last_call_completion_tokens": self._last_call_completion_tokens,
            "last_call_total_tokens": self._last_call_total_tokens,
            "last_call_cost_usd": self._last_call_cost_usd,
            "session_cost_usd": float(cost_stats["estimated_cost_usd"]),
            "cost_calls": int(cost_stats["calls"]),
            "log_path": str(self._last_log_path) if self._last_log_path is not None else None,
        }
