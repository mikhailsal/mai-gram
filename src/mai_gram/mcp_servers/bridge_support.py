"""Internal helpers for MCP bridge tool loops."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from mai_gram.mcp_servers.manager import MCPManager, RegisteredTool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCPToolCall:
    server_name: str
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolLoopConfig:
    model: str | None
    temperature: float
    max_tokens: int | None
    tool_choice: str | dict[str, Any] | None
    extra_params: dict[str, Any] | None


@dataclass(slots=True)
class _StreamIterationState:
    content_parts: list[str]
    reasoning_parts: list[str]
    tool_call_deltas: list[list[dict[str, Any]]]
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    cost: float | None = None
    is_byok: bool = False

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning(self) -> str | None:
        joined = "".join(self.reasoning_parts)
        return joined or None

    @classmethod
    def create(cls) -> _StreamIterationState:
        return cls(content_parts=[], reasoning_parts=[], tool_call_deltas=[])


def create_stream_iteration_state() -> _StreamIterationState:
    return _StreamIterationState.create()


@dataclass(slots=True)
class UsageAccumulator:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    is_byok: bool = False

    def add(self, usage: TokenUsage | None, cost: float | None, is_byok: bool) -> None:
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        if cost is not None:
            self.cost = (self.cost or 0.0) + cost
        self.is_byok = is_byok

    def final_chunk(self, finish_reason: str) -> StreamChunk:
        return StreamChunk(
            content="",
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
            cost=self.cost,
            is_byok=self.is_byok,
        )


def build_tool_definitions(tools: list[RegisteredTool]) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in tools
    ]


async def generate_response(
    llm: LLMProvider,
    conversation: list[ChatMessage],
    tool_defs: list[ToolDefinition],
    config: ToolLoopConfig,
) -> LLMResponse:
    return await llm.generate(
        conversation,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tools=tool_defs or None,
        tool_choice=config.tool_choice if tool_defs else None,
        extra_params=config.extra_params,
    )


async def openai_tool_call_to_mcp(tool_call: ToolCall, manager: MCPManager) -> MCPToolCall:
    if tool_call.arguments.strip():
        try:
            parsed = json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for tool '{tool_call.name}': {exc.msg}") from exc
    else:
        parsed = {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments for '{tool_call.name}' must be a JSON object")

    server_name = await manager.resolve_tool_server(tool_call.name)
    return MCPToolCall(
        server_name=server_name,
        tool_name=tool_call.name,
        arguments=parsed,
        tool_call_id=tool_call.id,
    )


def mcp_result_to_openai(result: Any) -> str:
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False, default=str)


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if inspect.isawaitable(result):
        await result


async def _emit_intermediate_content(
    content: str,
    callback: Callable[[str], Awaitable[None] | None] | None,
) -> None:
    if callback is None or not content.strip():
        return
    await _maybe_await(callback(content.strip()))


async def _emit_assistant_tool_call(
    *,
    content: str,
    tool_calls: list[ToolCall],
    callback: Callable[..., Awaitable[None] | None] | None,
) -> None:
    if callback is None:
        return
    await _maybe_await(
        callback(
            content=content,
            tool_calls=tool_calls,
        )
    )


def _append_assistant_tool_message(
    conversation: list[ChatMessage],
    *,
    content: str,
    tool_calls: list[ToolCall],
    reasoning: str | None,
) -> None:
    conversation.append(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
        )
    )


async def _execute_tool_call(
    tool_call: ToolCall,
    manager: MCPManager,
    *,
    on_tool_result: Callable[..., Awaitable[None] | None] | None,
) -> str:
    try:
        resolved = await openai_tool_call_to_mcp(tool_call, manager)
        raw_result = await manager.call_tool(
            resolved.server_name,
            resolved.tool_name,
            resolved.arguments,
        )
        tool_content = mcp_result_to_openai(raw_result)
        tool_error = None
    except (
        ValueError,
        RuntimeError,
        OSError,
        LookupError,
        json.JSONDecodeError,
        asyncio.TimeoutError,
    ) as exc:
        tool_content = f"Tool execution error: {exc}"
        raw_result = None
        resolved = None
        tool_error = str(exc)

    if on_tool_result is not None:
        await _maybe_await(
            on_tool_result(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=raw_result,
                content=tool_content,
                error=tool_error,
                server_name=resolved.server_name if resolved is not None else None,
            )
        )

    return tool_content


async def _append_tool_results(
    conversation: list[ChatMessage],
    tool_calls: list[ToolCall],
    manager: MCPManager,
    *,
    on_tool_result: Callable[..., Awaitable[None] | None] | None,
) -> None:
    for tool_call in tool_calls:
        tool_content = await _execute_tool_call(
            tool_call,
            manager,
            on_tool_result=on_tool_result,
        )
        conversation.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=tool_content,
                tool_call_id=tool_call.id,
            )
        )


async def complete_tool_turn(
    conversation: list[ChatMessage],
    *,
    content: str,
    tool_calls: list[ToolCall],
    reasoning: str | None,
    manager: MCPManager,
    on_tool_result: Callable[..., Awaitable[None] | None] | None,
    on_intermediate_content: Callable[[str], Awaitable[None] | None] | None,
    on_assistant_tool_call: Callable[..., Awaitable[None] | None] | None,
) -> None:
    await _emit_intermediate_content(content, on_intermediate_content)
    await _emit_assistant_tool_call(
        content=content,
        tool_calls=tool_calls,
        callback=on_assistant_tool_call,
    )
    _append_assistant_tool_message(
        conversation,
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
    )
    await _append_tool_results(
        conversation,
        tool_calls,
        manager,
        on_tool_result=on_tool_result,
    )


def reassemble_tool_calls_from_deltas(
    deltas: list[list[dict[str, Any]]],
) -> list[ToolCall]:
    by_index: dict[int, dict[str, str]] = {}
    for batch in deltas:
        for item in batch:
            idx = item.get("index", 0)
            if idx not in by_index:
                by_index[idx] = {"id": "", "name": "", "arguments": ""}

            entry = by_index[idx]
            if item.get("id"):
                entry["id"] = item["id"]
            func = item.get("function") or {}
            if func.get("name"):
                entry["name"] += func["name"]
            if func.get("arguments"):
                entry["arguments"] += func["arguments"]

    result: list[ToolCall] = []
    for idx in sorted(by_index):
        entry = by_index[idx]
        if entry["id"] and entry["name"]:
            result.append(
                ToolCall(
                    id=entry["id"],
                    name=entry["name"],
                    arguments=entry["arguments"],
                )
            )
    return result


async def stream_iteration(
    llm: LLMProvider,
    conversation: list[ChatMessage],
    tool_defs: list[ToolDefinition],
    config: ToolLoopConfig,
    state: _StreamIterationState,
) -> AsyncIterator[StreamChunk]:
    async for chunk in llm.generate_stream(
        conversation,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        tools=tool_defs or None,
        tool_choice=config.tool_choice if tool_defs else None,
        extra_params=config.extra_params,
    ):
        if chunk.content:
            state.content_parts.append(chunk.content)
        if chunk.reasoning:
            state.reasoning_parts.append(chunk.reasoning)
        if chunk.tool_calls_delta:
            state.tool_call_deltas.append(chunk.tool_calls_delta)
        if chunk.finish_reason:
            state.finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            state.usage = chunk.usage
            state.cost = chunk.cost
            state.is_byok = chunk.is_byok
        if chunk.content or chunk.reasoning:
            yield chunk


async def process_stream_tool_turn(
    conversation: list[ChatMessage],
    state: _StreamIterationState,
    manager: MCPManager,
    usage_totals: UsageAccumulator,
    *,
    on_tool_result: Callable[..., Awaitable[None] | None] | None,
    on_intermediate_content: Callable[[str], Awaitable[None] | None] | None,
    on_assistant_tool_call: Callable[..., Awaitable[None] | None] | None,
) -> tuple[bool, StreamChunk]:
    tool_calls = reassemble_tool_calls_from_deltas(state.tool_call_deltas)
    if not tool_calls:
        return False, usage_totals.final_chunk(state.finish_reason or "stop")

    await complete_tool_turn(
        conversation,
        content=state.content,
        tool_calls=tool_calls,
        reasoning=state.reasoning,
        manager=manager,
        on_tool_result=on_tool_result,
        on_intermediate_content=on_intermediate_content,
        on_assistant_tool_call=on_assistant_tool_call,
    )
    return True, StreamChunk(content="", turn_complete=True)


async def stream_final_response(
    llm: LLMProvider,
    conversation: list[ChatMessage],
    usage_totals: UsageAccumulator,
    config: ToolLoopConfig,
) -> AsyncIterator[StreamChunk]:
    logger.warning("Tool loop hit max_iterations; streaming final response")
    async for chunk in llm.generate_stream(
        conversation,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        extra_params=config.extra_params,
    ):
        usage_totals.add(chunk.usage, chunk.cost, chunk.is_byok)
        yield chunk
