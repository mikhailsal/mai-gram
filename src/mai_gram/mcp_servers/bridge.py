"""Bridge between MCP tools and OpenAI-style tool calling."""

from __future__ import annotations

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


def mcp_tools_to_openai(tools: list[RegisteredTool]) -> list[dict[str, Any]]:
    """Convert registered MCP tools to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


@dataclass(frozen=True, slots=True)
class MCPToolCall:
    """Resolved tool call destination and arguments."""

    server_name: str
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


async def openai_tool_call_to_mcp(tool_call: ToolCall, manager: MCPManager) -> MCPToolCall:
    """Resolve an OpenAI ToolCall into a concrete MCP manager call."""
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
    """Convert a tool result into text suitable for a `tool` role message."""
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    return json.dumps(result, ensure_ascii=False, default=str)


def serialize_tool_calls(tool_calls: list[ToolCall]) -> str:
    """Serialize tool calls to JSON for database storage."""
    return json.dumps(
        [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
        ensure_ascii=False,
    )


async def run_with_tools(
    llm: LLMProvider,
    manager: MCPManager,
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_iterations: int = 5,
    tool_choice: str | dict[str, Any] | None = "auto",
    extra_params: dict[str, Any] | None = None,
    on_tool_result: Callable[..., Awaitable[None] | None] | None = None,
    on_intermediate_content: Callable[[str], Awaitable[None] | None] | None = None,
    on_assistant_tool_call: Callable[..., Awaitable[None] | None] | None = None,
) -> LLMResponse:
    """Run an agentic loop: LLM -> tools -> LLM until completion.

    Parameters
    ----------
    on_intermediate_content:
        Called when the LLM produces text alongside tool calls.  This
        enables multi-message behaviour: the text is dispatched as a
        standalone message *before* the tool executes, so the human
        sees it immediately.
    on_assistant_tool_call:
        Called when the assistant produces a message with tool calls.
        Receives: content (str), tool_calls_json (str).
        Use this to persist the assistant's tool-calling decision to the database.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    conversation = list(messages)
    registered_tools = await manager.list_all_tools()
    tool_defs = [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in registered_tools
    ]

    intermediate_texts: list[str] = []
    last_response: LLMResponse | None = None
    for _ in range(max_iterations):
        response = await llm.generate(
            conversation,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_defs or None,
            tool_choice=tool_choice if tool_defs else None,
            extra_params=extra_params,
        )
        last_response = response

        if not response.tool_calls:
            return response

        # When the LLM produces text alongside tool calls, treat it as
        # an intermediate message that should be sent to the human now.
        if response.content and response.content.strip():
            intermediate_texts.append(response.content.strip())
            if on_intermediate_content is not None:
                cb_result = on_intermediate_content(response.content.strip())
                if inspect.isawaitable(cb_result):
                    await cb_result

        # Notify caller about the assistant's tool-calling decision for persistence.
        # This allows the full conversation flow (including tool calls) to be saved.
        if on_assistant_tool_call is not None:
            tool_calls_json = serialize_tool_calls(response.tool_calls)
            cb_result = on_assistant_tool_call(
                content=response.content or "",
                tool_calls_json=tool_calls_json,
            )
            if inspect.isawaitable(cb_result):
                await cb_result

        conversation.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
                reasoning=response.reasoning,
            )
        )

        for tool_call in response.tool_calls:
            try:
                resolved = await openai_tool_call_to_mcp(tool_call, manager)
                raw_result = await manager.call_tool(
                    resolved.server_name,
                    resolved.tool_name,
                    resolved.arguments,
                )
                tool_content = mcp_result_to_openai(raw_result)
                tool_error = None
            except Exception as exc:  # pragma: no cover - exercised in tests
                tool_content = f"Tool execution error: {exc}"
                raw_result = None
                resolved = None
                tool_error = str(exc)

            if on_tool_result is not None:
                callback_result = on_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=raw_result,
                    content=tool_content,
                    error=tool_error,
                    server_name=resolved.server_name if resolved is not None else None,
                )
                if inspect.isawaitable(callback_result):
                    await callback_result

            conversation.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_content,
                    tool_call_id=tool_call.id,
                )
            )

    if last_response is None:  # pragma: no cover
        raise RuntimeError("run_with_tools executed without producing a response")

    return LLMResponse(
        content=last_response.content,
        model=last_response.model,
        usage=last_response.usage,
        finish_reason="max_tool_iterations",
        tool_calls=last_response.tool_calls,
    )


def _reassemble_tool_calls_from_deltas(
    deltas: list[list[dict[str, Any]]],
) -> list[ToolCall]:
    """Reconstruct complete ToolCall objects from streaming delta fragments.

    OpenAI-compatible streaming sends tool calls as incremental deltas
    keyed by ``index``.  Each delta may contain a partial ``id``,
    ``function.name``, or ``function.arguments`` that must be concatenated
    across all chunks sharing the same index.
    """
    by_index: dict[int, dict[str, str]] = {}
    for batch in deltas:
        for item in batch:
            idx = item.get("index", 0)
            if idx not in by_index:
                by_index[idx] = {"id": "", "name": "", "arguments": ""}

            entry = by_index[idx]
            if "id" in item and item["id"]:
                entry["id"] = item["id"]
            func = item.get("function") or {}
            if func.get("name"):
                entry["name"] += func["name"]
            if func.get("arguments"):
                entry["arguments"] += func["arguments"]

    result: list[ToolCall] = []
    for idx in sorted(by_index):
        e = by_index[idx]
        if e["id"] and e["name"]:
            result.append(ToolCall(id=e["id"], name=e["name"], arguments=e["arguments"]))
    return result


async def run_with_tools_stream(
    llm: LLMProvider,
    manager: MCPManager,
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_iterations: int = 5,
    tool_choice: str | dict[str, Any] | None = "auto",
    extra_params: dict[str, Any] | None = None,
    on_tool_result: Callable[..., Awaitable[None] | None] | None = None,
    on_intermediate_content: Callable[[str], Awaitable[None] | None] | None = None,
    on_assistant_tool_call: Callable[..., Awaitable[None] | None] | None = None,
) -> AsyncIterator[StreamChunk]:
    """Run the agentic tool loop, streaming every LLM response.

    Every LLM call uses ``generate_stream()``.  Content/reasoning
    chunks are yielded to the caller immediately for live display.
    If the stream also contains tool-call deltas, they are buffered
    and reassembled; the tools are executed and the loop continues.

    Yields ``StreamChunk`` objects — the caller sees tokens as soon
    as the provider emits them.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    conversation = list(messages)
    registered_tools = await manager.list_all_tools()
    tool_defs = [
        ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in registered_tools
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost: float | None = None
    last_is_byok = False

    for _ in range(max_iterations):
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_deltas: list[list[dict[str, Any]]] = []
        finish_reason: str | None = None
        iter_usage: TokenUsage | None = None
        iter_cost: float | None = None

        async for chunk in llm.generate_stream(
            conversation,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_defs or None,
            tool_choice=tool_choice if tool_defs else None,
            extra_params=extra_params,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if chunk.tool_calls_delta:
                tool_call_deltas.append(chunk.tool_calls_delta)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                iter_usage = chunk.usage
                iter_cost = chunk.cost
                last_is_byok = chunk.is_byok

            if chunk.content or chunk.reasoning:
                yield chunk

        if iter_usage is not None:
            total_prompt_tokens += iter_usage.prompt_tokens
            total_completion_tokens += iter_usage.completion_tokens
            if iter_cost is not None:
                total_cost = (total_cost or 0.0) + iter_cost

        assembled_tool_calls = _reassemble_tool_calls_from_deltas(tool_call_deltas)

        if not assembled_tool_calls:
            agg_usage = TokenUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )
            yield StreamChunk(
                content="",
                finish_reason=finish_reason or "stop",
                usage=agg_usage,
                cost=total_cost,
                is_byok=last_is_byok,
            )
            return

        yield StreamChunk(content="", turn_complete=True)

        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts) or None

        if full_content.strip() and on_intermediate_content is not None:
            cb_result = on_intermediate_content(full_content.strip())
            if inspect.isawaitable(cb_result):
                await cb_result

        if on_assistant_tool_call is not None:
            tool_calls_json = serialize_tool_calls(assembled_tool_calls)
            cb_result = on_assistant_tool_call(
                content=full_content,
                tool_calls_json=tool_calls_json,
            )
            if inspect.isawaitable(cb_result):
                await cb_result

        conversation.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=full_content,
                tool_calls=assembled_tool_calls,
                reasoning=full_reasoning,
            )
        )

        for tool_call in assembled_tool_calls:
            try:
                resolved = await openai_tool_call_to_mcp(tool_call, manager)
                raw_result = await manager.call_tool(
                    resolved.server_name,
                    resolved.tool_name,
                    resolved.arguments,
                )
                tool_content = mcp_result_to_openai(raw_result)
                tool_error = None
            except Exception as exc:
                tool_content = f"Tool execution error: {exc}"
                raw_result = None
                resolved = None
                tool_error = str(exc)

            if on_tool_result is not None:
                callback_result = on_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    result=raw_result,
                    content=tool_content,
                    error=tool_error,
                    server_name=resolved.server_name if resolved is not None else None,
                )
                if inspect.isawaitable(callback_result):
                    await callback_result

            conversation.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=tool_content,
                    tool_call_id=tool_call.id,
                )
            )

    # Max iterations exhausted -- stream a final response without tools
    logger.warning("Tool loop hit max_iterations=%d; streaming final response", max_iterations)
    async for chunk in llm.generate_stream(
        conversation,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
    ):
        if chunk.usage is not None:
            total_prompt_tokens += chunk.usage.prompt_tokens
            total_completion_tokens += chunk.usage.completion_tokens
            if chunk.cost is not None:
                total_cost = (total_cost or 0.0) + chunk.cost
            last_is_byok = chunk.is_byok
        yield chunk

    agg_usage = TokenUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )
    yield StreamChunk(
        content="",
        finish_reason="max_tool_iterations",
        usage=agg_usage,
        cost=total_cost,
        is_byok=last_is_byok,
    )
