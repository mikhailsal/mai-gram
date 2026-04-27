"""Bridge between MCP tools and OpenAI-style tool calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    StreamChunk,
)
from mai_gram.mcp_servers.bridge_support import (
    MCPToolCall as MCPToolCall,
)
from mai_gram.mcp_servers.bridge_support import (
    ToolLoopConfig,
    UsageAccumulator,
    build_tool_definitions,
    complete_tool_turn,
    create_stream_iteration_state,
    generate_response,
    process_stream_tool_turn,
    stream_final_response,
    stream_iteration,
)
from mai_gram.mcp_servers.bridge_support import (
    mcp_result_to_openai as mcp_result_to_openai,
)
from mai_gram.mcp_servers.bridge_support import (
    openai_tool_call_to_mcp as openai_tool_call_to_mcp,
)
from mai_gram.mcp_servers.bridge_support import (
    serialize_tool_calls as serialize_tool_calls,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from mai_gram.mcp_servers.manager import MCPManager, RegisteredTool


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
    """Run the non-streaming LLM -> tools -> LLM loop."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    conversation = list(messages)
    tool_defs = build_tool_definitions(await manager.list_all_tools())
    config = ToolLoopConfig(model, temperature, max_tokens, tool_choice, extra_params)

    last_response: LLMResponse | None = None
    for _ in range(max_iterations):
        response = await generate_response(llm, conversation, tool_defs, config)
        last_response = response

        if not response.tool_calls:
            return response

        await complete_tool_turn(
            conversation,
            content=response.content or "",
            tool_calls=response.tool_calls,
            reasoning=response.reasoning,
            manager=manager,
            on_tool_result=on_tool_result,
            on_intermediate_content=on_intermediate_content,
            on_assistant_tool_call=on_assistant_tool_call,
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
    """Run the streaming LLM -> tools -> LLM loop."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    conversation = list(messages)
    tool_defs = build_tool_definitions(await manager.list_all_tools())
    config = ToolLoopConfig(model, temperature, max_tokens, tool_choice, extra_params)
    usage_totals = UsageAccumulator()

    for _ in range(max_iterations):
        state = create_stream_iteration_state()
        async for chunk in stream_iteration(llm, conversation, tool_defs, config, state):
            yield chunk
        usage_totals.add(state.usage, state.cost, state.is_byok)

        should_continue, final_or_marker = await process_stream_tool_turn(
            conversation,
            state,
            manager,
            usage_totals,
            on_tool_result=on_tool_result,
            on_intermediate_content=on_intermediate_content,
            on_assistant_tool_call=on_assistant_tool_call,
        )
        yield final_or_marker
        if not should_continue:
            return

    async for chunk in stream_final_response(llm, conversation, usage_totals, config):
        yield chunk

    yield usage_totals.final_chunk("max_tool_iterations")
