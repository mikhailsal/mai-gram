"""Bridge between MCP tools and OpenAI-style tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mai_companion.llm.provider import ChatMessage, LLMProvider, LLMResponse, MessageRole, ToolCall, ToolDefinition
from mai_companion.mcp_servers.manager import MCPManager, RegisteredTool


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


async def run_with_tools(
    llm: LLMProvider,
    manager: MCPManager,
    messages: list[ChatMessage],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    max_iterations: int = 5,
    tool_choice: str | dict | None = "auto",
) -> LLMResponse:
    """Run an agentic loop: LLM -> tools -> LLM until completion."""
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

    last_response: LLMResponse | None = None
    for _ in range(max_iterations):
        response = await llm.generate(
            conversation,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tool_defs or None,
            tool_choice=tool_choice if tool_defs else None,
        )
        last_response = response

        if not response.tool_calls:
            return response

        conversation.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
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
            except Exception as exc:  # pragma: no cover - exercised in tests
                tool_content = f"Tool execution error: {exc}"

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
