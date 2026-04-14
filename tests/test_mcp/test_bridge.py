"""Tests for MCP bridge helpers and agentic loop."""

from __future__ import annotations

from typing import Any, AsyncIterator

from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    StreamChunk,
    ToolCall,
    TokenUsage,
)
from mai_gram.mcp_servers.bridge import (
    mcp_result_to_openai,
    mcp_tools_to_openai,
    openai_tool_call_to_mcp,
    run_with_tools,
)
from mai_gram.mcp_servers.manager import MCPManager, RegisteredTool
from mai_gram.mcp_servers.messages_server import MCPToolSpec


class _FakeServer:
    def __init__(self, tools: list[MCPToolSpec], response: str = "tool-ok", fail: bool = False) -> None:
        self._tools = tools
        self._response = response
        self._fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[MCPToolSpec]:
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((tool_name, arguments))
        if self._fail:
            raise RuntimeError("boom")
        return self._response


class _MockLLMProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[list[ChatMessage]] = []
        self.last_tools: list[Any] | None = None
        self.last_tool_choice: str | dict | None = None

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        return self._responses[len(self.calls) - 1]

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del messages, model, temperature, max_tokens, tools, tool_choice
        if False:  # pragma: no cover
            yield StreamChunk(content="")

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        del model
        return sum(len(message.content) for message in messages)

    async def close(self) -> None:
        return None


class TestBridgeHelpers:
    def test_mcp_tools_to_openai_format(self) -> None:
        tools = [
            RegisteredTool(
                server_name="messages",
                name="search_messages",
                description="Search",
                input_schema={"type": "object"},
            )
        ]

        converted = mcp_tools_to_openai(tools)

        assert converted == [
            {
                "type": "function",
                "function": {
                    "name": "search_messages",
                    "description": "Search",
                    "parameters": {"type": "object"},
                },
            }
        ]

    async def test_openai_tool_call_to_mcp_routing(self) -> None:
        manager = MCPManager()
        manager.register_server(
            "messages",
            _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})]),
        )
        tool_call = ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}')

        resolved = await openai_tool_call_to_mcp(tool_call, manager)

        assert resolved.server_name == "messages"
        assert resolved.tool_name == "search_messages"
        assert resolved.arguments == {"query": "Paris"}
        assert resolved.tool_call_id == "call_1"

    def test_mcp_result_to_openai_string(self) -> None:
        assert mcp_result_to_openai("hello") == "hello"
        assert mcp_result_to_openai({"ok": True}) == '{"ok": true}'


class TestRunWithTools:
    async def test_run_with_tools_no_tool_calls(self) -> None:
        llm = _MockLLMProvider(
            [LLMResponse(content="Final answer", model="mock", usage=TokenUsage(), finish_reason="stop")]
        )
        manager = MCPManager()
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

        result = await run_with_tools(llm, manager, messages)

        assert result.content == "Final answer"
        assert len(llm.calls) == 1

    async def test_run_with_tools_single_tool_call(self) -> None:
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}')
                    ],
                ),
                LLMResponse(content="Here is what I found.", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        server = _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})], "found")
        manager.register_server("messages", server)

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Remember Paris?")],
        )

        assert result.content == "Here is what I found."
        assert server.calls == [("search_messages", {"query": "Paris"})]
        assert len(llm.calls) == 2

    async def test_run_with_tools_multiple_tool_calls(self) -> None:
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}'),
                        ToolCall(id="call_2", name="search_messages", arguments='{"query":"Tokyo"}'),
                    ],
                ),
                LLMResponse(content="Combined results", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        server = _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})], "result")
        manager.register_server("messages", server)

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Search two cities")],
        )

        assert result.content == "Combined results"
        assert server.calls == [
            ("search_messages", {"query": "Paris"}),
            ("search_messages", {"query": "Tokyo"}),
        ]

    async def test_run_with_tools_max_iterations(self) -> None:
        tool_response = LLMResponse(
            content="",
            model="mock",
            tool_calls=[ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}')],
        )
        llm = _MockLLMProvider([tool_response, tool_response, tool_response])
        manager = MCPManager()
        manager.register_server(
            "messages",
            _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})], "result"),
        )

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Loop")],
            max_iterations=3,
        )

        assert result.finish_reason == "max_tool_iterations"
        assert len(llm.calls) == 3

    async def test_run_with_tools_intermediate_content_callback(self) -> None:
        """When the LLM produces text alongside a tool call, the
        on_intermediate_content callback should fire with that text."""
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="Hey!",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="sleep", arguments='{"duration":0}')
                    ],
                ),
                LLMResponse(content="What's up?", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        manager.register_server(
            "sleep",
            _FakeServer([MCPToolSpec("sleep", "Pause", {"type": "object"})], "ok"),
        )

        delivered: list[str] = []

        async def capture(text: str) -> None:
            delivered.append(text)

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Hi")],
            on_intermediate_content=capture,
        )

        assert delivered == ["Hey!"]
        assert result.content == "What's up?"

    async def test_run_with_tools_intermediate_content_not_fired_for_empty(self) -> None:
        """The callback should NOT fire when the assistant content is empty."""
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="sleep", arguments='{}')
                    ],
                ),
                LLMResponse(content="Done", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        manager.register_server(
            "sleep",
            _FakeServer([MCPToolSpec("sleep", "Pause", {"type": "object"})], "ok"),
        )

        delivered: list[str] = []

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Hi")],
            on_intermediate_content=lambda text: delivered.append(text),
        )

        assert delivered == []
        assert result.content == "Done"

    async def test_run_with_tools_multiple_intermediate_messages(self) -> None:
        """Multiple tool-call iterations should deliver multiple intermediate texts."""
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="First message",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="sleep", arguments='{"duration":0}')
                    ],
                ),
                LLMResponse(
                    content="Second message",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_2", name="sleep", arguments='{"duration":0}')
                    ],
                ),
                LLMResponse(content="Third message", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        manager.register_server(
            "sleep",
            _FakeServer([MCPToolSpec("sleep", "Pause", {"type": "object"})], "ok"),
        )

        delivered: list[str] = []

        async def capture(text: str) -> None:
            delivered.append(text)

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Tell me a lot")],
            on_intermediate_content=capture,
        )

        assert delivered == ["First message", "Second message"]
        assert result.content == "Third message"

    async def test_run_with_tools_tool_error_handling(self) -> None:
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}')
                    ],
                ),
                LLMResponse(content="Recovered after tool error", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        manager.register_server(
            "messages",
            _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})], fail=True),
        )

        result = await run_with_tools(
            llm,
            manager,
            [ChatMessage(role=MessageRole.USER, content="Try tool")],
        )

        assert result.content == "Recovered after tool error"
        second_call_messages = llm.calls[1]
        assert any(
            message.role == MessageRole.TOOL and "Tool execution error" in message.content
            for message in second_call_messages
        )
