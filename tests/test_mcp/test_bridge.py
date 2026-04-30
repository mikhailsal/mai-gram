"""Tests for MCP bridge helpers and agentic loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from mai_gram.llm.provider import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from mai_gram.mcp_servers.bridge import (
    mcp_result_to_openai,
    mcp_tools_to_openai,
    openai_tool_call_to_mcp,
    run_with_tools,
    run_with_tools_stream,
)
from mai_gram.mcp_servers.manager import MCPManager, RegisteredTool
from mai_gram.mcp_servers.messages_server import MCPToolSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FakeServer:
    def __init__(
        self, tools: list[MCPToolSpec], response: str = "tool-ok", fail: bool = False
    ) -> None:
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
    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        stream_sequences: list[list[StreamChunk]] | None = None,
    ) -> None:
        self._responses = responses
        self._stream_sequences = stream_sequences or []
        self.calls: list[list[ChatMessage]] = []
        self.stream_calls: list[list[ChatMessage]] = []
        self.last_tools: list[Any] | None = None
        self.last_tool_choice: str | dict[str, Any] | None = None

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
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        del model, temperature, max_tokens, tools, tool_choice, extra_params
        self.stream_calls.append(list(messages))
        for chunk in self._stream_sequences[len(self.stream_calls) - 1]:
            yield chunk

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
    async def test_run_with_tools_rejects_non_positive_max_iterations(self) -> None:
        llm = _MockLLMProvider([])
        manager = MCPManager()

        with pytest.raises(ValueError, match="max_iterations"):
            await run_with_tools(
                llm,
                manager,
                [ChatMessage(role=MessageRole.USER, content="Hello")],
                max_iterations=0,
            )

    async def test_run_with_tools_no_tool_calls(self) -> None:
        llm = _MockLLMProvider(
            [
                LLMResponse(
                    content="Final answer", model="mock", usage=TokenUsage(), finish_reason="stop"
                )
            ]
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
        server = _FakeServer(
            [MCPToolSpec("search_messages", "Search", {"type": "object"})], "found"
        )
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
                        ToolCall(
                            id="call_1", name="search_messages", arguments='{"query":"Paris"}'
                        ),
                        ToolCall(
                            id="call_2", name="search_messages", arguments='{"query":"Tokyo"}'
                        ),
                    ],
                ),
                LLMResponse(content="Combined results", model="mock", finish_reason="stop"),
            ]
        )
        manager = MCPManager()
        server = _FakeServer(
            [MCPToolSpec("search_messages", "Search", {"type": "object"})], "result"
        )
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
            tool_calls=[
                ToolCall(id="call_1", name="search_messages", arguments='{"query":"Paris"}')
            ],
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
                    tool_calls=[ToolCall(id="call_1", name="sleep", arguments='{"duration":0}')],
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
                    tool_calls=[ToolCall(id="call_1", name="sleep", arguments="{}")],
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
                    tool_calls=[ToolCall(id="call_1", name="sleep", arguments='{"duration":0}')],
                ),
                LLMResponse(
                    content="Second message",
                    model="mock",
                    tool_calls=[ToolCall(id="call_2", name="sleep", arguments='{"duration":0}')],
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
                LLMResponse(
                    content="Recovered after tool error", model="mock", finish_reason="stop"
                ),
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


class TestRunWithToolsStream:
    async def test_run_with_tools_stream_rejects_non_positive_max_iterations(self) -> None:
        llm = _MockLLMProvider([])
        manager = MCPManager()

        with pytest.raises(ValueError, match="max_iterations"):
            [
                chunk
                async for chunk in run_with_tools_stream(
                    llm,
                    manager,
                    [ChatMessage(role=MessageRole.USER, content="Hi")],
                    max_iterations=0,
                )
            ]

    async def test_run_with_tools_stream_yields_final_usage_without_tools(self) -> None:
        llm = _MockLLMProvider(
            [],
            stream_sequences=[
                [
                    StreamChunk(content="Hello ", reasoning="think", usage=None),
                    StreamChunk(
                        content="world",
                        finish_reason="stop",
                        usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
                        cost=0.25,
                        is_byok=True,
                    ),
                ]
            ],
        )
        manager = MCPManager()

        chunks = [
            chunk
            async for chunk in run_with_tools_stream(
                llm,
                manager,
                [ChatMessage(role=MessageRole.USER, content="Hi")],
            )
        ]

        assert [chunk.content for chunk in chunks[:-1]] == ["Hello ", "world"]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 5
        assert chunks[-1].cost == 0.25
        assert chunks[-1].is_byok is True

    async def test_run_with_tools_stream_executes_tool_loop_and_callbacks(self) -> None:
        llm = _MockLLMProvider(
            [],
            stream_sequences=[
                [
                    StreamChunk(content="Working", usage=None),
                    StreamChunk(
                        content="",
                        tool_calls_delta=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "sleep", "arguments": '{"duration":'},
                            }
                        ],
                    ),
                    StreamChunk(
                        content="",
                        tool_calls_delta=[{"index": 0, "function": {"arguments": "0}"}}],
                        finish_reason="tool_calls",
                        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                    ),
                ],
                [
                    StreamChunk(
                        content="Done",
                        finish_reason="stop",
                        usage=TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                    )
                ],
            ],
        )
        manager = MCPManager()
        server = _FakeServer([MCPToolSpec("sleep", "Pause", {"type": "object"})], "ok")
        manager.register_server("sleep", server)

        delivered: list[str] = []
        assistant_tool_calls: list[tuple[str, list[ToolCall]]] = []
        tool_results: list[str] = []

        async def on_intermediate_content(text: str) -> None:
            delivered.append(text)

        async def on_assistant_tool_call(*, content: str, tool_calls: list[ToolCall]) -> None:
            assistant_tool_calls.append((content, tool_calls))

        async def on_tool_result(**kwargs: Any) -> None:
            tool_results.append(kwargs["content"])

        chunks = [
            chunk
            async for chunk in run_with_tools_stream(
                llm,
                manager,
                [ChatMessage(role=MessageRole.USER, content="Hi")],
                on_intermediate_content=on_intermediate_content,
                on_assistant_tool_call=on_assistant_tool_call,
                on_tool_result=on_tool_result,
            )
        ]

        assert server.calls == [("sleep", {"duration": 0})]
        assert delivered == ["Working"]
        assert assistant_tool_calls == [
            (
                "Working",
                [ToolCall(id="call_1", name="sleep", arguments='{"duration":0}')],
            )
        ]
        assert tool_results == ["ok"]
        assert any(chunk.turn_complete for chunk in chunks)
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 10

    async def test_run_with_tools_stream_emits_max_iteration_final_chunk(self) -> None:
        llm = _MockLLMProvider(
            [],
            stream_sequences=[
                [
                    StreamChunk(content="Working", usage=None),
                    StreamChunk(
                        content="",
                        tool_calls_delta=[
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "sleep", "arguments": '{"duration":'},
                            }
                        ],
                    ),
                    StreamChunk(
                        content="",
                        tool_calls_delta=[{"index": 0, "function": {"arguments": "0}"}}],
                        finish_reason="tool_calls",
                        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                    ),
                ],
                [
                    StreamChunk(
                        content="Fallback answer",
                        finish_reason="stop",
                        usage=TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                    )
                ],
            ],
        )
        manager = MCPManager()
        manager.register_server(
            "sleep",
            _FakeServer([MCPToolSpec("sleep", "Pause", {"type": "object"})], "ok"),
        )

        chunks = [
            chunk
            async for chunk in run_with_tools_stream(
                llm,
                manager,
                [ChatMessage(role=MessageRole.USER, content="Hi")],
                max_iterations=1,
            )
        ]

        assert any(chunk.turn_complete for chunk in chunks)
        assert any(chunk.content == "Fallback answer" for chunk in chunks)
        assert chunks[-1].finish_reason == "max_tool_iterations"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 10
