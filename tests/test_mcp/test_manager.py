"""Tests for MCPManager."""

from __future__ import annotations

from typing import Any

import pytest

from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MCPToolSpec


class _FakeServer:
    def __init__(self, tools: list[MCPToolSpec], response: str = "ok") -> None:
        self._tools = tools
        self._response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[MCPToolSpec]:
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((tool_name, arguments))
        return self._response


class TestMCPManager:
    async def test_register_and_list_tools(self) -> None:
        manager = MCPManager()
        manager.register_server(
            "messages",
            _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})]),
        )
        manager.register_server(
            "wiki",
            _FakeServer([MCPToolSpec("wiki_read", "Read", {"type": "object"})]),
        )

        tools = await manager.list_all_tools()

        assert [(tool.server_name, tool.name) for tool in tools] == [
            ("messages", "search_messages"),
            ("wiki", "wiki_read"),
        ]

    async def test_call_tool_routes_correctly(self) -> None:
        manager = MCPManager()
        server = _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})], "done")
        manager.register_server("messages", server)

        result = await manager.call_tool("messages", "search_messages", {"query": "Paris"})

        assert result == "done"
        assert server.calls == [("search_messages", {"query": "Paris"})]

    async def test_call_tool_unknown_server(self) -> None:
        manager = MCPManager()

        with pytest.raises(ValueError, match="Unknown MCP server"):
            await manager.call_tool("missing", "search_messages", {"query": "x"})

    async def test_call_tool_unknown_tool(self) -> None:
        manager = MCPManager()
        manager.register_server(
            "messages", _FakeServer([MCPToolSpec("search_messages", "Search", {})])
        )

        with pytest.raises(ValueError, match="Unknown tool"):
            await manager.call_tool("messages", "missing_tool", {})

    async def test_register_server_empty_name(self) -> None:
        manager = MCPManager()
        with pytest.raises(ValueError, match="must not be empty"):
            manager.register_server("", _FakeServer([]))

    async def test_register_server_duplicate_name(self) -> None:
        manager = MCPManager()
        manager.register_server("wiki", _FakeServer([]))
        with pytest.raises(ValueError, match="already registered"):
            manager.register_server("wiki", _FakeServer([]))

    async def test_enabled_tools_filter(self) -> None:
        manager = MCPManager(enabled_tools=["wiki_read"])
        manager.register_server(
            "wiki",
            _FakeServer(
                [
                    MCPToolSpec("wiki_read", "Read wiki", {"type": "object"}),
                    MCPToolSpec("wiki_write", "Write wiki", {"type": "object"}),
                ]
            ),
        )
        tools = await manager.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "wiki_read"

    async def test_disabled_tools_filter(self) -> None:
        manager = MCPManager(disabled_tools=["wiki_write"])
        manager.register_server(
            "wiki",
            _FakeServer(
                [
                    MCPToolSpec("wiki_read", "Read wiki", {"type": "object"}),
                    MCPToolSpec("wiki_write", "Write wiki", {"type": "object"}),
                ]
            ),
        )
        tools = await manager.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "wiki_read"

    async def test_resolve_tool_server_success(self) -> None:
        manager = MCPManager()
        manager.register_server(
            "messages",
            _FakeServer([MCPToolSpec("search_messages", "Search", {"type": "object"})]),
        )
        manager.register_server(
            "wiki",
            _FakeServer([MCPToolSpec("wiki_read", "Read", {"type": "object"})]),
        )

        result = await manager.resolve_tool_server("wiki_read")
        assert result == "wiki"

    async def test_resolve_tool_server_not_found(self) -> None:
        manager = MCPManager()
        manager.register_server("wiki", _FakeServer([]))
        with pytest.raises(ValueError, match="No MCP server exposes"):
            await manager.resolve_tool_server("nonexistent_tool")

    async def test_resolve_tool_server_ambiguous(self) -> None:
        manager = MCPManager()
        manager.register_server(
            "server_a",
            _FakeServer([MCPToolSpec("shared_tool", "Tool A", {"type": "object"})]),
        )
        manager.register_server(
            "server_b",
            _FakeServer([MCPToolSpec("shared_tool", "Tool B", {"type": "object"})]),
        )
        with pytest.raises(ValueError, match="ambiguous"):
            await manager.resolve_tool_server("shared_tool")
