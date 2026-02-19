"""Tests for MCPManager."""

from __future__ import annotations

from typing import Any

import pytest

from mai_companion.mcp_servers.manager import MCPManager
from mai_companion.mcp_servers.messages_server import MCPToolSpec


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
        manager.register_server("messages", _FakeServer([MCPToolSpec("search_messages", "Search", {})]))

        with pytest.raises(ValueError, match="Unknown tool"):
            await manager.call_tool("messages", "missing_tool", {})
