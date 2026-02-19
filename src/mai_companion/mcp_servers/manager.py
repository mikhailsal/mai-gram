"""MCP server registry and tool routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mai_companion.mcp_servers.messages_server import MCPToolSpec


class MCPServer(Protocol):
    """Protocol implemented by in-process MCP servers."""

    async def list_tools(self) -> list[MCPToolSpec]:
        """List supported tools."""

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute one tool call."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Tool info with server origin."""

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPManager:
    """Registers MCP servers and routes tool calls."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}

    def register_server(self, server_name: str, server: MCPServer) -> None:
        """Register a named MCP server."""
        if not server_name.strip():
            raise ValueError("server_name must not be empty")
        if server_name in self._servers:
            raise ValueError(f"Server '{server_name}' is already registered")
        self._servers[server_name] = server

    async def list_all_tools(self) -> list[RegisteredTool]:
        """Return tools from all registered servers."""
        all_tools: list[RegisteredTool] = []
        for server_name, server in self._servers.items():
            for tool in await server.list_tools():
                all_tools.append(
                    RegisteredTool(
                        server_name=server_name,
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.input_schema,
                    )
                )
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Route a tool call to the target server."""
        server = self._servers.get(server_name)
        if server is None:
            raise ValueError(f"Unknown MCP server: {server_name}")

        available_names = {tool.name for tool in await server.list_tools()}
        if tool_name not in available_names:
            raise ValueError(f"Unknown tool '{tool_name}' on server '{server_name}'")

        return await server.call_tool(tool_name, arguments)

    async def resolve_tool_server(self, tool_name: str) -> str:
        """Return the unique server name that owns tool_name."""
        matches: list[str] = []
        for server_name, server in self._servers.items():
            names = {tool.name for tool in await server.list_tools()}
            if tool_name in names:
                matches.append(server_name)

        if not matches:
            raise ValueError(f"No MCP server exposes tool '{tool_name}'")
        if len(matches) > 1:
            names = ", ".join(sorted(matches))
            raise ValueError(f"Tool '{tool_name}' is ambiguous across servers: {names}")
        return matches[0]
