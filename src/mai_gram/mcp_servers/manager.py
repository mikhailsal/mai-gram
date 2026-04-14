"""MCP server registry and tool routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mai_gram.mcp_servers.messages_server import MCPToolSpec


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

    def __init__(
        self,
        *,
        enabled_tools: list[str] | None = None,
        disabled_tools: list[str] | None = None,
    ) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._enabled_tools = set(enabled_tools) if enabled_tools else None
        self._disabled_tools = set(disabled_tools) if disabled_tools else None

    def register_server(self, server_name: str, server: MCPServer) -> None:
        """Register a named MCP server."""
        if not server_name.strip():
            raise ValueError("server_name must not be empty")
        if server_name in self._servers:
            raise ValueError(f"Server '{server_name}' is already registered")
        self._servers[server_name] = server

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed by the enable/disable filter."""
        if self._enabled_tools is not None:
            return tool_name in self._enabled_tools
        if self._disabled_tools is not None:
            return tool_name not in self._disabled_tools
        return True

    async def list_all_tools(self) -> list[RegisteredTool]:
        """Return tools from all registered servers, filtered by enable/disable config."""
        all_tools: list[RegisteredTool] = []
        for server_name, server in self._servers.items():
            for tool in await server.list_tools():
                if self._is_tool_allowed(tool.name):
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
