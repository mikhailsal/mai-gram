"""MCP messages server backed by MessageStore."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mai_companion.memory.messages import MessageStore


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """Minimal MCP-compatible tool metadata."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MessagesMCPServer:
    """Expose message history tools for MCP-style tool calling."""

    def __init__(self, store: MessageStore, companion_id: str) -> None:
        self._store = store
        self._companion_id = companion_id

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return tools exposed by this server."""
        return [
            MCPToolSpec(
                name="search_messages",
                description=(
                    "Search your conversation history by keywords. Use this when you need "
                    "to recall what was said about a specific topic. Works like searching "
                    "in a messenger app."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a named tool with JSON-like arguments."""
        if tool_name != "search_messages":
            raise ValueError(f"Unknown tool: {tool_name}")

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_messages requires non-empty string 'query'")

        raw_limit = arguments.get("limit", 20)
        if not isinstance(raw_limit, int):
            raise ValueError("search_messages 'limit' must be an integer")
        limit = max(1, min(raw_limit, 50))  # Clamp to [1, 50]

        results = await self._store.search(self._companion_id, query.strip(), limit=limit)
        if not results:
            return "No messages found."

        lines = [self._format_message(row.timestamp, row.role, row.content) for row in results]
        return "\n".join(lines)

    @staticmethod
    def _format_message(timestamp: datetime, role: str, content: str) -> str:
        ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] {role}: {content}"
