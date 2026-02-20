"""MCP sleep server — creates pauses between messages.

When the AI companion wants to send multiple short messages (like a real
person texting), it calls the ``sleep`` tool between them.  Each tool call
creates a natural gap: the text produced *before* the call is dispatched
as a standalone message, the pause runs, and then the LLM continues with
the next piece of text.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mai_companion.mcp_servers.messages_server import MCPToolSpec

# Limits to prevent abuse or accidental long waits.
_MIN_SECONDS: float = 0.0
_MAX_SECONDS: float = 5.0
_DEFAULT_SECONDS: float = 1.0


class SleepMCPServer:
    """Expose a ``sleep`` tool for MCP-style tool calling.

    The tool itself is intentionally trivial — its purpose is to act as a
    *message separator*.  The agentic loop in ``run_with_tools`` delivers
    the assistant's intermediate text to the human whenever a tool call is
    made, so calling ``sleep`` between sentences produces the illusion of
    multiple individual messages.
    """

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return tools exposed by this server."""
        return [
            MCPToolSpec(
                name="sleep",
                description=(
                    "Pause before sending the next message. Use this when you want "
                    "to send several short messages instead of one long one — just "
                    "like a real person texting. Write your first message, call "
                    "sleep, then write the next message."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "number",
                            "minimum": _MIN_SECONDS,
                            "maximum": _MAX_SECONDS,
                            "default": _DEFAULT_SECONDS,
                            "description": (
                                f"Pause duration in seconds "
                                f"(default {_DEFAULT_SECONDS}, max {_MAX_SECONDS})."
                            ),
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            ),
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute the sleep tool."""
        if tool_name != "sleep":
            raise ValueError(f"Unknown tool: {tool_name}")

        raw_duration = arguments.get("duration", _DEFAULT_SECONDS)
        if not isinstance(raw_duration, (int, float)):
            raise ValueError("sleep 'duration' must be a number")

        duration = max(_MIN_SECONDS, min(float(raw_duration), _MAX_SECONDS))
        if duration > 0:
            await asyncio.sleep(duration)

        return "ok"
