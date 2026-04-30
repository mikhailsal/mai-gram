"""MCP messages server backed by MessageStore."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from mai_gram.db.models import Message
    from mai_gram.memory.messages import MessageStore


@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    """Minimal MCP-compatible tool metadata."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MessageTimeRangeQuery:
    """Parsed arguments for the timerange messages tool."""

    start_date_str: str
    start_date: date
    end_date_str: str | None
    end_date: date | None
    limit: int
    offset: int
    oldest_first: bool


_MESSAGE_TOOL_SPECS = (
    MCPToolSpec(
        name="search_messages",
        description=(
            "Search your conversation history by keywords. Use this when you need "
            "to recall what was said about a specific topic. Works like searching "
            "in a messenger app. Returns message IDs that you can use with "
            "get_message_context for more detail."
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
                "oldest_first": {
                    "type": "boolean",
                    "description": (
                        "If true, return oldest matches first (useful for finding"
                        " when something was first mentioned)."
                        " Default: false (newest first)."
                    ),
                    "default": False,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    MCPToolSpec(
        name="get_message_context",
        description=(
            "Get detailed context around a specific message. Use this after "
            "search_messages when you want to see what was said before and after "
            "a particular message. Provide the message ID from search results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "integer",
                    "description": "The message ID to get context for (from search results)",
                },
                "before": {
                    "type": "integer",
                    "description": "Number of messages to show before the target",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 5,
                },
                "after": {
                    "type": "integer",
                    "description": "Number of messages to show after the target",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
    ),
    MCPToolSpec(
        name="get_messages_by_timerange",
        description=(
            "Get messages from a specific time period. When only start_date "
            "is given, returns messages from that date onward (no upper bound). "
            "Supports pagination to avoid overloading context — call multiple "
            "times with increasing offset to see more messages."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": (
                        "Start date in YYYY-MM-DD format (inclusive). "
                        "When used alone, returns all messages from this date onward."
                    ),
                },
                "end_date": {
                    "type": "string",
                    "description": (
                        "End date in YYYY-MM-DD format (inclusive). "
                        "If not provided, no upper date limit is applied."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default 10, max 20)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many messages (for pagination)",
                    "minimum": 0,
                    "default": 0,
                },
                "oldest_first": {
                    "type": "boolean",
                    "description": "If true (default), show oldest messages first",
                    "default": True,
                },
            },
            "required": ["start_date"],
            "additionalProperties": False,
        },
    ),
)


def _parse_bounded_int(
    arguments: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw_value = arguments.get(key, default)
    if not isinstance(raw_value, int):
        raise ValueError(f"'{key}' must be an integer")

    parsed = max(minimum, raw_value)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _parse_boolean(arguments: dict[str, Any], key: str, *, default: bool) -> bool:
    value = arguments.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean")
    return value


def _parse_iso_date(value: Any, *, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"get_messages_by_timerange requires '{field_name}' string")

    try:
        return date.fromisoformat(value)
    except ValueError as err:
        raise ValueError(f"Invalid {field_name} format: {value}. Use YYYY-MM-DD.") from err


def _parse_optional_iso_date(value: Any) -> tuple[str | None, date | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise ValueError("'end_date' must be a string")
    try:
        return value, date.fromisoformat(value)
    except ValueError as err:
        raise ValueError(f"Invalid end_date format: {value}. Use YYYY-MM-DD.") from err


def _build_time_range_query(arguments: dict[str, Any]) -> MessageTimeRangeQuery:
    raw_start_date = arguments.get("start_date")
    start_date = _parse_iso_date(raw_start_date, field_name="start_date")
    start_date_str = cast("str", raw_start_date)
    end_date_str, end_date = _parse_optional_iso_date(arguments.get("end_date"))
    return MessageTimeRangeQuery(
        start_date_str=start_date_str,
        start_date=start_date,
        end_date_str=end_date_str,
        end_date=end_date,
        limit=_parse_bounded_int(arguments, "limit", default=10, minimum=1, maximum=20),
        offset=_parse_bounded_int(arguments, "offset", default=0, minimum=0),
        oldest_first=_parse_boolean(arguments, "oldest_first", default=True),
    )


def _format_empty_time_range_result(query: MessageTimeRangeQuery) -> str:
    if query.end_date_str and query.end_date_str != query.start_date_str:
        return f"No messages found from {query.start_date_str} to {query.end_date_str}."
    if query.end_date_str is None:
        return f"No messages found from {query.start_date_str} onward."
    return f"No messages found for {query.start_date_str}."


def _build_timerange_header(*, offset: int, result_count: int, total_count: int) -> str:
    showing_end = min(offset + result_count, total_count)
    header = f"Showing messages {offset + 1}-{showing_end} of {total_count} total"
    if showing_end < total_count:
        header += f" (use offset={showing_end} to see more)"
    return header


class MessagesMCPServer:
    """Expose message history tools for MCP-style tool calling."""

    def __init__(self, store: MessageStore, chat_id: str) -> None:
        self._store = store
        self._chat_id = chat_id

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return tools exposed by this server."""
        return list(_MESSAGE_TOOL_SPECS)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a named tool with JSON-like arguments."""
        if tool_name == "search_messages":
            return await self._call_search_messages(arguments)
        elif tool_name == "get_message_context":
            return await self._call_get_message_context(arguments)
        elif tool_name == "get_messages_by_timerange":
            return await self._call_get_messages_by_timerange(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _call_search_messages(self, arguments: dict[str, Any]) -> str:
        """Handle search_messages tool call."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_messages requires non-empty string 'query'")

        raw_limit = arguments.get("limit", 20)
        if not isinstance(raw_limit, int):
            raise ValueError("search_messages 'limit' must be an integer")
        limit = max(1, min(raw_limit, 50))

        oldest_first = arguments.get("oldest_first", False)
        if not isinstance(oldest_first, bool):
            raise ValueError("search_messages 'oldest_first' must be a boolean")

        results = await self._store.search(
            self._chat_id,
            query.strip(),
            limit=limit,
            oldest_first=oldest_first,
        )
        if not results:
            return "No messages found."

        lines = [self._format_message_with_id(row) for row in results]
        return "\n".join(lines)

    async def _call_get_message_context(self, arguments: dict[str, Any]) -> str:
        """Handle get_message_context tool call."""
        message_id = arguments.get("message_id")
        if not isinstance(message_id, int):
            raise ValueError("get_message_context requires integer 'message_id'")

        raw_before = arguments.get("before", 5)
        raw_after = arguments.get("after", 5)
        if not isinstance(raw_before, int) or not isinstance(raw_after, int):
            raise ValueError("'before' and 'after' must be integers")

        before = max(0, min(raw_before, 10))
        after = max(0, min(raw_after, 10))

        messages_before, target, messages_after = await self._store.get_message_context(
            self._chat_id,
            message_id,
            before=before,
            after=after,
        )

        if target is None:
            return f"Message #{message_id} not found."

        lines: list[str] = []

        if messages_before:
            lines.append("--- Before ---")
            for msg in messages_before:
                lines.append(self._format_message_with_id(msg))

        lines.append("--- Target message ---")
        lines.append(self._format_message_with_id(target))

        if messages_after:
            lines.append("--- After ---")
            for msg in messages_after:
                lines.append(self._format_message_with_id(msg))

        return "\n".join(lines)

    async def _call_get_messages_by_timerange(self, arguments: dict[str, Any]) -> str:
        """Handle get_messages_by_timerange tool call."""
        query = _build_time_range_query(arguments)

        messages, total_count = await self._store.get_messages_paginated(
            self._chat_id,
            limit=query.limit,
            offset=query.offset,
            oldest_first=query.oldest_first,
            start_date=query.start_date,
            end_date=query.end_date,
        )

        if not messages:
            return _format_empty_time_range_result(query)

        lines = [
            _build_timerange_header(
                offset=query.offset,
                result_count=len(messages),
                total_count=total_count,
            ),
            "",
        ]
        for msg in messages:
            lines.append(self._format_message_with_id(msg))

        return "\n".join(lines)

    @staticmethod
    def _format_message_with_id(msg: Message) -> str:
        """Format a message with ID, timestamp in its stored timezone, role, and content."""
        show_dt = getattr(msg, "show_datetime", True)
        if show_dt is False:
            ts_str = "imported, real date unknown"
        else:
            tz_name = getattr(msg, "timezone", "UTC") or "UTC"
            try:
                tz: ZoneInfo | timezone = ZoneInfo(tz_name)
            except (KeyError, ValueError):
                tz = timezone.utc
                tz_name = "UTC"
            ts = msg.timestamp.replace(tzinfo=timezone.utc).astimezone(tz)
            ts_str = f"{ts.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}"
        return f"[#{msg.id}] [{ts_str}] {msg.role}: {msg.content}"

    @staticmethod
    def _format_message(timestamp: datetime, role: str, content: str) -> str:
        """Format a message without ID (legacy method)."""
        ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts} UTC] {role}: {content}"
