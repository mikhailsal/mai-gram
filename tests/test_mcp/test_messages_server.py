"""Tests for MessagesMCPServer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def _create_companion(session: AsyncSession, chat_id: str = "test@testbot") -> str:
    companion = Chat(
        id=chat_id, user_id="test", bot_id="testbot", llm_model="test/model", system_prompt="test"
    )
    session.add(companion)
    await session.flush()
    return chat_id


class TestMessagesMCPServer:
    async def test_list_tools(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        tools = await server.list_tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "search_messages" in tool_names
        assert "get_message_context" in tool_names
        assert "get_messages_by_timerange" in tool_names

    async def test_search_messages_tool_schema(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        tools = await server.list_tools()
        schema = next(t for t in tools if t.name == "search_messages").input_schema
        properties = schema["properties"]
        assert properties["query"]["type"] == "string"
        assert properties["limit"]["type"] == "integer"
        assert properties["oldest_first"]["type"] == "boolean"
        assert "query" in schema["required"]

    async def test_call_search_messages(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        msg = await store.save_message(
            chat_id,
            "user",
            "I was in Paris",
            timestamp=datetime(2026, 2, 14, 9, 30, 0, tzinfo=timezone.utc),
        )
        server = MessagesMCPServer(store, chat_id)

        result = await server.call_tool("search_messages", {"query": "Paris"})

        # Should include message ID
        assert f"[#{msg.id}]" in result
        assert "[2026-02-14 09:30:00 UTC] user: I was in Paris" in result

    async def test_call_search_messages_empty(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        result = await server.call_tool("search_messages", {"query": "missing"})

        assert result == "No messages found."

    async def test_call_search_messages_limit_clamped_to_max(self, session: AsyncSession) -> None:
        """Verify that limit is clamped to max 50 even if a larger value is requested."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        # Create 60 messages
        for i in range(60):
            await store.save_message(
                chat_id,
                "user",
                f"keyword-{i}",
                timestamp=datetime(2026, 2, 14, 9, 0, i, tzinfo=timezone.utc),
            )
        server = MessagesMCPServer(store, chat_id)

        # Request limit=100, but should be clamped to 50
        result = await server.call_tool("search_messages", {"query": "keyword", "limit": 100})

        # Count the number of lines (one per message)
        lines = result.strip().split("\n")
        assert len(lines) == 50

    async def test_call_search_messages_oldest_first(self, session: AsyncSession) -> None:
        """Verify oldest_first parameter works in MCP tool."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 9, 0, 0, tzinfo=timezone.utc)
        await store.save_message(chat_id, "user", "keyword-old", timestamp=base)
        await store.save_message(
            chat_id, "user", "keyword-new", timestamp=base + timedelta(hours=1)
        )
        server = MessagesMCPServer(store, chat_id)

        # Default (newest first)
        result_newest = await server.call_tool("search_messages", {"query": "keyword"})
        lines_newest = result_newest.strip().split("\n")
        assert "keyword-new" in lines_newest[0]

        # Oldest first
        result_oldest = await server.call_tool(
            "search_messages", {"query": "keyword", "oldest_first": True}
        )
        lines_oldest = result_oldest.strip().split("\n")
        assert "keyword-old" in lines_oldest[0]


class TestGetMessageContext:
    async def test_get_message_context_basic(self, session: AsyncSession) -> None:
        """Verify get_message_context returns surrounding messages."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 9, 0, 0, tzinfo=timezone.utc)

        messages = []
        for i in range(7):
            msg = await store.save_message(
                chat_id,
                "user" if i % 2 == 0 else "assistant",
                f"msg-{i}",
                timestamp=base + timedelta(minutes=i),
            )
            messages.append(msg)

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_message_context",
            {"message_id": messages[3].id, "before": 2, "after": 2},
        )

        assert "--- Before ---" in result
        assert "--- Target message ---" in result
        assert "--- After ---" in result
        assert "msg-1" in result
        assert "msg-2" in result
        assert "msg-3" in result  # target
        assert "msg-4" in result
        assert "msg-5" in result

    async def test_get_message_context_not_found(self, session: AsyncSession) -> None:
        """Verify get_message_context handles missing message."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        result = await server.call_tool("get_message_context", {"message_id": 99999})

        assert "not found" in result.lower()

    async def test_get_message_context_limits_clamped(self, session: AsyncSession) -> None:
        """Verify before/after are clamped to max 10."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 9, 0, 0, tzinfo=timezone.utc)

        messages = []
        for i in range(25):
            msg = await store.save_message(
                chat_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )
            messages.append(msg)

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_message_context",
            {"message_id": messages[12].id, "before": 100, "after": 100},
        )

        # Count messages in result (excluding section headers)
        lines = [line for line in result.split("\n") if line.startswith("[#")]
        # Should be 10 before + 1 target + 10 after = 21 max
        assert len(lines) <= 21

    async def test_get_message_context_invalid_message_id(self, session: AsyncSession) -> None:
        """Verify get_message_context requires integer message_id."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="integer"):
            await server.call_tool("get_message_context", {"message_id": "not-an-int"})


class TestGetMessagesByTimerange:
    async def test_get_messages_by_timerange_basic(self, session: AsyncSession) -> None:
        """Without end_date, returns all messages from start_date onward."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(
            chat_id,
            "user",
            "msg-day1",
            timestamp=datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc),
        )
        await store.save_message(
            chat_id,
            "user",
            "msg-day2",
            timestamp=datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc),
        )

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool("get_messages_by_timerange", {"start_date": "2026-02-14"})

        assert "msg-day1" in result
        assert "msg-day2" in result
        assert "Showing messages 1-2 of 2 total" in result

    async def test_get_messages_by_timerange_single_day(self, session: AsyncSession) -> None:
        """With end_date == start_date, returns only that day's messages."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(
            chat_id,
            "user",
            "msg-day1",
            timestamp=datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc),
        )
        await store.save_message(
            chat_id,
            "user",
            "msg-day2",
            timestamp=datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc),
        )

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "end_date": "2026-02-14"},
        )

        assert "msg-day1" in result
        assert "msg-day2" not in result
        assert "Showing messages 1-1 of 1 total" in result

    async def test_get_messages_by_timerange_with_end_date(self, session: AsyncSession) -> None:
        """Verify get_messages_by_timerange respects end_date."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        for i in range(5):
            await store.save_message(
                chat_id,
                "user",
                f"day{i + 10}",
                timestamp=datetime(2026, 2, 10 + i, 10, 0, tzinfo=timezone.utc),
            )

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-11", "end_date": "2026-02-13"},
        )

        assert "day11" in result
        assert "day12" in result
        assert "day13" in result
        assert "day10" not in result
        assert "day14" not in result
        assert "of 3 total" in result

    async def test_get_messages_by_timerange_pagination(self, session: AsyncSession) -> None:
        """Verify pagination works correctly."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(15):
            await store.save_message(
                chat_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        server = MessagesMCPServer(store, chat_id)

        # First page
        result1 = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "limit": 5, "offset": 0},
        )
        assert "msg-0" in result1
        assert "msg-4" in result1
        assert "msg-5" not in result1
        assert "Showing messages 1-5 of 15 total" in result1
        assert "use offset=5 to see more" in result1

        # Second page
        result2 = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "limit": 5, "offset": 5},
        )
        assert "msg-5" in result2
        assert "msg-9" in result2
        assert "Showing messages 6-10 of 15 total" in result2

    async def test_get_messages_by_timerange_newest_first(self, session: AsyncSession) -> None:
        """Verify oldest_first=false returns newest messages first."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            await store.save_message(
                chat_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "oldest_first": False},
        )

        lines = result.split("\n")
        # Find first message line (skip header)
        msg_lines = [line for line in lines if "msg-" in line]
        assert "msg-4" in msg_lines[0]  # Newest first

    async def test_get_messages_by_timerange_empty(self, session: AsyncSession) -> None:
        """Verify empty result message when no end_date is given."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        result = await server.call_tool("get_messages_by_timerange", {"start_date": "2026-02-14"})

        assert "No messages found from 2026-02-14 onward" in result

    async def test_get_messages_by_timerange_invalid_date(self, session: AsyncSession) -> None:
        """Verify invalid date format raises error."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="Invalid start_date format"):
            await server.call_tool("get_messages_by_timerange", {"start_date": "not-a-date"})

    async def test_get_messages_by_timerange_limit_clamped(self, session: AsyncSession) -> None:
        """Verify limit is clamped to max 20."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(30):
            await store.save_message(
                chat_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        server = MessagesMCPServer(store, chat_id)
        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "limit": 100},
        )

        # Count message lines
        msg_lines = [line for line in result.split("\n") if line.startswith("[#")]
        assert len(msg_lines) == 20  # Clamped to max

    async def test_get_messages_by_timerange_empty_with_end_date(
        self, session: AsyncSession
    ) -> None:
        """Verify empty result message when end_date is different from start_date."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "end_date": "2026-02-16"},
        )
        assert "No messages found from 2026-02-14 to 2026-02-16" in result

    async def test_get_messages_by_timerange_empty_same_date(self, session: AsyncSession) -> None:
        """Verify empty result message when end_date == start_date."""
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        result = await server.call_tool(
            "get_messages_by_timerange",
            {"start_date": "2026-02-14", "end_date": "2026-02-14"},
        )
        assert "No messages found for 2026-02-14" in result


class TestMessagesServerErrorHandling:
    async def test_unknown_tool_raises(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="Unknown tool"):
            await server.call_tool("nonexistent_tool", {})

    async def test_search_messages_empty_query(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="non-empty string"):
            await server.call_tool("search_messages", {"query": ""})

    async def test_search_messages_non_string_query(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="non-empty string"):
            await server.call_tool("search_messages", {"query": 123})

    async def test_search_messages_non_int_limit(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="'limit' must be an integer"):
            await server.call_tool("search_messages", {"query": "test", "limit": "ten"})

    async def test_search_messages_non_bool_oldest_first(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="'oldest_first' must be a boolean"):
            await server.call_tool("search_messages", {"query": "test", "oldest_first": "yes"})

    async def test_get_context_non_int_before_after(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="must be integers"):
            await server.call_tool("get_message_context", {"message_id": 1, "before": "abc"})

    async def test_timerange_non_string_start_date(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="requires"):
            await server.call_tool("get_messages_by_timerange", {"start_date": 123})

    async def test_timerange_non_string_end_date(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="must be a string"):
            await server.call_tool(
                "get_messages_by_timerange",
                {"start_date": "2026-02-14", "end_date": 123},
            )

    async def test_timerange_invalid_end_date(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="Invalid end_date"):
            await server.call_tool(
                "get_messages_by_timerange",
                {"start_date": "2026-02-14", "end_date": "not-a-date"},
            )

    async def test_parse_bounded_int_non_integer(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="must be an integer"):
            await server.call_tool(
                "get_messages_by_timerange",
                {"start_date": "2026-02-14", "limit": "abc"},
            )

    async def test_parse_boolean_non_bool(self, session: AsyncSession) -> None:
        chat_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), chat_id)

        with pytest.raises(ValueError, match="must be a boolean"):
            await server.call_tool(
                "get_messages_by_timerange",
                {"start_date": "2026-02-14", "oldest_first": "yes"},
            )


class TestFormatMessageWithId:
    async def test_format_invalid_timezone_fallback(self, session: AsyncSession) -> None:
        """Messages with invalid timezone should fallback to UTC."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        msg = await store.save_message(
            chat_id,
            "user",
            "test message",
            timestamp=datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc),
        )
        msg.timezone = "Invalid/Timezone"

        result = MessagesMCPServer._format_message_with_id(msg)
        assert "UTC" in result
        assert "test message" in result

    async def test_format_message_show_datetime_false(self, session: AsyncSession) -> None:
        """Messages with show_datetime=False use 'imported' text."""
        chat_id = await _create_companion(session)
        store = MessageStore(session)
        msg = await store.save_message(
            chat_id,
            "user",
            "imported msg",
            timestamp=datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc),
        )
        msg.show_datetime = False

        result = MessagesMCPServer._format_message_with_id(msg)
        assert "imported, real date unknown" in result
        assert "imported msg" in result
