"""Tests for MessagesMCPServer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import Companion
from mai_companion.memory.messages import MessageStore
from mai_companion.mcp_servers.messages_server import MessagesMCPServer


async def _create_companion(session: AsyncSession, companion_id: str = "comp-mcp-msg") -> str:
    companion = Companion(id=companion_id, name="MCP Messages")
    session.add(companion)
    await session.flush()
    return companion_id


class TestMessagesMCPServer:
    async def test_list_tools(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), companion_id)

        tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_messages"

    async def test_search_messages_tool_schema(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), companion_id)

        schema = (await server.list_tools())[0].input_schema
        properties = schema["properties"]
        assert properties["query"]["type"] == "string"
        assert properties["limit"]["type"] == "integer"
        assert "query" in schema["required"]

    async def test_call_search_messages(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(
            companion_id,
            "user",
            "I was in Paris",
            timestamp=datetime(2026, 2, 14, 9, 30, 0),
        )
        server = MessagesMCPServer(store, companion_id)

        result = await server.call_tool("search_messages", {"query": "Paris"})

        assert "[2026-02-14 09:30:00] user: I was in Paris" in result

    async def test_call_search_messages_empty(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        server = MessagesMCPServer(MessageStore(session), companion_id)

        result = await server.call_tool("search_messages", {"query": "missing"})

        assert result == "No messages found."

    async def test_call_search_messages_limit_clamped_to_max(self, session: AsyncSession) -> None:
        """Verify that limit is clamped to max 50 even if a larger value is requested."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        # Create 60 messages
        for i in range(60):
            await store.save_message(
                companion_id,
                "user",
                f"keyword-{i}",
                timestamp=datetime(2026, 2, 14, 9, 0, i),
            )
        server = MessagesMCPServer(store, companion_id)

        # Request limit=100, but should be clamped to 50
        result = await server.call_tool("search_messages", {"query": "keyword", "limit": 100})

        # Count the number of lines (one per message)
        lines = result.strip().split("\n")
        assert len(lines) == 50
