"""Tests for SleepMCPServer."""

from __future__ import annotations

import asyncio
import time

import pytest

from mai_companion.mcp_servers.sleep_server import SleepMCPServer


class TestSleepMCPServer:
    async def test_list_tools(self) -> None:
        server = SleepMCPServer()

        tools = await server.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "sleep"

    async def test_sleep_tool_schema(self) -> None:
        server = SleepMCPServer()

        schema = (await server.list_tools())[0].input_schema
        properties = schema["properties"]
        assert properties["duration"]["type"] == "number"
        assert properties["duration"]["minimum"] == 0.0
        assert properties["duration"]["maximum"] == 5.0

    async def test_call_sleep_default_duration(self) -> None:
        server = SleepMCPServer()

        start = time.monotonic()
        result = await server.call_tool("sleep", {})
        elapsed = time.monotonic() - start

        assert result == "ok"
        # Default is 1 second; allow some tolerance
        assert elapsed >= 0.9

    async def test_call_sleep_custom_duration(self) -> None:
        server = SleepMCPServer()

        start = time.monotonic()
        result = await server.call_tool("sleep", {"duration": 0.1})
        elapsed = time.monotonic() - start

        assert result == "ok"
        assert elapsed >= 0.05
        assert elapsed < 1.0  # Much less than default

    async def test_call_sleep_zero_duration(self) -> None:
        server = SleepMCPServer()

        start = time.monotonic()
        result = await server.call_tool("sleep", {"duration": 0})
        elapsed = time.monotonic() - start

        assert result == "ok"
        assert elapsed < 0.5

    async def test_call_sleep_clamped_to_max(self) -> None:
        """Duration above the maximum should be clamped to 5 seconds."""
        server = SleepMCPServer()

        start = time.monotonic()
        result = await server.call_tool("sleep", {"duration": 100})
        elapsed = time.monotonic() - start

        assert result == "ok"
        # Should be clamped to 5 seconds, not 100
        assert elapsed < 6.0

    async def test_call_sleep_negative_clamped_to_zero(self) -> None:
        """Negative duration should be clamped to 0."""
        server = SleepMCPServer()

        start = time.monotonic()
        result = await server.call_tool("sleep", {"duration": -5})
        elapsed = time.monotonic() - start

        assert result == "ok"
        assert elapsed < 0.5

    async def test_call_sleep_invalid_duration_type(self) -> None:
        server = SleepMCPServer()

        with pytest.raises(ValueError, match="must be a number"):
            await server.call_tool("sleep", {"duration": "fast"})

    async def test_call_unknown_tool(self) -> None:
        server = SleepMCPServer()

        with pytest.raises(ValueError, match="Unknown tool"):
            await server.call_tool("nap", {})

    async def test_sleep_description_mentions_message_splitting(self) -> None:
        """The tool description should guide the LLM to use it for multi-message."""
        server = SleepMCPServer()
        tool = (await server.list_tools())[0]

        assert "several short messages" in tool.description
        assert "sleep" in tool.name
