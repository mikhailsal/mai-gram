"""Functional message-search tool pipeline test.

What this test verifies:
1. The search_messages MCP tool can find seeded messages by keyword
   (uses SQL LIKE under the hood).
2. The AI can reference old conversation content when asked about past topics.
"""

from __future__ import annotations

from datetime import date

import async_timeout
import pytest

from mai_companion.db import get_session
from mai_companion.memory.messages import MessageStore
from mai_companion.mcp_servers.messages_server import MessagesMCPServer

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 120


@pytest.mark.asyncio
async def test_message_search_pipeline_and_recall(functional_runtime) -> None:
    """Verify search_messages tool works and AI can recall old conversation topics."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-search"
        search_day = date(2026, 1, 17)
        await functional_runtime.complete_onboarding(chat_id, companion_name="Echo")

        # Seed 15 messages about a Paris trip
        seeded = await functional_runtime.seed_messages(chat_id, "paris_trip.jsonl")
        assert seeded >= 15, f"Expected ≥15 seeded messages, got {seeded}"

        # ── Part 1: Directly exercise the search_messages MCP tool ──
        # Use a fresh session from the same DB (get_session auto-commits, so
        # seeded messages are visible). Search for a single keyword since
        # MessageStore.search uses SQL LIKE with a single pattern.
        async with get_session() as session:
            message_store = MessageStore(session)
            messages_server = MessagesMCPServer(message_store, chat_id)

            # Search for "Paris" — should match seeded messages
            search_result = await messages_server.call_tool("search_messages", {
                "query": "Paris",
                "limit": 10,
            })
            assert "No messages found" not in search_result, (
                f"search_messages returned no results for 'Paris'. "
                f"Seeded {seeded} messages but search found nothing."
            )
            assert "paris" in search_result.lower(), (
                f"search_messages should return Paris-related content. Got: {search_result[:300]}"
            )

            # Also verify a more specific search works
            eiffel_result = await messages_server.call_tool("search_messages", {
                "query": "Eiffel",
                "limit": 5,
            })
            assert "No messages found" not in eiffel_result, (
                f"search_messages should find 'Eiffel' in seeded messages"
            )

        # ── Part 2: Ask the AI about the old topic and check it references Paris ──
        response = await functional_runtime.send_message(
            chat_id,
            "Remember when we talked about visiting Paris? What were the highlights?",
            target_date=search_day,
        )
        assert response, "AI response should not be empty"
        lowered = response.lower()
        assert any(keyword in lowered for keyword in ("paris", "eiffel", "louvre", "versailles", "trip")), (
            f"AI should reference Paris trip details. Got: {response[:400]}"
        )
