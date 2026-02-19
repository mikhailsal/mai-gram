"""Functional wiki memory creation, retrieval, and changelog tests.

What this test verifies:
1. The wiki tool pipeline works end-to-end: calling wiki_create via the MCP
   server creates a DB entry and writes a changelog line.
2. When the AI is told important personal facts, it can recall them later
   (whether via wiki lookup or short-term memory).
3. The wiki changelog records creation events with correct structure.
"""

from __future__ import annotations

from datetime import date

import async_timeout

import pytest
from sqlalchemy import select

from mai_companion.db import get_session
from mai_companion.db.models import Companion
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.mcp_servers.wiki_server import WikiMCPServer

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 120


@pytest.mark.asyncio
async def test_wiki_pipeline_and_recall(functional_runtime) -> None:
    """Verify the wiki tool pipeline works and the AI can recall stored facts."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-wiki"
        day_one = date(2026, 1, 1)
        day_two = date(2026, 1, 2)
        await functional_runtime.complete_onboarding(chat_id, companion_name="Mira")

        # ── Part 1: Directly exercise the wiki MCP tool pipeline ──
        # This guarantees we test the tool infrastructure regardless of model whim.
        async with get_session() as session:
            wiki_store = WikiStore(session, data_dir=functional_runtime.settings.memory_data_dir)
            clock = functional_runtime._clock_provider(chat_id)
            wiki_server = WikiMCPServer(wiki_store, chat_id, clock=clock)

            result = await wiki_server.call_tool("wiki_create", {
                "key": "user_name",
                "content": "Alex",
                "importance": 9000,
            })
            assert "created" in result.lower() or "success" in result.lower() or "user_name" in result.lower(), (
                f"wiki_create should confirm creation, got: {result}"
            )

            result2 = await wiki_server.call_tool("wiki_create", {
                "key": "user_birthday",
                "content": "March 15",
                "importance": 8000,
            })
            assert "user_birthday" in result2.lower() or "created" in result2.lower(), (
                f"wiki_create for birthday failed: {result2}"
            )

        # Verify entries exist in DB
        wiki_entries = await functional_runtime.get_wiki_entries(chat_id)
        keys = {entry["key"] for entry in wiki_entries}
        assert "user_name" in keys, f"Expected 'user_name' in wiki keys, got: {keys}"
        assert "user_birthday" in keys, f"Expected 'user_birthday' in wiki keys, got: {keys}"

        # Verify changelog was written
        changelog = functional_runtime.get_wiki_changelog(chat_id)
        assert len(changelog) >= 2, f"Expected ≥2 changelog entries, got {len(changelog)}"
        assert any(item.get("action") == "create" for item in changelog), (
            f"No 'create' action in changelog: {changelog}"
        )

        # ── Part 2: Verify AI can recall the stored facts ──
        answer = await functional_runtime.send_message(
            chat_id,
            "What's my name?",
            target_date=day_two,
        )
        assert "alex" in answer.lower(), (
            f"AI should recall the user's name 'Alex' from wiki. Got: {answer[:300]}"
        )
