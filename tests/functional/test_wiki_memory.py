"""Functional wiki memory creation, retrieval, and changelog tests.

What this test verifies:
1. The wiki tool pipeline works end-to-end: calling wiki_create via the MCP
   server creates a DB entry and writes a changelog line.
2. When the AI is told important personal facts, it ACTUALLY calls the
   wiki_create tool (not just claims to remember).
3. The AI can recall stored facts in later conversation turns.
4. The wiki changelog records creation events with correct structure.
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


@pytest.mark.asyncio
async def test_ai_calls_wiki_create_on_personal_info(functional_runtime) -> None:
    """Verify the AI ACTUALLY calls wiki_create when told important personal facts.

    This is the critical end-to-end test: the user tells the AI their name
    and birthday, and we verify that wiki_create was invoked (not just that
    the AI claimed to remember). The wiki entries must appear in the DB and
    the changelog must record the creation events.
    """
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-wiki-tool-call"
        day = date(2026, 2, 1)
        await functional_runtime.complete_onboarding(chat_id, companion_name="Nova")

        # Before sending the message, wiki should be empty
        wiki_before = await functional_runtime.get_wiki_entries(chat_id)
        assert len(wiki_before) == 0, (
            f"Wiki should be empty before the test, got: {wiki_before}"
        )

        # Send a message with clear, important personal information that
        # should trigger wiki_create. We make it very explicit to maximize
        # the chance of tool use across different models.
        response = await functional_runtime.send_message(
            chat_id,
            "Hey! My name is Jordan and my birthday is July 4th, 1995. "
            "Please save this to your memory, it's really important to me.",
            target_date=day,
        )

        # ── Verify wiki entries were created ──
        wiki_after = await functional_runtime.get_wiki_entries(chat_id)
        assert len(wiki_after) >= 1, (
            f"AI should have called wiki_create at least once. "
            f"Wiki entries: {wiki_after}. AI response: {response[:300]}"
        )

        # Check that at least some of the key personal info was saved
        all_wiki_text = " ".join(
            f"{e['key']} {e['value']}".lower() for e in wiki_after
        )
        has_name = "jordan" in all_wiki_text
        has_birthday = "july" in all_wiki_text or "1995" in all_wiki_text or "birthday" in all_wiki_text
        assert has_name or has_birthday, (
            f"Wiki should contain 'jordan' or birthday info. "
            f"Wiki entries: {wiki_after}"
        )

        # ── Verify changelog was written by the tool (not manually) ──
        changelog = functional_runtime.get_wiki_changelog(chat_id)
        assert len(changelog) >= 1, (
            f"Expected ≥1 changelog entries from wiki_create tool calls, "
            f"got {len(changelog)}"
        )
        assert any(item.get("action") == "create" for item in changelog), (
            f"No 'create' action in changelog: {changelog}"
        )

        # ── Verify the AI can recall the info in a follow-up ──
        recall_response = await functional_runtime.send_message(
            chat_id,
            "What's my name?",
            target_date=day,
        )
        assert "jordan" in recall_response.lower(), (
            f"AI should recall 'Jordan' from wiki. Got: {recall_response[:300]}"
        )
