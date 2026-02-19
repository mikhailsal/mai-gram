"""Functional checks for tool-call representation in daily summaries.

What this test verifies:
1. When a conversation includes tool-call interactions (wiki creates, searches),
   the resulting daily summary is clean prose — no raw tool syntax, no JSON
   fragments, no "tool_calls" keys.
2. The summary captures the *intent* of the actions (e.g. "remembered user's
   birthday") rather than technical artifacts.

Strategy: We seed messages that describe tool-call interactions (as they would
appear in real conversation), trigger a daily summary via the summarizer, and
inspect the output for forbidden patterns and expected content.
"""

from __future__ import annotations

from datetime import date

import async_timeout

import pytest

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 120


@pytest.mark.asyncio
async def test_tool_calls_are_not_rendered_as_raw_syntax_in_summary(functional_runtime) -> None:
    """Verify daily summaries contain prose, not raw tool syntax."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-tool-summary"
        target_day = date(2026, 1, 5)
        await functional_runtime.complete_onboarding(chat_id, companion_name="Nia")

        # Send enough messages to trigger a daily summary (threshold=5).
        # Each send_message creates user+assistant = 2 DB messages.
        # We need count % 5 == 0 → 5 exchanges should give us ~10 messages.
        conversations = [
            "My name is Alex and my birthday is March 15. Please remember this.",
            "I also love hiking in the mountains on weekends.",
            "What's the best trail near Portland for a day hike?",
            "I'm planning a trip to Paris next summer. Any tips?",
            "Can you remind me what my birthday is?",
        ]
        for msg in conversations:
            await functional_runtime.send_message(
                chat_id, msg, target_date=target_day,
            )

        # Verify summary was created
        summary_path = functional_runtime.get_summary_path(chat_id, "daily", target_day.isoformat())
        assert summary_path.exists(), (
            f"Daily summary should exist at {summary_path} after {len(conversations)} exchanges"
        )

        summary_text = summary_path.read_text(encoding="utf-8")
        lowered = summary_text.lower()

        # ── Forbidden patterns: raw tool syntax should never appear ──
        forbidden = [
            "wiki_create(",
            "wiki_edit(",
            "search_messages(",
            '"tool_calls"',
            '"function":{',
            "tool execution error",
            '"name":"wiki_',
            "arguments={",
        ]
        for pattern in forbidden:
            assert pattern not in lowered, (
                f"Summary contains raw tool syntax '{pattern}':\n{summary_text[:500]}"
            )

        # ── Expected content: summary should capture conversation topics ──
        assert len(summary_text.strip()) > 50, (
            f"Summary is suspiciously short ({len(summary_text)} chars): {summary_text!r}"
        )
        # At least some of the conversation topics should appear
        topic_hits = sum(
            1 for keyword in ("alex", "birthday", "march", "hiking", "paris", "portland", "trip")
            if keyword in lowered
        )
        assert topic_hits >= 2, (
            f"Summary should mention at least 2 conversation topics, found {topic_hits}. "
            f"Summary:\n{summary_text[:500]}"
        )
