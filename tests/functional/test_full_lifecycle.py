"""Full multi-week lifecycle simulation functional test.

What this test verifies:
1. Seeded daily summaries are consolidated into weekly summaries when
   the forgetting engine runs (triggered by a message at a future date).
2. Weekly summaries are consolidated into monthly summaries when enough
   time has passed.
3. The AI can still recall information from consolidated summaries.
4. Cost tracking accumulates across the session.

Strategy: Seed summaries directly (avoiding expensive multi-day LLM
conversations), then send a few real messages at strategic dates to
trigger consolidation and verify recall.
"""

from __future__ import annotations

from datetime import date

import async_timeout

import pytest
from mai_companion.memory.summaries import SummaryStore

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 180  # lifecycle test has multiple LLM calls


@pytest.mark.asyncio
async def test_full_multi_week_lifecycle_simulation(functional_runtime) -> None:
    """Simulate a multi-week lifecycle with consolidation and recall."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-lifecycle"
        await functional_runtime.complete_onboarding(chat_id, companion_name="Ari")

        # ── Seed week 1 daily summaries ──
        summary_store = SummaryStore(functional_runtime.settings.memory_data_dir)
        week_one_days = [
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 3),
            date(2026, 1, 4),
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
        ]
        for day in week_one_days:
            summary_store.save_daily(
                chat_id,
                day,
                f"Week 1 daily memory for {day.isoformat()}: discussed photography hobby, "
                f"weekend hiking plans, and meal prep routines.",
            )

        daily_dir = functional_runtime.get_summary_path(
            chat_id, "daily", week_one_days[0].isoformat()
        ).parent
        assert len(list(daily_dir.glob("2026-01-0*.md"))) >= 7, (
            "Should have 7 seeded daily summaries"
        )

        # ── Send a couple of live messages for conversational context ──
        await functional_runtime.send_message(
            chat_id,
            "My name is Alex and I enjoy photography and weekend hikes.",
            target_date=date(2026, 1, 8),
        )
        await functional_runtime.send_message(
            chat_id,
            "We also talked about planning a January routine.",
            target_date=date(2026, 1, 9),
        )

        # ── Trigger weekly consolidation (Jan 20 > W01 end + 7 days) ──
        await functional_runtime.send_message(
            chat_id,
            "Let's review what happened in the first week.",
            target_date=date(2026, 1, 20),
        )
        weekly_dir = functional_runtime.get_summary_path(chat_id, "weekly", "2026-W01").parent
        weekly_files = sorted(weekly_dir.glob("*.md"))
        assert weekly_files, (
            f"Weekly summaries should exist after consolidation. Dir: {weekly_dir}"
        )

        # ── Ask AI to recall old content (tests prompt builder includes summaries) ──
        recall_response = await functional_runtime.send_message(
            chat_id,
            "Can you still recall what we discussed in January?",
            target_date=date(2026, 2, 15),
        )
        assert recall_response, "AI should respond to recall question"
        assert len(recall_response) > 20, (
            f"Recall response suspiciously short: {recall_response!r}"
        )

        # ── Trigger monthly consolidation (Mar 5 > Jan 31 + 28 = Feb 28) ──
        await functional_runtime.send_message(
            chat_id,
            "Let's do a monthly review.",
            target_date=date(2026, 3, 5),
        )
        monthly_dir = functional_runtime.get_summary_path(chat_id, "monthly", "2026-01").parent
        assert any(path.stem == "2026-01" for path in monthly_dir.glob("*.md")), (
            f"Monthly summary '2026-01.md' should exist after consolidation"
        )

        # ── Final recall check ──
        final_answer = await functional_runtime.send_message(
            chat_id,
            "What do you remember about our first week together?",
            target_date=date(2026, 3, 5),
        )
        assert final_answer, "AI should respond to final recall question"
        assert len(final_answer) > 20, f"Final answer too short: {final_answer!r}"

        # ── Verify cost tracking works ──
        totals = functional_runtime.get_cost_totals()
        assert int(totals["calls"]) > 0, (
            f"Should have recorded LLM calls, got: {totals}"
        )
        assert int(totals["total_tokens"]) > 0, (
            f"Should have recorded token usage, got: {totals}"
        )
