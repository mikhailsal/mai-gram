"""Functional test for the sleep tool and multi-message delivery.

What this test verifies:
1. A purpose-built "robot" companion (English, formal, concise, obedient)
   can be created directly in the DB without going through onboarding.
2. When explicitly asked to send multiple separate messages, the companion
   uses the ``sleep`` tool to split its response into several parts.
3. Each part is delivered as a separate ``--- AI Response ---`` block in
   the console output, proving the intermediate-message pipeline works
   end-to-end through the real LLM → bridge → handler → messenger stack.

Strategy: We create a minimal, instruction-following companion (low
creativity, high directness, formal style) and give it a very explicit
prompt asking for exactly 3 separate messages.  Because LLM behaviour
is non-deterministic, we allow a retry and accept ≥ 2 response blocks
as success.
"""

from __future__ import annotations

import async_timeout
import pytest

from mai_companion.personality.character import (
    CommunicationStyle,
    Gender,
    Verbosity,
)

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 120


@pytest.mark.asyncio
async def test_sleep_tool_produces_multiple_messages(functional_runtime) -> None:
    """Verify that the sleep tool delivers intermediate messages."""
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-sleep-multi"

        # ── Step 1: Create a robot-like companion directly (zero LLM calls) ──
        companion = await functional_runtime.create_companion_directly(
            chat_id,
            name="Robo",
            language="English",
            traits={
                "warmth": 0.3,
                "humor": 0.0,
                "patience": 0.9,
                "directness": 0.9,
                "laziness": 0.0,
                "mood_volatility": 0.1,
            },
            communication_style=CommunicationStyle.FORMAL,
            verbosity=Verbosity.CONCISE,
            gender=Gender.NEUTRAL,
        )
        assert companion is not None
        assert companion.name == "Robo"

        # ── Step 2: Send a message that strongly encourages multi-message ──
        # The prompt is deliberately explicit to maximise the chance the
        # LLM will use the sleep tool.
        prompt = (
            "I need you to tell me exactly 3 facts about the Moon. "
            "Send each fact as a SEPARATE message — use the sleep tool "
            "between them so I receive 3 individual messages. "
            "Keep each message to one sentence."
        )

        responses = await functional_runtime.send_message_multi(chat_id, prompt)

        # ── Step 3: Verify we got multiple response blocks ──
        # Ideal: 3 blocks (one per fact).
        # Acceptable: ≥ 2 blocks (proves the sleep tool was used at least once).
        # If only 1 block, the LLM chose not to use the tool — retry once.
        if len(responses) < 2:
            # Retry with an even more explicit prompt
            retry_prompt = (
                "Please send me 3 separate messages. After each message, "
                "call the sleep tool before writing the next one. "
                "Message 1: say 'First'. Message 2: say 'Second'. "
                "Message 3: say 'Third'."
            )
            responses = await functional_runtime.send_message_multi(
                chat_id, retry_prompt,
            )

        assert len(responses) >= 2, (
            f"Expected ≥2 separate AI response blocks (proving sleep tool usage), "
            f"but got {len(responses)}: {responses!r}"
        )

        # Every response block should contain non-trivial text
        for i, resp in enumerate(responses):
            assert len(resp.strip()) > 0, (
                f"Response block {i} is empty: {resp!r}"
            )

        # ── Step 4: Verify debug log shows sleep tool calls ──
        from datetime import date, timezone, datetime

        today = datetime.now(timezone.utc).date()
        debug_entries = functional_runtime.get_debug_log(chat_id, today)
        if debug_entries:
            # Check that at least one entry mentions the sleep tool
            all_text = str(debug_entries)
            assert "sleep" in all_text.lower(), (
                "Debug log should contain references to the sleep tool"
            )
