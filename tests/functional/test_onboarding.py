"""Functional onboarding flow via the console messenger.

What this test verifies:
- /start command produces the welcome message with language prompt
- Walking through the onboarding wizard (language → name → preset → appearance → confirm)
  creates a Companion record in the DB with correct fields
- The companion can respond to a first message without errors
"""

from __future__ import annotations

import json

import async_timeout

import pytest

pytestmark = pytest.mark.functional

TIMEOUT_SECONDS = 90  # generous for LLM round-trips during onboarding


@pytest.mark.asyncio
async def test_onboarding_via_console(functional_runtime) -> None:
    async with async_timeout.timeout(TIMEOUT_SECONDS):
        chat_id = "func-onboarding"

        # Step 1: /start → expect welcome text
        welcome = await functional_runtime.send_start(chat_id)
        assert "Hello! I'm about to become your AI companion." in welcome, (
            f"Welcome message missing expected greeting. Got:\n{welcome[:300]}"
        )
        assert "what language would you like me to speak" in welcome.lower(), (
            f"Welcome message missing language prompt. Got:\n{welcome[:300]}"
        )

        # Step 2: Walk through onboarding wizard
        await functional_runtime.send_message(chat_id, "English")
        await functional_runtime.send_message(chat_id, "Nova")
        await functional_runtime.send_callback(chat_id, "personality:presets")
        await functional_runtime.send_callback(chat_id, "preset:balanced_friend")
        await functional_runtime.send_callback(chat_id, "preset_confirm:yes")
        await functional_runtime.send_callback(chat_id, "appearance:skip")
        await functional_runtime.send_callback(chat_id, "confirm:yes")

        # Step 3: Verify companion was created in DB
        companion = await functional_runtime.get_companion(chat_id)
        assert companion is not None, "Companion should exist in DB after onboarding"
        assert companion.name == "Nova", f"Expected name 'Nova', got '{companion.name}'"
        assert companion.human_language.lower() == "english", (
            f"Expected language 'english', got '{companion.human_language}'"
        )
        traits = json.loads(companion.personality_traits)
        assert isinstance(traits, dict) and len(traits) > 0, (
            f"Personality traits should be a non-empty dict, got: {traits}"
        )
        assert "warmth" in traits, f"'warmth' trait missing from: {list(traits.keys())}"

        # Step 4: Send a real message and verify the AI responds coherently
        response = await functional_runtime.send_message(
            chat_id,
            "Hi Nova! Say hello in your style.",
        )
        assert response, "AI response should not be empty"
        assert len(response) > 10, f"AI response suspiciously short: {response!r}"
        assert "trouble thinking right now" not in response.lower(), (
            f"AI returned error fallback: {response[:200]}"
        )
