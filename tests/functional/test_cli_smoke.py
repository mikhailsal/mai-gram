from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import read_console_state

pytestmark = pytest.mark.functional


def test_help_prints_expected_options(functional_cli) -> None:
    result = functional_cli.run_cli("--help", allow_retry=False)

    assert result.returncode == 0
    assert "--start" in result.stdout
    assert "--command COMMAND" in result.stdout
    assert "--cb DATA" in result.stdout


def test_list_on_fresh_database_shows_no_chats(functional_cli) -> None:
    result = functional_cli.list_chats()

    assert result.returncode == 0
    assert "=== All Chats ===" in result.stdout
    assert "(no chats found)" in result.stdout


def test_sending_message_without_chat_id_fails(functional_cli) -> None:
    result = functional_cli.run_cli("Hello", allow_retry=False)

    assert result.returncode != 0
    assert "Error: no chat ID available." in result.output


def test_real_llm_action_without_api_key_fails(functional_cli) -> None:
    functional_cli.start_chat("func-no-key").require_ok()

    result = functional_cli.send_message(
        "func-no-key",
        "Reply with exactly NO_KEY.",
        env_overrides={"OPENROUTER_API_KEY": ""},
    )

    assert result.returncode != 0
    assert "Error: OPENROUTER_API_KEY is required." in result.output


def test_console_state_remembers_last_chat(functional_cli) -> None:
    first = functional_cli.read_history("func-state")
    second = functional_cli.run_cli("--history")

    assert first.returncode == 0
    assert second.returncode == 0
    assert read_console_state(functional_cli.root)["last_chat_id"] == "func-state"
    assert "=== History: func-state ===" in second.stdout


def test_callback_without_setup_prints_ignore_hint(functional_cli) -> None:
    result = functional_cli.send_callback("func-callback", "model:openrouter/free")

    assert result.returncode == 0
    assert "ignored" in result.stdout
    assert "no setup session active" in result.stdout


def test_missing_chat_history_and_prompt_preview_fail_cleanly(functional_cli) -> None:
    history = functional_cli.read_history("func-missing-history")
    prompt = functional_cli.show_prompt("func-missing-prompt")

    assert history.returncode == 0
    assert "=== History: func-missing-history ===" in history.stdout
    assert "(no messages)" in history.stdout
    assert prompt.returncode != 0
    assert "Error: no chat found for 'func-missing-prompt'. Run --start first." in prompt.output
