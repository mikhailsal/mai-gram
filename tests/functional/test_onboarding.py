from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import fetch_chat
from tests.functional.helpers.cli import FREE_MODEL

pytestmark = pytest.mark.functional


def test_start_with_model_and_prompt_creates_chat(functional_cli) -> None:
    result = functional_cli.start_chat("func-onboarding")

    assert result.returncode == 0
    assert "Chat created!" in result.stdout
    assert f"Model: {FREE_MODEL}" in result.stdout

    chat = fetch_chat(functional_cli.db_path, "func-onboarding")
    assert chat is not None
    assert chat["llm_model"] == FREE_MODEL
    assert chat["prompt_name"] == "default"
    assert bool(chat["show_reasoning"]) is True
    assert bool(chat["show_tool_calls"]) is True
    assert bool(chat["send_datetime"]) is True


def test_callback_setup_matches_one_shot_setup(functional_cli) -> None:
    direct_chat = "func-direct"
    callback_chat = "func-callback-setup"

    functional_cli.start_chat(direct_chat).require_ok()
    functional_cli.run_cli(
        "-c",
        callback_chat,
        "--start",
        "--cb",
        f"model:{FREE_MODEL}",
        "--cb",
        "prompt:default",
        "--cb",
        "template:empty",
    ).require_ok()

    direct_row = fetch_chat(functional_cli.db_path, direct_chat)
    callback_row = fetch_chat(functional_cli.db_path, callback_chat)
    assert direct_row is not None
    assert callback_row is not None
    assert direct_row["llm_model"] == callback_row["llm_model"]
    assert direct_row["prompt_name"] == callback_row["prompt_name"]
    assert direct_row["system_prompt"] == callback_row["system_prompt"]
    assert bool(direct_row["show_reasoning"]) == bool(callback_row["show_reasoning"])
    assert bool(direct_row["show_tool_calls"]) == bool(callback_row["show_tool_calls"])
    assert bool(direct_row["send_datetime"]) == bool(callback_row["send_datetime"])


def test_rerunning_start_reports_existing_chat(functional_cli) -> None:
    functional_cli.start_chat("func-existing").require_ok()

    result = functional_cli.start_chat("func-existing")

    assert result.returncode == 0
    assert "Chat already configured" in result.stdout
    assert "Use /reset to start over." in result.stdout


def test_custom_prompt_setup_works_with_setup_text(functional_cli) -> None:
    result = functional_cli.run_cli(
        "-c",
        "func-custom",
        "--start",
        "--cb",
        f"model:{FREE_MODEL}",
        "--cb",
        "prompt:__custom__",
        "--cb",
        "template:empty",
        "You are a custom prompt for integration tests.",
    )

    assert result.returncode == 0
    assert "Type your custom system prompt:" in result.stdout
    assert "Chat created!" in result.stdout

    chat = fetch_chat(functional_cli.db_path, "func-custom")
    assert chat is not None
    assert chat["prompt_name"] is None
    assert "custom prompt for integration tests" in chat["system_prompt"].lower()


def test_created_chat_appears_in_list(functional_cli) -> None:
    functional_cli.start_chat("func-listing").require_ok()

    result = functional_cli.list_chats()

    assert result.returncode == 0
    assert "func-listing" in result.stdout
    assert FREE_MODEL in result.stdout
