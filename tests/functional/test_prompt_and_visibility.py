from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import fetch_chat

pytestmark = pytest.mark.functional


def test_show_prompt_includes_sections_and_honors_tool_filter(functional_cli) -> None:
    functional_cli.write_prompt(
        "filtered_tools",
        "You are a prompt-preview test assistant.",
        toml="""
        [tools]
        disabled = ["wiki_create", "wiki_edit"]
        """,
    )
    functional_cli.start_chat("func-filtered-tools", prompt="filtered_tools").require_ok()

    preview = functional_cli.show_prompt("func-filtered-tools")

    assert preview.returncode == 0
    assert "--- Prompt Preview ---" in preview.stdout
    assert "--- Available Tools ---" in preview.stdout
    assert "--- Message Context ---" in preview.stdout
    assert "wiki_create" not in preview.stdout
    assert "wiki_edit" not in preview.stdout
    assert "wiki_list" in preview.stdout


def test_prompt_can_disable_entire_mcp_server_groups(functional_cli) -> None:
    functional_cli.write_prompt(
        "no_messages_tools",
        "You are a prompt-preview test assistant.",
        toml="""
        [mcp_servers]
        disabled = ["messages"]
        """,
    )
    functional_cli.start_chat("func-no-messages", prompt="no_messages_tools").require_ok()

    preview = functional_cli.show_prompt("func-no-messages")

    assert preview.returncode == 0
    assert "search_messages" not in preview.stdout
    assert "get_messages_by_timerange" not in preview.stdout
    assert "wiki_list" in preview.stdout


def test_prompt_config_persists_visibility_defaults(functional_cli) -> None:
    functional_cli.write_prompt(
        "hidden_defaults",
        "You are a hidden-defaults assistant.",
        toml="""
        show_reasoning = false
        show_tool_calls = false
        send_datetime = false
        """,
    )

    functional_cli.start_chat("func-hidden-defaults", prompt="hidden_defaults").require_ok()
    chat = fetch_chat(functional_cli.db_path, "func-hidden-defaults")

    assert chat is not None
    assert bool(chat["show_reasoning"]) is False
    assert bool(chat["show_tool_calls"]) is False
    assert bool(chat["send_datetime"]) is False


def test_send_datetime_false_removes_timestamp_from_future_prompt_context(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.write_prompt(
        "no_datetime",
        "You are a no-datetime assistant.",
        toml="send_datetime = false",
    )
    functional_cli.start_chat("func-no-datetime", prompt="no_datetime").require_ok()
    functional_cli.send_message(
        "func-no-datetime",
        "Reply with exactly NO_DATETIME.",
    ).require_ok()

    preview = functional_cli.show_prompt("func-no-datetime")

    assert preview.returncode == 0
    assert "[user] Reply with exactly NO_DATETIME." in preview.stdout
    assert "UTC] Reply with exactly NO_DATETIME." not in preview.stdout
