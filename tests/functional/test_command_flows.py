from __future__ import annotations

import json

import pytest

from tests.functional.helpers.artifacts import fetch_chat, fetch_knowledge_entries, fetch_messages

pytestmark = pytest.mark.functional


def test_help_and_model_commands_cover_existing_and_missing_chats(functional_cli) -> None:
    help_result = functional_cli.run_command("func-help", "help")
    missing_model = functional_cli.run_command("func-missing-model", "model")
    functional_cli.start_chat("func-model").require_ok()
    existing_model = functional_cli.run_command("func-model", "model")

    assert help_result.returncode == 0
    assert "/start" in help_result.stdout
    assert "/reset" in help_result.stdout
    assert "/model" in help_result.stdout
    assert "/timezone" in help_result.stdout
    assert "/datetime" in help_result.stdout
    assert "/reasoning" in help_result.stdout
    assert "/toolcalls" in help_result.stdout
    assert "/resend_last" in help_result.stdout
    assert "/help" in help_result.stdout
    assert "No chat exists yet." in missing_model.stdout
    assert "Current model: openrouter/free" in existing_model.stdout


def test_toggles_and_timezone_persist_to_chat_record(functional_cli) -> None:
    chat_id = "func-flags"
    functional_cli.start_chat(chat_id).require_ok()

    functional_cli.run_command(chat_id, "reasoning").require_ok()
    functional_cli.run_command(chat_id, "toolcalls").require_ok()
    functional_cli.run_command(chat_id, "datetime").require_ok()
    functional_cli.run_command(chat_id, "timezone", args="Europe/Moscow").require_ok()

    chat = fetch_chat(functional_cli.db_path, chat_id)
    assert chat is not None
    assert bool(chat["show_reasoning"]) is False
    assert bool(chat["show_tool_calls"]) is False
    assert bool(chat["send_datetime"]) is False
    assert chat["timezone"] == "Europe/Moscow"


def test_datetime_and_timezone_affect_future_prompt_assembly(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    chat_id = "func-timezone"
    functional_cli.start_chat(chat_id).require_ok()
    functional_cli.run_command(chat_id, "datetime").require_ok()
    functional_cli.run_command(chat_id, "timezone", args="Europe/Moscow").require_ok()
    functional_cli.send_message_with_live_retry(
        chat_id,
        "Reply with exactly FIRST_PASS.",
    ).require_ok()

    hidden_preview = functional_cli.show_prompt(chat_id)
    assert "[user] Reply with exactly FIRST_PASS." in hidden_preview.stdout
    assert "Europe/Moscow] Reply with exactly FIRST_PASS." not in hidden_preview.stdout

    functional_cli.run_command(chat_id, "datetime").require_ok()
    functional_cli.send_message_with_live_retry(
        chat_id,
        "Reply with exactly SECOND_PASS.",
    ).require_ok()

    visible_preview = functional_cli.show_prompt(chat_id)
    assert "SECOND_PASS" in visible_preview.stdout
    assert "Europe/Moscow" in visible_preview.stdout


def test_resend_last_replays_last_assistant_message(functional_cli) -> None:
    chat_id = "func-resend"
    payload = json.dumps(
        [
            {"role": "user", "content": "Imported hello"},
            {"role": "assistant", "content": "Imported assistant reply"},
        ]
    )
    json_path = functional_cli.write_json_fixture("resend-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    functional_cli.import_json(chat_id, json_path).require_ok()
    result = functional_cli.run_command(chat_id, "resend_last")

    assert result.returncode == 0
    assert "Imported assistant reply" in result.stdout
    assert "Resent last AI message" in result.stdout
    assert "Regenerate" in result.stdout
    assert "Cut this & above" in result.stdout


def test_access_control_and_isolated_state(functional_cli, functional_cli_factory) -> None:
    env_overrides = {"ALLOWED_USERS": "approved-user"}

    denied = functional_cli.start_chat(
        "func-private",
        user_id="intruder",
        env_overrides=env_overrides,
    )
    allowed = functional_cli.start_chat(
        "func-private-allowed",
        user_id="approved-user",
        env_overrides=env_overrides,
    )

    assert "Access denied" in denied.stdout
    assert "Chat created!" in allowed.stdout

    first_env = functional_cli_factory("functional-a")
    second_env = functional_cli_factory("functional-b")
    first_env.read_history("func-isolated").require_ok()
    result = second_env.run_cli("--history", allow_retry=False)

    assert result.returncode != 0
    assert "Error: no chat ID available." in result.output


def test_different_chat_ids_keep_history_and_wiki_state_separate(functional_cli) -> None:
    first_chat_id = "func-separate-a"
    second_chat_id = "func-separate-b"

    first_payload = json.dumps(
        [
            {"role": "user", "content": "History only for alpha"},
            {"role": "assistant", "content": "Reply only for alpha"},
        ]
    )
    second_payload = json.dumps(
        [
            {"role": "user", "content": "History only for beta"},
            {"role": "assistant", "content": "Reply only for beta"},
        ]
    )

    functional_cli.start_chat(first_chat_id).require_ok()
    functional_cli.start_chat(second_chat_id).require_ok()
    functional_cli.import_json(
        first_chat_id,
        functional_cli.write_json_fixture("separate-a.json", first_payload),
    ).require_ok()
    functional_cli.import_json(
        second_chat_id,
        functional_cli.write_json_fixture("separate-b.json", second_payload),
    ).require_ok()

    first_wiki_dir = functional_cli.chat_wiki_dir(first_chat_id)
    second_wiki_dir = functional_cli.chat_wiki_dir(second_chat_id)
    first_wiki_dir.mkdir(parents=True, exist_ok=True)
    second_wiki_dir.mkdir(parents=True, exist_ok=True)
    (first_wiki_dir / "1001_profile.md").write_text("Favorite tea: oolong.", encoding="utf-8")
    (second_wiki_dir / "1002_profile.md").write_text("Favorite tea: sencha.", encoding="utf-8")
    functional_cli.repair_wiki(first_chat_id).require_ok()
    functional_cli.repair_wiki(second_chat_id).require_ok()

    first_history = functional_cli.read_history(first_chat_id)
    second_history = functional_cli.read_history(second_chat_id)
    first_wiki = functional_cli.read_wiki(first_chat_id)
    second_wiki = functional_cli.read_wiki(second_chat_id)

    assert "History only for alpha" in first_history.stdout
    assert "History only for beta" not in first_history.stdout
    assert "History only for beta" in second_history.stdout
    assert "History only for alpha" not in second_history.stdout
    assert "oolong" in first_wiki.stdout.lower()
    assert "sencha" not in first_wiki.stdout.lower()
    assert "sencha" in second_wiki.stdout.lower()
    assert "oolong" not in second_wiki.stdout.lower()
    assert len(fetch_messages(functional_cli.db_path, first_chat_id)) == 2
    assert len(fetch_messages(functional_cli.db_path, second_chat_id)) == 2
    assert {
        entry["value"] for entry in fetch_knowledge_entries(functional_cli.db_path, first_chat_id)
    } == {"Favorite tea: oolong."}
    assert {
        entry["value"] for entry in fetch_knowledge_entries(functional_cli.db_path, second_chat_id)
    } == {"Favorite tea: sencha."}
