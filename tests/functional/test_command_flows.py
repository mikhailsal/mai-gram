from __future__ import annotations

import json

import pytest

from tests.functional.helpers.artifacts import fetch_chat, fetch_knowledge_entries, fetch_messages

pytestmark = pytest.mark.functional_local


def test_help_and_model_commands_cover_existing_and_missing_chats(shared_functional_cli) -> None:
    cli = shared_functional_cli
    help_result = cli.run_command("func-help", "help")
    missing_model = cli.run_command("func-missing-model", "model")
    cli.start_chat("func-model").require_ok()
    existing_model = cli.run_command("func-model", "model")

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


def test_model_command_offers_picker_with_cancel(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-model-picker"
    cli.start_chat(chat_id).require_ok()

    picker = cli.run_command(chat_id, "model")

    assert picker.returncode == 0
    assert "Current model: openrouter/free" in picker.stdout
    assert "Choose a new model:" in picker.stdout
    assert "--- Buttons ---" in picker.stdout
    assert "setmodel:openrouter/free" in picker.stdout
    assert "setmodel:openrouter/free-alt" in picker.stdout
    assert "Cancel  ->  cancel_action" in picker.stdout


def test_model_switch_callback_changes_model_without_wiping_history(
    shared_functional_cli,
) -> None:
    cli = shared_functional_cli
    chat_id = "func-model-switch"
    payload = json.dumps(
        [
            {"role": "user", "content": "Keep me after switch"},
            {"role": "assistant", "content": "I will stay"},
        ]
    )
    json_path = cli.write_json_fixture("model-switch-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()

    switch = cli.send_callback(chat_id, "setmodel:openrouter/free-alt")

    assert switch.returncode == 0
    assert "Model changed: openrouter/free → openrouter/free-alt" in switch.stdout

    chat = fetch_chat(cli.db_path, chat_id)
    assert chat is not None
    assert chat["llm_model"] == "openrouter/free-alt"
    # History must survive an in-place model switch (unlike /reset + /start).
    messages = fetch_messages(cli.db_path, chat_id)
    assert [m["content"] for m in messages] == ["Keep me after switch", "I will stay"]


def test_toggles_and_timezone_persist_to_chat_record(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-flags"
    cli.start_chat(chat_id).require_ok()

    cli.run_command(chat_id, "reasoning").require_ok()
    cli.run_command(chat_id, "toolcalls").require_ok()
    cli.run_command(chat_id, "datetime").require_ok()
    result_tz = cli.run_command(chat_id, "timezone", args="Europe/Moscow")
    result_tz.require_ok()

    chat = fetch_chat(cli.db_path, chat_id)
    assert chat is not None
    assert bool(chat["show_reasoning"]) is False
    assert bool(chat["show_tool_calls"]) is False
    assert bool(chat["send_datetime"]) is False
    assert chat["timezone"] == "Europe/Moscow"
    assert "Timezone set to: Europe/Moscow\n" in result_tz.stdout


@pytest.mark.functional_live
def test_datetime_and_timezone_affect_future_prompt_assembly(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    chat_id = "func-timezone"
    cli.start_chat(chat_id).require_ok()
    cli.run_command(chat_id, "datetime").require_ok()
    cli.run_command(chat_id, "timezone", args="Europe/Moscow").require_ok()
    cli.send_message_with_live_retry(
        chat_id,
        "Reply with exactly FIRST_PASS.",
    ).require_ok()

    hidden_preview = cli.show_prompt(chat_id)
    assert "[user] Reply with exactly FIRST_PASS." in hidden_preview.stdout
    assert "Europe/Moscow] Reply with exactly FIRST_PASS." not in hidden_preview.stdout

    cli.run_command(chat_id, "datetime").require_ok()
    cli.send_message_with_live_retry(
        chat_id,
        "Reply with exactly SECOND_PASS.",
    ).require_ok()

    visible_preview = cli.show_prompt(chat_id)
    assert "SECOND_PASS" in visible_preview.stdout
    assert "Europe/Moscow" in visible_preview.stdout


def test_resend_last_replays_last_assistant_message(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-resend"
    payload = json.dumps(
        [
            {"role": "user", "content": "Imported hello"},
            {"role": "assistant", "content": "Imported assistant reply"},
        ]
    )
    json_path = cli.write_json_fixture("resend-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()
    result = cli.run_command(chat_id, "resend_last")

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

    assert "Access denied." in denied.stdout
    assert "Chat created!" in allowed.stdout

    first_env = functional_cli_factory("functional-a")
    second_env = functional_cli_factory("functional-b")
    first_env.read_history("func-isolated").require_ok()
    result = second_env.run_cli("--history", allow_retry=False)

    assert result.returncode != 0
    assert "Error: no chat ID available." in result.output


def test_different_chat_ids_keep_history_and_wiki_state_separate(shared_functional_cli) -> None:
    cli = shared_functional_cli
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

    cli.start_chat(first_chat_id).require_ok()
    cli.start_chat(second_chat_id).require_ok()
    cli.import_json(
        first_chat_id,
        cli.write_json_fixture("separate-a.json", first_payload),
    ).require_ok()
    cli.import_json(
        second_chat_id,
        cli.write_json_fixture("separate-b.json", second_payload),
    ).require_ok()

    first_wiki_dir = cli.chat_wiki_dir(first_chat_id)
    second_wiki_dir = cli.chat_wiki_dir(second_chat_id)
    first_wiki_dir.mkdir(parents=True, exist_ok=True)
    second_wiki_dir.mkdir(parents=True, exist_ok=True)
    (first_wiki_dir / "1001_profile.md").write_text("Favorite tea: oolong.", encoding="utf-8")
    (second_wiki_dir / "1002_profile.md").write_text("Favorite tea: sencha.", encoding="utf-8")
    cli.repair_wiki(first_chat_id).require_ok()
    cli.repair_wiki(second_chat_id).require_ok()

    first_history = cli.read_history(first_chat_id)
    second_history = cli.read_history(second_chat_id)
    first_wiki = cli.read_wiki(first_chat_id)
    second_wiki = cli.read_wiki(second_chat_id)

    assert "History only for alpha" in first_history.stdout
    assert "History only for beta" not in first_history.stdout
    assert "History only for beta" in second_history.stdout
    assert "History only for alpha" not in second_history.stdout
    assert "oolong" in first_wiki.stdout.lower()
    assert "sencha" not in first_wiki.stdout.lower()
    assert "sencha" in second_wiki.stdout.lower()
    assert "oolong" not in second_wiki.stdout.lower()
    assert len(fetch_messages(cli.db_path, first_chat_id)) == 2
    assert len(fetch_messages(cli.db_path, second_chat_id)) == 2
    assert {entry["value"] for entry in fetch_knowledge_entries(cli.db_path, first_chat_id)} == {
        "Favorite tea: oolong."
    }
    assert {entry["value"] for entry in fetch_knowledge_entries(cli.db_path, second_chat_id)} == {
        "Favorite tea: sencha."
    }
