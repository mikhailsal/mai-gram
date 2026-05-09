from __future__ import annotations

import json

import pytest

from tests.functional.helpers.artifacts import list_backups
from tests.functional.helpers.parsing import extract_last_response_body, find_callback

pytestmark = pytest.mark.functional_local


def test_reset_creates_backup_and_clears_chat_artifacts(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-reset"
    payload = json.dumps(
        [
            {"role": "user", "content": "Hello reset"},
            {"role": "assistant", "content": "Reset me"},
        ]
    )
    json_path = cli.write_json_fixture("reset-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()
    wiki_dir = cli.chat_wiki_dir(chat_id)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "3000_profile.md").write_text("Profile fact", encoding="utf-8")
    cli.repair_wiki(chat_id).require_ok()

    confirm = cli.run_command(chat_id, "reset")
    assert confirm.returncode == 0
    assert "Reset this chat?" in confirm.stdout
    assert "All history and wiki entries will be deleted." in confirm.stdout

    result = cli.send_callback(chat_id, f"confirm_reset:{chat_id}")
    assert result.returncode == 0
    assert "Creating backup" in result.stdout
    assert "Chat reset. All history deleted." in result.stdout
    assert list_backups(cli.data_dir)
    assert not cli.chat_data_dir(chat_id).exists()
    assert "(no messages)" in cli.read_history(chat_id).stdout


@pytest.mark.functional_live
def test_regenerate_matches_normal_response_contract(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    chat_id = "func-regen"
    payload = json.dumps(
        [
            {"role": "user", "content": "Reply with exactly PARITY."},
            {"role": "assistant", "content": "Old answer"},
        ]
    )
    json_path = cli.write_json_fixture("regen-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()

    resend = cli.run_command(chat_id, "resend_last")
    assert "Regenerate" in resend.stdout
    regen_cb = find_callback(resend.stdout, "regen:")
    regen_prompt = cli.send_callback(chat_id, regen_cb)
    assert "Regenerate this response?" in regen_prompt.stdout

    confirm_cb = find_callback(regen_prompt.stdout, "confirm_regen:")
    regenerated = cli.send_callback_with_live_retry(chat_id, confirm_cb)
    assert regenerated.returncode == 0
    assert "--- AI Response" in regenerated.stdout
    assert "PARITY" in extract_last_response_body(regenerated.stdout).upper()
    assert "Regenerate" in regenerated.stdout
    assert "Cut this & above" in regenerated.stdout


@pytest.mark.functional_live
def test_regenerate_on_older_message_removes_subsequent_history(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    """Pressing Regenerate on a non-last assistant message should delete all
    messages that came after the user message preceding that response, then
    regenerate from that point.

    This test builds a three-turn conversation, discovers the DB message ID of
    the first assistant response via resend_last (which surfaces regen:<id>
    buttons), computes the first assistant's ID, then triggers regenerate
    targeting that older response.  Afterwards it verifies that turns 2 and 3
    have been removed from the persisted history.
    """
    cli = shared_functional_cli
    chat_id = "func-regen-older"
    payload = json.dumps(
        [
            {"role": "user", "content": "Reply with exactly ALPHA."},
            {"role": "assistant", "content": "First answer: ALPHA"},
            {"role": "user", "content": "Reply with exactly BETA."},
            {"role": "assistant", "content": "Second answer: BETA"},
            {"role": "user", "content": "Reply with exactly GAMMA."},
            {"role": "assistant", "content": "Third answer: GAMMA"},
        ]
    )
    json_path = cli.write_json_fixture("regen-older-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()

    history_before = cli.read_history(chat_id)
    assert "ALPHA" in history_before.stdout
    assert "BETA" in history_before.stdout
    assert "GAMMA" in history_before.stdout

    resend = cli.run_command(chat_id, "resend_last")
    last_regen_cb = find_callback(resend.stdout, "regen:")
    last_assistant_id = int(last_regen_cb.split(":")[1])
    first_assistant_id = last_assistant_id - 4

    regen_prompt = cli.send_callback(chat_id, f"regen:{first_assistant_id}")
    assert "Regenerate this response?" in regen_prompt.stdout

    confirm_cb = find_callback(regen_prompt.stdout, "confirm_regen:")
    regenerated = cli.send_callback_with_live_retry(chat_id, confirm_cb)
    assert regenerated.returncode == 0
    assert "--- AI Response" in regenerated.stdout

    history_after = cli.read_history(chat_id)

    assert "ALPHA" in history_after.stdout, (
        "The first user message ('Reply with exactly ALPHA.') should remain — "
        "it is the prompt being regenerated from"
    )
    assert "BETA" not in history_after.stdout, (
        "Second turn (BETA) still in history after regenerating from the first turn. "
        "Regenerate should have deleted all messages after the first user message."
    )
    assert "GAMMA" not in history_after.stdout, (
        "Third turn (GAMMA) still in history after regenerating from the first turn. "
        "Regenerate should have deleted all messages after the first user message."
    )


def test_cut_above_hides_old_messages_from_prompt_but_keeps_history(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-cut"
    payload = json.dumps(
        [
            {"role": "user", "content": "First imported question"},
            {"role": "assistant", "content": "First imported answer"},
            {"role": "user", "content": "Second imported question"},
            {"role": "assistant", "content": "Second imported answer"},
        ]
    )
    json_path = cli.write_json_fixture("cut-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    cli.import_json(chat_id, json_path).require_ok()

    resend = cli.run_command(chat_id, "resend_last")
    cut_callback = find_callback(resend.stdout, "cut:")
    confirm = cli.send_callback(chat_id, cut_callback)
    assert "Cut this message and all above?" in confirm.stdout

    confirm_cut_callback = find_callback(confirm.stdout, "confirm_cut:")
    confirm_cut = cli.send_callback(chat_id, confirm_cut_callback)
    assert confirm_cut.returncode == 0
    assert "✂️ History cut applied\n" in confirm_cut.stdout
    assert "hidden from AI" in confirm_cut.stdout

    preview = cli.show_prompt(chat_id)
    history = cli.read_history(chat_id)
    assert "[HISTORY NOTE]" in preview.stdout
    assert "search_messages" in preview.stdout
    assert "First imported question" in history.stdout
    assert "Second imported question" in history.stdout
