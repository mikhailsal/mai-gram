from __future__ import annotations

import json

import pytest

from tests.functional.helpers.artifacts import list_backups
from tests.functional.helpers.parsing import extract_last_response_body, find_callback

pytestmark = pytest.mark.functional


def test_reset_creates_backup_and_clears_chat_artifacts(functional_cli) -> None:
    chat_id = "func-reset"
    payload = json.dumps(
        [
            {"role": "user", "content": "Hello reset"},
            {"role": "assistant", "content": "Reset me"},
        ]
    )
    json_path = functional_cli.write_json_fixture("reset-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    functional_cli.import_json(chat_id, json_path).require_ok()
    wiki_dir = functional_cli.chat_wiki_dir(chat_id)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "3000_profile.md").write_text("Profile fact", encoding="utf-8")
    functional_cli.repair_wiki(chat_id).require_ok()

    confirm = functional_cli.run_command(chat_id, "reset")
    assert confirm.returncode == 0
    assert "Reset this chat?" in confirm.stdout
    assert "All history and wiki entries will be deleted." in confirm.stdout

    result = functional_cli.send_callback(chat_id, f"confirm_reset:{chat_id}")
    assert result.returncode == 0
    assert "Creating backup" in result.stdout
    assert "Chat reset. All history deleted." in result.stdout
    assert list_backups(functional_cli.data_dir)
    assert not functional_cli.chat_data_dir(chat_id).exists()
    assert "(no messages)" in functional_cli.read_history(chat_id).stdout


def test_regenerate_matches_normal_response_contract(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    chat_id = "func-regen"
    payload = json.dumps(
        [
            {"role": "user", "content": "Reply with exactly PARITY."},
            {"role": "assistant", "content": "Old answer"},
        ]
    )
    json_path = functional_cli.write_json_fixture("regen-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    functional_cli.import_json(chat_id, json_path).require_ok()

    resend = functional_cli.run_command(chat_id, "resend_last")
    assert "Regenerate" in resend.stdout
    regen_prompt = functional_cli.send_callback(chat_id, "regen")
    assert "Regenerate this response?" in regen_prompt.stdout

    regenerated = functional_cli.send_callback(chat_id, "confirm_regen")
    assert regenerated.returncode == 0
    assert "--- AI Response" in regenerated.stdout
    assert "PARITY" in extract_last_response_body(regenerated.stdout).upper()
    assert "Regenerate" in regenerated.stdout
    assert "Cut this & above" in regenerated.stdout


def test_cut_above_hides_old_messages_from_prompt_but_keeps_history(functional_cli) -> None:
    chat_id = "func-cut"
    payload = json.dumps(
        [
            {"role": "user", "content": "First imported question"},
            {"role": "assistant", "content": "First imported answer"},
            {"role": "user", "content": "Second imported question"},
            {"role": "assistant", "content": "Second imported answer"},
        ]
    )
    json_path = functional_cli.write_json_fixture("cut-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    functional_cli.import_json(chat_id, json_path).require_ok()

    resend = functional_cli.run_command(chat_id, "resend_last")
    cut_callback = find_callback(resend.stdout, "cut:")
    confirm = functional_cli.send_callback(chat_id, cut_callback)
    assert "Cut this message and all above?" in confirm.stdout

    confirm_cut_callback = find_callback(confirm.stdout, "confirm_cut:")
    confirm_cut = functional_cli.send_callback(chat_id, confirm_cut_callback)
    assert confirm_cut.returncode == 0
    assert "History cut applied" in confirm_cut.stdout
    assert "hidden from AI" in confirm_cut.stdout

    preview = functional_cli.show_prompt(chat_id)
    history = functional_cli.read_history(chat_id)
    assert "[HISTORY NOTE]" in preview.stdout
    assert "search_messages" in preview.stdout
    assert "First imported question" in history.stdout
    assert "Second imported question" in history.stdout
