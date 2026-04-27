from __future__ import annotations

import json

import pytest

from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional


def test_import_openai_style_json_and_show_history(functional_cli) -> None:
    chat_id = "func-import-openai"
    payload = json.dumps(
        [
            {"role": "system", "content": "ignored system"},
            {"role": "user", "content": "Hello from import"},
            {"role": "assistant", "content": "Imported reply"},
        ]
    )
    json_path = functional_cli.write_json_fixture("openai-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    imported = functional_cli.import_json(chat_id, json_path)
    history = functional_cli.read_history(chat_id)

    assert imported.returncode == 0
    assert "Imported 2 messages into chat 'func-import-openai'." in imported.stdout
    assert "USER: Hello from import" in history.stdout
    assert "ASSISTANT: Imported reply" in history.stdout


def test_import_proxy_json_and_continue_conversation(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    chat_id = "func-import-proxy"
    payload = json.dumps(
        {
            "timestamp": "2026-04-25T12:00:00Z",
            "request_body": {
                "messages": [
                    {"role": "user", "content": "Imported question"},
                ]
            },
            "response_body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Imported answer",
                        }
                    }
                ]
            },
        }
    )
    json_path = functional_cli.write_json_fixture("proxy-import.json", payload)

    functional_cli.start_chat(chat_id).require_ok()
    imported = functional_cli.import_json(chat_id, json_path)
    follow_up = functional_cli.send_message_with_live_retry(
        chat_id,
        "Reply with exactly CONTINUED.",
    )

    assert imported.returncode == 0
    assert "Imported 2 messages into chat 'func-import-proxy'." in imported.stdout
    assert "CONTINUED" in extract_last_response_body(follow_up.stdout).upper()


def test_invalid_and_empty_import_inputs_fail(functional_cli) -> None:
    chat_id = "func-import-errors"
    invalid_json = functional_cli.write_json_fixture("invalid-import.json", "{not valid json")
    empty_json = functional_cli.write_json_fixture("empty-import.json", "[]")

    functional_cli.start_chat(chat_id).require_ok()

    invalid = functional_cli.import_json(chat_id, invalid_json)
    empty = functional_cli.import_json(chat_id, empty_json)

    assert invalid.returncode != 0
    assert "Error: Invalid JSON:" in invalid.output
    assert empty.returncode != 0
    assert "Error: no messages could be imported" in empty.output
