from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import read_debug_log_entries
from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional


def test_simple_message_produces_non_empty_ai_response(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.start_chat("func-conversation").require_ok()

    result = functional_cli.send_message("func-conversation", "Reply with exactly READY.")

    assert result.returncode == 0
    assert "--- AI Response" in result.stdout
    assert "READY" in extract_last_response_body(result.stdout).upper()

    history = functional_cli.read_history("func-conversation")
    assert "USER: Reply with exactly READY." in history.stdout
    assert "ASSISTANT:" in history.stdout


def test_debug_logging_writes_jsonl_and_summary(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.start_chat("func-debug").require_ok()

    result = functional_cli.send_message(
        "func-debug",
        "Reply with exactly DEBUG_OK.",
        debug=True,
    )

    assert result.returncode == 0
    assert "--- Debug Info ---" in result.stdout
    assert "--- Session Cost ---" in result.stdout
    entries = read_debug_log_entries(functional_cli.data_dir, "func-debug")
    assert entries
    assert any(entry.get("entry_type") in {"llm_call", "llm_stream_call"} for entry in entries)
    assert any(entry.get("request", {}).get("messages") for entry in entries)


def test_stream_debug_shows_intermediate_edits(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.start_chat("func-stream").require_ok()

    result = functional_cli.send_message(
        "func-stream",
        "In 2 short sentences, explain why rainbows appear in the sky.",
        stream_debug=True,
    )

    assert result.returncode == 0
    assert "Edited AI Response" in result.stdout


def test_prompt_preview_reflects_test_mode_and_real_mode(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    functional_cli.start_chat("func-prompt-preview").require_ok()
    functional_cli.send_message("func-prompt-preview", "Reply with exactly PREVIEW.").require_ok()

    preview = functional_cli.show_prompt("func-prompt-preview")
    real_preview = functional_cli.show_prompt("func-prompt-preview", real=True)

    assert preview.returncode == 0
    assert real_preview.returncode == 0
    assert "--- Prompt Preview ---" in preview.stdout
    assert "--- Available Tools ---" in preview.stdout
    assert "--- Message Context ---" in preview.stdout
    assert "[TEST MODE]" in preview.stdout
    assert "[TEST MODE]" not in real_preview.stdout
