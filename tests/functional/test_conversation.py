from __future__ import annotations

import pytest

from tests.functional.helpers.artifacts import read_debug_log_entries
from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional


def test_simple_message_produces_non_empty_ai_response(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.start_chat("func-conversation").require_ok()

    result = cli.send_message_with_live_retry(
        "func-conversation",
        "Reply with exactly READY.",
    )

    assert result.returncode == 0
    assert "--- AI Response ---" in result.stdout
    assert "READY" in extract_last_response_body(result.stdout).upper()

    history = cli.read_history("func-conversation")
    assert "USER: Reply with exactly READY." in history.stdout
    assert "ASSISTANT:" in history.stdout


def test_debug_logging_writes_jsonl_and_summary(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.start_chat("func-debug").require_ok()

    result = cli.send_message_with_live_retry(
        "func-debug",
        "Reply with exactly DEBUG_OK.",
        debug=True,
    )

    assert result.returncode == 0
    assert "--- Debug Info ---" in result.stdout
    assert "--- Session Cost ---" in result.stdout
    entries = read_debug_log_entries(cli.data_dir, "func-debug")
    assert entries
    assert any(entry.get("entry_type") in {"llm_call", "llm_stream_call"} for entry in entries)
    assert any(entry.get("request", {}).get("messages") for entry in entries)


def test_stream_debug_shows_intermediate_edits(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.start_chat("func-stream").require_ok()

    result = cli.send_message_with_live_retry(
        "func-stream",
        "In 2 short sentences, explain why rainbows appear in the sky.",
        stream_debug=True,
    )

    assert result.returncode == 0
    assert "Edited AI Response" in result.stdout


def test_prompt_preview_reflects_test_mode_and_real_mode(
    shared_functional_cli,
    requires_openrouter_api_key,
) -> None:
    cli = shared_functional_cli
    cli.start_chat("func-prompt-preview").require_ok()
    cli.send_message_with_live_retry(
        "func-prompt-preview",
        "Reply with exactly PREVIEW.",
    ).require_ok()

    preview = cli.show_prompt("func-prompt-preview")
    real_preview = cli.show_prompt("func-prompt-preview", real=True)

    assert preview.returncode == 0
    assert real_preview.returncode == 0
    assert "--- Prompt Preview ---" in preview.stdout
    assert "--- Available Tools ---" in preview.stdout
    assert "--- Message Context ---" in preview.stdout
    assert "[TEST MODE]" in preview.stdout
    assert "[TEST MODE]" not in real_preview.stdout
