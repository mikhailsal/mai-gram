from __future__ import annotations

import json
import time

import pytest

from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional

_MAX_LLM_RETRIES = 5


def test_import_openai_style_json_and_show_history(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-import-openai"
    payload = json.dumps(
        [
            {"role": "system", "content": "ignored system"},
            {"role": "user", "content": "Hello from import"},
            {"role": "assistant", "content": "Imported reply"},
        ]
    )
    json_path = cli.write_json_fixture("openai-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path)
    history = cli.read_history(chat_id)

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

    assert imported.returncode == 0
    assert "Imported 2 messages into chat 'func-import-proxy'." in imported.stdout

    last_output = ""
    for attempt in range(1, _MAX_LLM_RETRIES + 1):
        follow_up = functional_cli.send_message_with_live_retry(
            chat_id,
            "Reply with exactly CONTINUED.",
        )
        try:
            last_output = extract_last_response_body(follow_up.stdout)
            if "CONTINUED" in last_output.upper():
                break
        except AssertionError:
            last_output = follow_up.output

        if attempt < _MAX_LLM_RETRIES:
            time.sleep(2.0 * attempt)

    assert "CONTINUED" in last_output.upper(), (
        f"Expected 'CONTINUED' after {_MAX_LLM_RETRIES} outer attempts, got: {last_output!r}"
    )


def test_import_with_reasoning_template_transforms_reasoning(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-import-reasoning-tmpl"
    payload = json.dumps(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "*   Simple arithmetic.\n    *   2 + 2 = 4.",
            },
        ]
    )
    json_path = cli.write_json_fixture("reasoning-tmpl-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path, reasoning_template="gemma_reasoning_prefill")
    history = cli.read_history(chat_id)

    assert imported.returncode == 0
    assert "reasoning template: gemma_reasoning_prefill" in imported.stdout
    assert "<thought>" in history.stdout
    assert "Simple arithmetic." in history.stdout
    assert "<content>" in history.stdout
    assert "The answer is 4." in history.stdout


def test_import_with_reasoning_template_custom_params(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-import-custom-params"
    payload = json.dumps(
        [
            {"role": "user", "content": "Analyze this."},
            {
                "role": "assistant",
                "content": "Here is my analysis.",
                "reasoning_content": "*   Deep thought process here.",
            },
        ]
    )
    json_path = cli.write_json_fixture("custom-params-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(
        chat_id,
        json_path,
        reasoning_template="gemma_reasoning",
        reasoning_template_params={"reasoning_field": "analysis"},
    )
    history = cli.read_history(chat_id)

    assert imported.returncode == 0
    assert "<analysis>" in history.stdout
    assert "Deep thought process here." in history.stdout
    assert "<content>" in history.stdout
    assert "Here is my analysis." in history.stdout


def test_import_with_markdown_headers_template_uses_markdown_not_xml(shared_functional_cli) -> None:
    """Importing with markdown_headers template must wrap reasoning in ## headers, not XML tags.

    Regression: _wrap_reasoning_in_template used hardcoded XML wrapping regardless of template
    type, producing <Thought>...</Thought> instead of ## Thought / ## Content for markdown
    templates.
    """
    cli = shared_functional_cli
    chat_id = "func-import-md-headers"
    payload = json.dumps(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "Simple arithmetic. 2 + 2 = 4.",
            },
        ]
    )
    json_path = cli.write_json_fixture("md-headers-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path, reasoning_template="markdown_headers")
    history = cli.read_history(chat_id)

    assert imported.returncode == 0
    assert "reasoning template: markdown_headers" in imported.stdout

    hist_text = history.stdout
    assert "## Thought" in hist_text, (
        f"Expected '## Thought' header, got XML instead:\n{hist_text[:500]}"
    )
    assert "## Content" in hist_text, (
        f"Expected '## Content' header, got XML instead:\n{hist_text[:500]}"
    )
    assert "<Thought>" not in hist_text, (
        f"Found <Thought> XML tag -- should use ## headers:\n{hist_text[:500]}"
    )
    assert "<Content>" not in hist_text, (
        f"Found <Content> XML tag -- should use ## headers:\n{hist_text[:500]}"
    )
    assert "Simple arithmetic." in hist_text
    assert "The answer is 4." in hist_text


def test_import_with_markdown_headers_prefill_uses_markdown_not_xml(
    shared_functional_cli,
) -> None:
    """Same regression as markdown_headers but for the prefill variant."""
    cli = shared_functional_cli
    chat_id = "func-import-md-prefill"
    payload = json.dumps(
        [
            {"role": "user", "content": "Explain recursion."},
            {
                "role": "assistant",
                "content": "Recursion is when a function calls itself.",
                "reasoning_content": "User wants a short definition of recursion.",
            },
        ]
    )
    json_path = cli.write_json_fixture("md-prefill-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path, reasoning_template="markdown_headers_prefill")
    history = cli.read_history(chat_id)

    assert imported.returncode == 0

    hist_text = history.stdout
    assert "## Thought" in hist_text, (
        f"Expected '## Thought' header, got XML instead:\n{hist_text[:500]}"
    )
    assert "## Content" in hist_text, (
        f"Expected '## Content' header, got XML instead:\n{hist_text[:500]}"
    )
    assert "<Thought>" not in hist_text
    assert "<Content>" not in hist_text
    assert "User wants a short definition" in hist_text
    assert "Recursion is when a function calls itself." in hist_text


def test_import_with_json_template_uses_json_not_xml(shared_functional_cli) -> None:
    """Importing with json template must wrap reasoning as a JSON object, not XML tags.

    Regression: _wrap_reasoning_in_template used hardcoded XML wrapping for all templates,
    producing <thought>...</thought> instead of {"thought": "...", "content": "..."}.
    """
    cli = shared_functional_cli
    chat_id = "func-import-json-tmpl"
    payload = json.dumps(
        [
            {"role": "user", "content": "What is 3+3?"},
            {
                "role": "assistant",
                "content": "The answer is 6.",
                "reasoning_content": "Basic addition.",
            },
        ]
    )
    json_path = cli.write_json_fixture("json-tmpl-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path, reasoning_template="json")
    history = cli.read_history(chat_id)

    assert imported.returncode == 0
    assert "reasoning template: json" in imported.stdout

    hist_text = history.stdout
    assert '"thought"' in hist_text or '"content"' in hist_text, (
        f"Expected JSON keys in history, got XML tags instead:\n{hist_text[:500]}"
    )
    assert "<thought>" not in hist_text.lower(), (
        f"Found XML-style <thought> tag -- json template should produce JSON:\n{hist_text[:500]}"
    )
    assert "<content>" not in hist_text.lower(), (
        f"Found XML-style <content> tag -- json template should produce JSON:\n{hist_text[:500]}"
    )
    assert "Basic addition." in hist_text
    assert "The answer is 6." in hist_text


def test_import_with_json_prefill_template_uses_json_not_xml(shared_functional_cli) -> None:
    """Same regression as json but for the prefill variant."""
    cli = shared_functional_cli
    chat_id = "func-import-json-prefill"
    payload = json.dumps(
        [
            {"role": "user", "content": "What is 5+5?"},
            {
                "role": "assistant",
                "content": "The answer is 10.",
                "reasoning_content": "Simple math again.",
            },
        ]
    )
    json_path = cli.write_json_fixture("json-prefill-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    imported = cli.import_json(chat_id, json_path, reasoning_template="json_prefill")
    history = cli.read_history(chat_id)

    assert imported.returncode == 0

    hist_text = history.stdout
    assert '"thought"' in hist_text or '"content"' in hist_text, (
        f"Expected JSON keys in history, got XML tags instead:\n{hist_text[:500]}"
    )
    assert "<thought>" not in hist_text.lower(), (
        f"Found XML-style <thought> tag -- json_prefill should produce JSON:\n{hist_text[:500]}"
    )
    assert "<content>" not in hist_text.lower(), (
        f"Found XML-style <content> tag -- json_prefill should produce JSON:\n{hist_text[:500]}"
    )
    assert "Simple math again." in hist_text
    assert "The answer is 10." in hist_text


def test_import_with_invalid_reasoning_template_fails(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-import-bad-tmpl"
    payload = json.dumps(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
    )
    json_path = cli.write_json_fixture("bad-tmpl-import.json", payload)

    cli.start_chat(chat_id).require_ok()
    result = cli.import_json(chat_id, json_path, reasoning_template="nonexistent_template")

    assert result.returncode != 0
    assert "unknown reasoning template" in result.output


def test_invalid_and_empty_import_inputs_fail(shared_functional_cli) -> None:
    cli = shared_functional_cli
    chat_id = "func-import-errors"
    invalid_json = cli.write_json_fixture("invalid-import.json", "{not valid json")
    empty_json = cli.write_json_fixture("empty-import.json", "[]")

    cli.start_chat(chat_id).require_ok()

    invalid = cli.import_json(chat_id, invalid_json)
    empty = cli.import_json(chat_id, empty_json)

    assert invalid.returncode != 0
    assert "Error: Invalid JSON:" in invalid.output
    assert empty.returncode != 0
    assert "Error: no messages could be imported" in empty.output
