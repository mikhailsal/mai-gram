"""Functional tests for the response template system.

All tests use `func-` prefixed test companion IDs and hit real LLM APIs.
"""

from __future__ import annotations

import pytest

from tests.functional.helpers.parsing import extract_last_response_body

pytestmark = pytest.mark.functional


def test_empty_template_preserves_current_behavior(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """Starting a chat with the empty template should behave exactly like before."""
    functional_cli.start_chat("func-tpl-empty", template="empty").require_ok()

    result = functional_cli.send_message_with_live_retry(
        "func-tpl-empty",
        "Reply with exactly the word PINEAPPLE.",
    )
    assert result.returncode == 0
    body = extract_last_response_body(result.stdout)
    assert "PINEAPPLE" in body.upper()


def test_xml_template_produces_structured_response(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """With the XML template, the model should produce <thought> and <content> tags."""
    functional_cli.start_chat("func-tpl-xml", template="xml").require_ok()

    result = functional_cli.send_message_with_live_retry(
        "func-tpl-xml",
        "What is 2+2? Answer briefly.",
    )
    assert result.returncode == 0

    history = functional_cli.read_history("func-tpl-xml")
    assert history.returncode == 0
    full_output = history.stdout.lower()
    assert "<thought>" in full_output and "<content>" in full_output, (
        f"Expected XML tags in stored history output, got:\n{history.stdout[:500]}"
    )


def test_prompt_preview_includes_template_instructions(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """The prompt preview should contain template format instructions."""
    functional_cli.start_chat("func-tpl-preview", template="xml").require_ok()
    functional_cli.send_message_with_live_retry(
        "func-tpl-preview",
        "Hello",
    ).require_ok()

    preview = functional_cli.show_prompt("func-tpl-preview")
    assert preview.returncode == 0
    assert "RESPONSE FORMAT" in preview.stdout
    assert "<thought>" in preview.stdout
    assert "<content>" in preview.stdout


def test_template_field_toggle(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """The /toggle command should hide/show template fields."""
    functional_cli.start_chat("func-tpl-toggle", template="xml").require_ok()

    result = functional_cli.run_command("func-tpl-toggle", "toggle", args="thought")
    assert result.returncode == 0
    assert "HIDDEN" in result.stdout

    result = functional_cli.run_command("func-tpl-toggle", "toggle", args="thought")
    assert result.returncode == 0
    assert "VISIBLE" in result.stdout


def test_toggle_rejects_non_toggleable_field(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """Toggling a non-toggleable field should produce an error message."""
    functional_cli.start_chat("func-tpl-toggle-err", template="xml").require_ok()

    result = functional_cli.run_command("func-tpl-toggle-err", "toggle", args="content")
    assert result.returncode == 0
    assert "non-toggleable" in result.stdout.lower() or "unknown" in result.stdout.lower()


def test_toggle_with_empty_template_has_no_fields(
    functional_cli,
    requires_openrouter_api_key,
) -> None:
    """The empty template should have no toggleable fields."""
    functional_cli.start_chat("func-tpl-toggle-empty", template="empty").require_ok()

    result = functional_cli.run_command("func-tpl-toggle-empty", "toggle")
    assert result.returncode == 0
    assert "no toggleable fields" in result.stdout.lower()
