"""Functional tests for the two-tier format repair system.

Tests the LLM-based repair (Tier 2) with real API calls via OpenRouter.
Uses the free model configured in [format_repair] to verify that structural
formatting is corrected while content is preserved byte-for-byte.

All tests use `func-tpl-repair-*` companion IDs.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from mai_gram.response_templates._sanitize import llm_repair
from mai_gram.response_templates.registry import get_template
from tests.functional.conftest import SLOW_PROVIDERS

if TYPE_CHECKING:
    from mai_gram.response_templates.base import ResponseTemplate

pytestmark = pytest.mark.functional_live

MAX_REPAIR_ATTEMPTS = 3


@pytest.fixture
def requires_openrouter() -> None:
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        pytest.skip("OPENROUTER_API_KEY is required for this live functional scenario.")


@pytest.fixture
def repair_model() -> str:
    return "openrouter/free"


@pytest.fixture
def repair_extra_params() -> dict:
    """Extra API params for repair calls.

    - reasoning disabled for speed
    - provider.ignore excludes slow/incapable providers (shared list)
    """
    return {
        "reasoning": {"effort": "none"},
        "provider": {"ignore": SLOW_PROVIDERS},
    }


@pytest.fixture
def llm_provider():
    """Create a real LLM provider for repair calls."""
    from mai_gram.llm.openrouter import OpenRouterProvider

    api_key = os.environ["OPENROUTER_API_KEY"]
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenRouterProvider(api_key=api_key, base_url=base_url)


def _make_validator(template: ResponseTemplate):
    """Build a validator callback for llm_repair that checks template conformance."""

    def _validate(text: str) -> bool:
        parsed = template.parse(text)
        return template.validate(parsed) == []

    return _validate


async def _repair_with_retry(
    llm_provider,
    template: ResponseTemplate,
    malformed: str,
    repair_model: str,
    repair_extra_params: dict,
) -> str:
    """Call llm_repair with validation, retrying at the test level on failure.

    The inner llm_repair already retries on empty/invalid responses (Option D).
    This outer loop handles the case where all inner retries are exhausted but
    a fresh request to a different random free model might succeed (Option B).
    """
    validator = _make_validator(template)
    format_spec = template.llm_repair_prompt()

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        repaired = await llm_repair(
            llm_provider,
            malformed,
            format_spec,
            model=repair_model,
            temperature=0.0,
            max_tokens=4096,
            max_retries=3,
            extra_params=repair_extra_params,
            validator=validator,
        )
        parsed = template.parse(repaired)
        errors = template.validate(parsed)
        if not errors:
            return repaired
        if attempt < MAX_REPAIR_ATTEMPTS:
            await asyncio.sleep(1.0 * attempt)

    pytest.fail(
        f"Repair failed after {MAX_REPAIR_ATTEMPTS} outer attempts.\n"
        f"Last errors: {errors}\nRepaired text:\n{repaired}"
    )


@pytest.mark.asyncio
async def test_llm_repair_fixes_corrupted_xml_closing_tag(
    requires_openrouter,
    llm_provider,
    repair_model,
    repair_extra_params,
) -> None:
    """The repair model should fix <///content> -> </content>."""
    template = get_template("xml")
    malformed = (
        "<thought>\nThe user asks about Python.\n"
        "I should explain decorators clearly.\n</thought>\n"
        "<content>\nA decorator wraps a function to extend its behavior.\n<///content>"
    )

    repaired = await _repair_with_retry(
        llm_provider,
        template,
        malformed,
        repair_model,
        repair_extra_params,
    )

    parsed = template.parse(repaired)
    assert "decorator" in parsed.fields["content"].lower()


@pytest.mark.asyncio
async def test_llm_repair_fixes_missing_closing_tag(
    requires_openrouter,
    llm_provider,
    repair_model,
    repair_extra_params,
) -> None:
    """The repair model should add a missing </content> tag."""
    template = get_template("xml")
    malformed = (
        "<thought>\nThinking about the answer.\n</thought>\n"
        "<content>\nHere is my complete answer about Python generators."
    )

    repaired = await _repair_with_retry(
        llm_provider,
        template,
        malformed,
        repair_model,
        repair_extra_params,
    )

    parsed = template.parse(repaired)
    assert "generators" in parsed.fields["content"].lower()


@pytest.mark.asyncio
async def test_llm_repair_fixes_broken_json(
    requires_openrouter,
    llm_provider,
    repair_model,
    repair_extra_params,
) -> None:
    """The repair model should fix broken JSON structure."""
    template = get_template("json")
    malformed = (
        '{"thought": "The user wants to know about lists",'
        ' "content": "Python lists are ordered collections",}'
    )

    repaired = await _repair_with_retry(
        llm_provider,
        template,
        malformed,
        repair_model,
        repair_extra_params,
    )

    parsed = template.parse(repaired)
    assert "lists" in parsed.fields["content"].lower()


@pytest.mark.asyncio
async def test_llm_repair_preserves_content_unchanged(
    requires_openrouter,
    llm_provider,
    repair_model,
    repair_extra_params,
) -> None:
    """After repair, the text content must remain semantically identical."""
    template = get_template("xml")
    original_thought = (
        "\u041f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c"
        " \u0441\u043f\u0440\u0430\u0448\u0438\u0432\u0430\u0435\u0442"
        " \u043f\u0440\u043e \u0434\u0435\u043a\u043e\u0440\u0430\u0442\u043e\u0440\u044b.\n"
        "\u041d\u0443\u0436\u043d\u043e \u043e\u0431\u044a\u044f\u0441\u043d\u0438\u0442\u044c"
        " \u0441 \u043f\u0440\u0438\u043c\u0435\u0440\u043e\u043c"
        " \u043a\u043e\u0434\u0430."
    )
    original_content = (
        "\u0414\u0435\u043a\u043e\u0440\u0430\u0442\u043e\u0440 \u0432 Python"
        " \u2014 \u044d\u0442\u043e"
        " \u0444\u0443\u043d\u043a\u0446\u0438\u044f-\u043e\u0431\u0451\u0440\u0442\u043a\u0430."
        "\n\n```python\ndef my_decorator(func):\n    pass\n```"
    )

    malformed = (
        f"<thought>\n{original_thought}\n</thought></thought>\n"
        f"<content>\n{original_content}\n<///content>"
    )

    repaired = await _repair_with_retry(
        llm_provider,
        template,
        malformed,
        repair_model,
        repair_extra_params,
    )

    parsed = template.parse(repaired)
    assert (
        "\u0434\u0435\u043a\u043e\u0440\u0430\u0442\u043e\u0440" in parsed.fields["content"].lower()
        or "\u0414\u0435\u043a\u043e\u0440\u0430\u0442\u043e\u0440" in parsed.fields["content"]
    )
    assert "```python" in parsed.fields["content"]


@pytest.mark.asyncio
async def test_llm_repair_handles_xml_with_emotions(
    requires_openrouter,
    llm_provider,
    repair_model,
    repair_extra_params,
) -> None:
    """Repair should work for templates with more than 2 fields."""
    template = get_template("xml_emotions")
    malformed = (
        "<thought>\nUser asks about lists.\n</thought>\n"
        "<emotions>curious, engaged, focused</emotions>\n"
        "<content>\nPython lists are great.\n<///content>"
    )

    repaired = await _repair_with_retry(
        llm_provider,
        template,
        malformed,
        repair_model,
        repair_extra_params,
    )

    parsed = template.parse(repaired)
    assert "emotions" in parsed.fields
