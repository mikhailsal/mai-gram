"""JSON response template -- thought + content as JSON object."""

from __future__ import annotations

import json
import re

from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
    TemplateExample,
)
from mai_gram.response_templates.registry import register_template

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw_text: str) -> dict[str, str] | None:
    """Try to extract a JSON object from raw text.

    Handles both bare JSON and fenced code blocks.
    """
    block_match = _JSON_BLOCK_RE.search(raw_text)
    candidate = block_match.group(1) if block_match else None

    if candidate is None:
        bare_match = _BARE_JSON_RE.search(raw_text)
        candidate = bare_match.group(0) if bare_match else None

    if candidate is None:
        return None

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    return {k: str(v) for k, v in data.items()}


class JsonTemplate(ResponseTemplate):
    """Structured JSON template with "thought" and "content" keys."""

    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return 'JSON object: {"thought", "content"}'

    def get_fields(self) -> list[FieldDescriptor]:
        return [
            FieldDescriptor(
                name="thought",
                required=True,
                display_label="\U0001f4ad Thought",
                display_tag="blockquote",
                expandable=True,
                user_can_hide=True,
                order=0,
            ),
            FieldDescriptor(
                name="content",
                required=True,
                display_tag="none",
                order=1,
            ),
        ]

    def format_instruction(self) -> str:
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response as a single JSON object with "
            "exactly these keys (no extra keys):\n\n"
            '{"thought": "<your reasoning, at least 2 paragraphs>", '
            '"content": "<your response to the human>"}\n\n'
            "Do NOT wrap the JSON in a code fence. Output ONLY the raw JSON object. "
            "Escape newlines within values as \\n."
        )

    def examples(self) -> list[TemplateExample]:
        return [
            TemplateExample(
                text=(
                    '{"thought": "The user asks about decorators. I should explain '
                    "the concept clearly.\\n\\nDecorators wrap functions to extend "
                    'behavior.", "content": "A Python decorator is a function that '
                    'wraps another function..."}'
                ),
                is_positive=True,
            ),
            TemplateExample(
                text="Sure, here is the answer...",
                is_positive=False,
            ),
        ]

    def parse(self, raw_text: str) -> ParsedResponse:
        data = _extract_json(raw_text)
        if data is None:
            return ParsedResponse(fields={}, raw=raw_text)
        return ParsedResponse(fields=data, raw=raw_text)

    def validate(self, parsed: ParsedResponse) -> list[str]:
        errors: list[str] = []
        if not parsed.fields:
            errors.append("Could not parse a valid JSON object from the response")
            return errors
        for f in self.get_fields():
            if f.required and f.name not in parsed.fields:
                errors.append(f'Missing required key "{f.name}"')
            elif f.required and not parsed.fields.get(f.name, "").strip():
                errors.append(f'Empty value for key "{f.name}"')
        return errors


register_template(JsonTemplate())
