"""JSON response template -- reasoning + content as JSON object.

Supports user-configurable parameters:
- reasoning_field: JSON key for internal reasoning (default "thought")
- num_reasoning_paragraphs: minimum paragraphs in reasoning (default 2)
"""

from __future__ import annotations

import json
import re
from typing import Any

from mai_gram.response_templates._sanitize import sanitize_json_structure
from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
    StreamingParseResult,
    TemplateExample,
    TemplateParam,
)
from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_template import _generate_reasoning_example

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_KEY_VALUE_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _key_re(field_name: str) -> re.Pattern[str]:
    r"""Compile and cache a regex that finds ``"field_name"\s*:\s*"``."""
    if field_name not in _KEY_VALUE_RE_CACHE:
        escaped = re.escape(field_name)
        _KEY_VALUE_RE_CACHE[field_name] = re.compile(rf'"{escaped}"\s*:\s*"', re.DOTALL)
    return _KEY_VALUE_RE_CACHE[field_name]


def _extract_json_field_streaming(
    json_text: str,
    field_name: str,
) -> tuple[str | None, int, bool]:
    """Extract a JSON string field value from partially-received JSON text.

    Returns ``(value, end_position, is_complete)``.  When the closing
    quote hasn't arrived yet, *value* contains everything received so
    far and *is_complete* is ``False``.  Returns ``(None, 0, False)``
    when the key hasn't appeared yet.
    """
    m = _key_re(field_name).search(json_text)
    if m is None:
        return None, 0, False

    value_start = m.end()
    i = value_start
    n = len(json_text)
    parts: list[str] = []

    while i < n:
        ch = json_text[i]
        if ch == "\\" and i + 1 < n:
            esc = json_text[i + 1]
            if esc == "n":
                parts.append("\n")
            elif esc == "t":
                parts.append("\t")
            elif esc == "r":
                parts.append("\r")
            elif esc == '"':
                parts.append('"')
            elif esc == "\\":
                parts.append("\\")
            else:
                parts.append(ch + esc)
            i += 2
            continue
        if ch == '"':
            return "".join(parts), i + 1, True
        parts.append(ch)
        i += 1

    return "".join(parts), n, False


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
    """Structured JSON template with configurable reasoning + content keys."""

    def __init__(
        self,
        *,
        reasoning_field: str = "thought",
        num_reasoning_paragraphs: int = 2,
    ) -> None:
        self._reasoning_field = reasoning_field
        self._num_reasoning_paragraphs = num_reasoning_paragraphs

    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return f'JSON object: {{"{self._reasoning_field}", "content"}}'

    @property
    def group(self) -> str:
        return "json"

    def get_params(self) -> list[TemplateParam]:
        return [
            TemplateParam(
                key="reasoning_field",
                label="Reasoning field name",
                param_type="str",
                default="thought",
                description="JSON key for internal reasoning",
                suggestions=["thought", "think", "scratchpad", "reasoning", "reflection"],
            ),
            TemplateParam(
                key="num_reasoning_paragraphs",
                label="Reasoning paragraphs",
                param_type="int",
                default=2,
                description="Minimum number of reasoning paragraphs",
                min_value=1,
                max_value=8,
            ),
        ]

    def get_effective_params(self) -> dict[str, Any]:
        return {
            "reasoning_field": self._reasoning_field,
            "num_reasoning_paragraphs": self._num_reasoning_paragraphs,
        }

    def _build_with_params(self, params: dict[str, Any]) -> JsonTemplate:
        return JsonTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
        )

    def get_fields(self) -> list[FieldDescriptor]:
        rf = self._reasoning_field
        return [
            FieldDescriptor(
                name=rf,
                required=True,
                display_label=f"\U0001f4ad {rf.replace('_', ' ').title()}",
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

    def wrap_reasoning_for_import(self, reasoning: str, content: str) -> str:
        rf = self._reasoning_field
        return json.dumps({rf: reasoning, "content": content}, ensure_ascii=False)

    def format_instruction(self) -> str:
        rf = self._reasoning_field
        n = self._num_reasoning_paragraphs
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response as a single JSON object with "
            "exactly these keys (no extra keys):\n\n"
            f'{{"{rf}": "<your reasoning, at least {n} paragraph'
            f"{'s' if n != 1 else ''}"
            f'>", "content": "<your response to the human>"}}\n\n'
            "Do NOT wrap the JSON in a code fence. Output ONLY the raw JSON object. "
            "Escape newlines within values as \\n."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        reasoning = _generate_reasoning_example(self._num_reasoning_paragraphs)
        reasoning_escaped = reasoning.replace("\n", "\\n")
        return [
            TemplateExample(
                text=(
                    f'{{"{rf}": "{reasoning_escaped}", '
                    '"content": "A Python decorator is a function that '
                    'wraps another function..."}'
                ),
                is_positive=True,
            ),
            TemplateExample(
                text="Sure, here is the answer...",
                is_positive=False,
            ),
        ]

    def parse_streaming(self, accumulated_text: str) -> StreamingParseResult:
        field_names = [f.name for f in sorted(self.get_fields(), key=lambda f: f.order)]
        completed: dict[str, str] = {}
        active_field: str | None = None
        active_content = ""

        brace_pos = accumulated_text.find("{")
        if brace_pos == -1:
            return StreamingParseResult(
                completed_fields={},
                active_field=None,
                active_content=accumulated_text,
                preamble=accumulated_text,
            )

        preamble = accumulated_text[:brace_pos].strip()
        json_part = accumulated_text[brace_pos:]

        for name in field_names:
            value, _end_pos, is_complete = _extract_json_field_streaming(json_part, name)
            if value is None:
                continue
            if is_complete:
                completed[name] = value
            else:
                active_field = name
                active_content = value
                break

        if not active_field and not completed:
            return StreamingParseResult(
                completed_fields={},
                active_field=None,
                active_content=accumulated_text,
                preamble=preamble,
            )

        return StreamingParseResult(
            completed_fields=completed,
            active_field=active_field,
            active_content=active_content,
            preamble=preamble,
        )

    def sanitize(self, raw_text: str) -> str:
        return sanitize_json_structure(raw_text)

    def llm_repair_prompt(self) -> str:
        keys = [f.name for f in self.get_fields()]
        return f"JSON object with exactly these keys: {json.dumps(keys)}"

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
