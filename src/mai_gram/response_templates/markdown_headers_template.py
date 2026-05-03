"""Markdown headers response template -- reasoning + content under ## headers.

Supports user-configurable parameters:
- reasoning_field: section header for reasoning (default "Thought")
- num_reasoning_paragraphs: minimum paragraphs in reasoning (default 2)
"""

from __future__ import annotations

import re
from typing import Any

from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
    TemplateExample,
    TemplateParam,
)
from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_template import _generate_reasoning_example

_HEADER_RE = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)


def _extract_markdown_sections(
    raw_text: str,
    field_names: list[str],
) -> dict[str, str]:
    """Split text by ## headers and extract named sections."""
    field_names_lower = {n.lower(): n for n in field_names}
    fields: dict[str, str] = {}

    splits = _HEADER_RE.split(raw_text)
    i = 1
    while i < len(splits) - 1:
        header_raw = splits[i].strip().lower()
        body = splits[i + 1].strip()
        original_name = field_names_lower.get(header_raw)
        if original_name and original_name not in fields:
            fields[original_name] = body
        i += 2

    return fields


class MarkdownHeadersTemplate(ResponseTemplate):
    """Structured template using ## headers with configurable reasoning field."""

    def __init__(
        self,
        *,
        reasoning_field: str = "Thought",
        num_reasoning_paragraphs: int = 2,
    ) -> None:
        self._reasoning_field = reasoning_field
        self._num_reasoning_paragraphs = num_reasoning_paragraphs

    @property
    def name(self) -> str:
        return "markdown_headers"

    @property
    def description(self) -> str:
        return f"Markdown: ## {self._reasoning_field} + ## Content"

    def get_params(self) -> list[TemplateParam]:
        return [
            TemplateParam(
                key="reasoning_field",
                label="Reasoning section name",
                param_type="str",
                default="Thought",
                description="Markdown header name for the reasoning section",
                suggestions=["Thought", "Think", "Scratchpad", "Reasoning", "Reflection"],
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

    def _build_with_params(self, params: dict[str, Any]) -> MarkdownHeadersTemplate:
        return MarkdownHeadersTemplate(
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
                name="Content",
                required=True,
                display_tag="none",
                order=1,
            ),
        ]

    def content_field_name(self) -> str:
        return "Content"

    def format_instruction(self) -> str:
        rf = self._reasoning_field
        n = self._num_reasoning_paragraphs
        field_names = [f.name for f in self.get_fields()]
        headers_spec = "\n".join(f"## {name}\n..." for name in field_names)
        order_note = " -> ".join(f"## {name}" for name in field_names)
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using markdown level-2 headers. "
            "Do NOT output any text outside these sections.\n\n"
            f"{headers_spec}\n\n"
            f"Section order is strict: {order_note}\n"
            f"The {rf} section contains your internal reasoning "
            f"(at least {n} paragraph{'s' if n != 1 else ''}). "
            "The Content section contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        reasoning = _generate_reasoning_example(self._num_reasoning_paragraphs)
        return [
            TemplateExample(
                text=(
                    f"## {rf}\n{reasoning}\n\n"
                    "## Content\nA Python decorator is a function that wraps "
                    "another function..."
                ),
                is_positive=True,
            ),
            TemplateExample(
                text="Sure, here is the answer...",
                is_positive=False,
            ),
        ]

    def parse(self, raw_text: str) -> ParsedResponse:
        field_names = [f.name for f in self.get_fields()]
        fields = _extract_markdown_sections(raw_text, field_names)
        return ParsedResponse(fields=fields, raw=raw_text)

    def validate(self, parsed: ParsedResponse) -> list[str]:
        errors: list[str] = []
        for f in self.get_fields():
            if f.required and f.name not in parsed.fields:
                errors.append(f'Missing required section "## {f.name}"')
            elif f.required and not parsed.fields.get(f.name, "").strip():
                errors.append(f'Empty section "## {f.name}"')
        return errors


register_template(MarkdownHeadersTemplate())
