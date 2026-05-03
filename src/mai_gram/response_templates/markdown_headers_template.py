"""Markdown headers response template -- thought + content under ## headers."""

from __future__ import annotations

import re

from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
    TemplateExample,
)
from mai_gram.response_templates.registry import register_template

_HEADER_RE = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)


def _extract_markdown_sections(
    raw_text: str,
    field_names: list[str],
) -> dict[str, str]:
    """Split text by ## headers and extract named sections."""
    field_names_lower = {n.lower(): n for n in field_names}
    fields: dict[str, str] = {}

    splits = _HEADER_RE.split(raw_text)
    # splits alternates: [preamble, header1, body1, header2, body2, ...]
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
    """Structured template using ## Thought and ## Content markdown headers."""

    @property
    def name(self) -> str:
        return "markdown_headers"

    @property
    def description(self) -> str:
        return "Markdown: ## Thought + ## Content"

    def get_fields(self) -> list[FieldDescriptor]:
        return [
            FieldDescriptor(
                name="Thought",
                required=True,
                display_label="\U0001f4ad Thought",
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
        field_names = [f.name for f in self.get_fields()]
        headers_spec = "\n".join(f"## {n}\n..." for n in field_names)
        order_note = " -> ".join(f"## {n}" for n in field_names)
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using markdown level-2 headers. "
            "Do NOT output any text outside these sections.\n\n"
            f"{headers_spec}\n\n"
            f"Section order is strict: {order_note}\n"
            "The Thought section contains your internal reasoning (at least 2 paragraphs). "
            "The Content section contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        return [
            TemplateExample(
                text=(
                    "## Thought\nThe user is asking about decorators. "
                    "I should explain the concept clearly.\n\n"
                    "Decorators are a form of metaprogramming.\n\n"
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
