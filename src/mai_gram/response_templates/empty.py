"""Empty template -- preserves current behavior (no format constraints)."""

from __future__ import annotations

from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
)
from mai_gram.response_templates.registry import register_template


class EmptyTemplate(ResponseTemplate):
    """No-op template that passes raw LLM output through unchanged.

    This is the default for all existing chats and new chats that don't
    select a structured format.
    """

    @property
    def name(self) -> str:
        return "empty"

    @property
    def description(self) -> str:
        return "No format constraints (default)"

    def get_fields(self) -> list[FieldDescriptor]:
        return [
            FieldDescriptor(
                name="content",
                required=True,
                display_tag="none",
                order=0,
            ),
        ]

    def format_instruction(self) -> str:
        return ""

    def parse(self, raw_text: str) -> ParsedResponse:
        return ParsedResponse(fields={"content": raw_text}, raw=raw_text)

    def validate(self, parsed: ParsedResponse) -> list[str]:
        return []

    def render_field_html(
        self,
        field_name: str,
        content: str,
        *,
        expandable: bool = False,
    ) -> str:
        from mai_gram.core.md_to_telegram import markdown_to_html

        return markdown_to_html(content)


register_template(EmptyTemplate())
