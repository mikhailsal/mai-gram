"""XML response template -- thought + content in XML tags."""

from __future__ import annotations

import re

from mai_gram.response_templates.base import (
    FieldDescriptor,
    ParsedResponse,
    ResponseTemplate,
    TemplateExample,
)
from mai_gram.response_templates.registry import register_template

_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _extract_xml_fields(
    raw_text: str,
    field_names: list[str],
) -> dict[str, str]:
    """Extract content from XML-like tags, preserving order of first occurrence."""
    fields: dict[str, str] = {}
    for match in _TAG_RE.finditer(raw_text):
        tag_name = match.group(1)
        if tag_name in field_names and tag_name not in fields:
            fields[tag_name] = match.group(2).strip()
    return fields


class XmlTemplate(ResponseTemplate):
    """Structured XML template with <thought> and <content> tags.

    The model is instructed to wrap its reasoning in <thought> and its
    user-facing reply in <content>. Order matters.
    """

    @property
    def name(self) -> str:
        return "xml"

    @property
    def description(self) -> str:
        return "XML tags: <thought> + <content>"

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
        field_names = [f.name for f in self.get_fields()]
        tags_spec = "\n".join(f"<{n}>...</{n}>" for n in field_names)
        order_note = " -> ".join(f"<{n}>" for n in field_names)
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using the following XML tags. "
            "Do NOT output any text outside these tags.\n\n"
            f"{tags_spec}\n\n"
            f"Tag order is strict: {order_note}\n"
            "The <thought> tag contains your internal reasoning (at least 2 paragraphs). "
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        return [
            TemplateExample(
                text=(
                    "<thought>\nThe user is asking about Python decorators. "
                    "I should explain the concept clearly with a practical example.\n\n"
                    "Decorators are a form of metaprogramming. I'll start with the "
                    "basic pattern and then show a real-world use case.\n</thought>\n"
                    "<content>\nA Python decorator is a function that wraps another "
                    "function to extend its behavior...\n</content>"
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
        fields = _extract_xml_fields(raw_text, field_names)
        return ParsedResponse(fields=fields, raw=raw_text)

    def validate(self, parsed: ParsedResponse) -> list[str]:
        errors: list[str] = []
        for f in self.get_fields():
            if f.required and f.name not in parsed.fields:
                errors.append(f"Missing required <{f.name}> tag")
            elif f.required and not parsed.fields.get(f.name, "").strip():
                errors.append(f"Empty <{f.name}> tag")
        return errors

    def render_field_html(
        self,
        field_name: str,
        content: str,
        *,
        expandable: bool = False,
    ) -> str:
        from mai_gram.core.md_to_telegram import markdown_to_html

        descriptor = self._field_by_name(field_name)
        if descriptor is None or descriptor.display_tag == "none":
            return markdown_to_html(content.strip())

        inner_html = markdown_to_html(content.strip())
        tag = "blockquote expandable" if expandable else "blockquote"
        close_tag = tag.split()[0]
        return f"<{tag}>{descriptor.label}\n{inner_html}</{close_tag}>"


register_template(XmlTemplate())
