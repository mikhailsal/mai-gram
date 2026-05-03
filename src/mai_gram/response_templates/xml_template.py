"""XML response template -- reasoning + content in XML tags.

Supports user-configurable parameters:
- reasoning_field: tag name for the reasoning block (default "thought")
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

_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

_REASONING_PARAGRAPHS = [
    (
        "The user is asking about Python decorators. "
        "I should explain the concept clearly with a practical example."
    ),
    (
        "Decorators are a form of metaprogramming. I'll start with the "
        "basic pattern and then show a real-world use case."
    ),
    "I should mention common pitfalls like losing function metadata without functools.wraps.",
    "It would also be helpful to compare class-based and function-based decorators.",
    "Let me include a concise code snippet so the explanation is concrete and actionable.",
    "I'll wrap up by linking decorators to broader design patterns like the Adapter pattern.",
    "Finally, I could suggest further reading for those who want to go deeper.",
    "Let me also consider how decorators interact with async functions in modern Python.",
]


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


def _generate_reasoning_example(num_paragraphs: int) -> str:
    """Build example reasoning text with exactly *num_paragraphs* paragraphs."""
    paras = _REASONING_PARAGRAPHS[:num_paragraphs]
    while len(paras) < num_paragraphs:
        paras.append(f"(Additional reasoning paragraph {len(paras) + 1}.)")
    return "\n\n".join(paras)


class XmlTemplate(ResponseTemplate):
    """Structured XML template with configurable reasoning + content tags.

    The model is instructed to wrap its reasoning in a configurable tag
    (default ``<thought>``) and its user-facing reply in ``<content>``.
    """

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
        return "xml"

    @property
    def description(self) -> str:
        return f"XML tags: <{self._reasoning_field}> + <content>"

    def get_params(self) -> list[TemplateParam]:
        return [
            TemplateParam(
                key="reasoning_field",
                label="Reasoning field name",
                param_type="str",
                default="thought",
                description="XML tag name for internal reasoning",
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

    def _build_with_params(self, params: dict[str, Any]) -> XmlTemplate:
        return XmlTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
        )

    def get_fields(self) -> list[FieldDescriptor]:
        return [
            FieldDescriptor(
                name=self._reasoning_field,
                required=True,
                display_label=f"\U0001f4ad {self._reasoning_field.replace('_', ' ').title()}",
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
        rf = self._reasoning_field
        field_names = [f.name for f in self.get_fields()]
        tags_spec = "\n".join(f"<{n}>...</{n}>" for n in field_names)
        order_note = " -> ".join(f"<{n}>" for n in field_names)
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using the following XML tags. "
            "Do NOT output any text outside these tags.\n\n"
            f"{tags_spec}\n\n"
            f"Tag order is strict: {order_note}\n"
            f"The <{rf}> tag contains your internal reasoning "
            f"(at least {self._num_reasoning_paragraphs} paragraph"
            f"{'s' if self._num_reasoning_paragraphs != 1 else ''}). "
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        reasoning = _generate_reasoning_example(self._num_reasoning_paragraphs)
        return [
            TemplateExample(
                text=(
                    f"<{rf}>\n{reasoning}\n</{rf}>\n"
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
