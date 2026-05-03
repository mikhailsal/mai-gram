"""XML with emotions -- inherits from XmlTemplate, adds an emotions field.

Demonstrates template inheritance for DRY composition of fields.
"""

from __future__ import annotations

from mai_gram.response_templates.base import (
    FieldDescriptor,
    TemplateExample,
)
from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_template import XmlTemplate


class XmlWithEmotionsTemplate(XmlTemplate):
    """Extended XML template: <thought> + <emotions> + <content>.

    Inherits parsing, validation, and rendering logic from XmlTemplate.
    Only adds the `emotions` field between thought and content.
    """

    @property
    def name(self) -> str:
        return "xml_emotions"

    @property
    def description(self) -> str:
        return "XML tags: <thought> + <emotions> + <content>"

    def get_fields(self) -> list[FieldDescriptor]:
        base_fields = super().get_fields()
        emotions_field = FieldDescriptor(
            name="emotions",
            required=True,
            display_label="\U0001f3ad Emotions",
            display_tag="blockquote",
            expandable=False,
            user_can_hide=True,
            order=1,
        )
        thought = [f for f in base_fields if f.name == "thought"]
        content = [f for f in base_fields if f.name == "content"]
        content_reordered = [
            FieldDescriptor(
                name=f.name,
                required=f.required,
                display_label=f.display_label,
                display_tag=f.display_tag,
                expandable=f.expandable,
                user_can_hide=f.user_can_hide,
                order=2,
            )
            for f in content
        ]
        return [*thought, emotions_field, *content_reordered]

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
            "The <emotions> tag contains a brief description of your current emotional state. "
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        return [
            TemplateExample(
                text=(
                    "<thought>\nThe user is asking about Python decorators. "
                    "I should explain the concept clearly.\n\n"
                    "Decorators are a form of metaprogramming.\n</thought>\n"
                    "<emotions>curious, enthusiastic about teaching</emotions>\n"
                    "<content>\nA Python decorator is a function that wraps "
                    "another function...\n</content>"
                ),
                is_positive=True,
            ),
        ]


register_template(XmlWithEmotionsTemplate())
