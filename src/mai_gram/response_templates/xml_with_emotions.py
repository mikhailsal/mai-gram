"""XML with emotions -- inherits from XmlTemplate, adds an emotions field.

Demonstrates template inheritance for DRY composition of fields.
Supports additional user-configurable parameters:
- emotions_field: tag name for emotions (default "emotions")
- num_emotions: number of emotions to express (default 3)
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.base import (
    FieldDescriptor,
    TemplateExample,
    TemplateParam,
)
from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_template import XmlTemplate, _generate_reasoning_example

_EMOTION_POOL = [
    "curious",
    "enthusiastic about teaching",
    "slightly amused",
    "thoughtful",
    "empathetic",
    "inspired",
    "focused",
    "warmly engaged",
    "pleasantly surprised",
    "contemplative",
    "playfully intrigued",
    "genuinely interested",
]


def _generate_emotions_example(num_emotions: int) -> str:
    """Build an example emotions string with exactly *num_emotions* items."""
    items = _EMOTION_POOL[:num_emotions]
    while len(items) < num_emotions:
        items.append(f"emotion-{len(items) + 1}")
    return ", ".join(items)


class XmlWithEmotionsTemplate(XmlTemplate):
    """Extended XML template: <reasoning> + <emotions> + <content>.

    Inherits parsing, validation, and rendering logic from XmlTemplate.
    Adds an emotions field between reasoning and content.
    """

    def __init__(
        self,
        *,
        reasoning_field: str = "thought",
        num_reasoning_paragraphs: int = 2,
        emotions_field: str = "emotions",
        num_emotions: int = 3,
    ) -> None:
        super().__init__(
            reasoning_field=reasoning_field,
            num_reasoning_paragraphs=num_reasoning_paragraphs,
        )
        self._emotions_field = emotions_field
        self._num_emotions = num_emotions

    @property
    def name(self) -> str:
        return "xml_emotions"

    @property
    def description(self) -> str:
        rf = self._reasoning_field
        ef = self._emotions_field
        return f"XML tags: <{rf}> + <{ef}> + <content>"

    def get_params(self) -> list[TemplateParam]:
        base = super().get_params()
        return [
            *base,
            TemplateParam(
                key="emotions_field",
                label="Emotions field name",
                param_type="str",
                default="emotions",
                description="XML tag name for the emotions block",
                suggestions=["emotions", "feelings", "mood", "affect", "sentiment"],
            ),
            TemplateParam(
                key="num_emotions",
                label="Number of emotions",
                param_type="int",
                default=3,
                description="How many emotions to express per response",
                min_value=1,
                max_value=12,
            ),
        ]

    def get_effective_params(self) -> dict[str, Any]:
        base = super().get_effective_params()
        return {
            **base,
            "emotions_field": self._emotions_field,
            "num_emotions": self._num_emotions,
        }

    def _build_with_params(self, params: dict[str, Any]) -> XmlWithEmotionsTemplate:
        return XmlWithEmotionsTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
            emotions_field=params["emotions_field"],
            num_emotions=params["num_emotions"],
        )

    def get_fields(self) -> list[FieldDescriptor]:
        base_fields = super().get_fields()
        ef = self._emotions_field
        emotions_descriptor = FieldDescriptor(
            name=ef,
            required=True,
            display_label=f"\U0001f3ad {ef.replace('_', ' ').title()}",
            display_tag="blockquote",
            expandable=False,
            user_can_hide=True,
            order=1,
        )
        reasoning = [f for f in base_fields if f.name == self._reasoning_field]
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
        return [*reasoning, emotions_descriptor, *content_reordered]

    def format_instruction(self) -> str:
        rf = self._reasoning_field
        ef = self._emotions_field
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
            f"The <{ef}> tag contains exactly {self._num_emotions} "
            f"emotion{'s' if self._num_emotions != 1 else ''} describing "
            "your current emotional state, comma-separated. "
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        ef = self._emotions_field
        reasoning = _generate_reasoning_example(self._num_reasoning_paragraphs)
        emotions = _generate_emotions_example(self._num_emotions)
        return [
            TemplateExample(
                text=(
                    f"<{rf}>\n{reasoning}\n</{rf}>\n"
                    f"<{ef}>{emotions}</{ef}>\n"
                    "<content>\nA Python decorator is a function that wraps "
                    "another function...\n</content>"
                ),
                is_positive=True,
            ),
        ]


register_template(XmlWithEmotionsTemplate())
