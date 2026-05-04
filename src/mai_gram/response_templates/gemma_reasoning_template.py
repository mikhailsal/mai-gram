"""Gemma-4-style structured reasoning template.

Produces reasoning in the native Gemma-4 analytical format: a sequence of
blank-line-separated blocks using indented bullet points.  Each block is an
unlabelled cognitive phase -- the model decides what each block covers
(situation parsing, context recall, analysis, option drafting, style checks)
based on the problem at hand.

The format is modelled on 104 real ``reasoning_content`` messages produced by
``google/gemma-4-31b-it``.  Key observations from that corpus:

- **No explicit section labels** -- 100% of blocks are content-driven, not
  headed with bold or italic labels.
- **4-6 blocks** on average (median 6, range 3-27), separated by blank lines.
- **Indented bullets** (``    *   ``) are the dominant structural element
  (80/104 examples).
- **First block** typically parses the user's input (intent, tone, context).
- **Middle blocks** recall constraints, analyse the problem, and/or draft
  candidate responses with inline option labels like ``*Option 1 (Too polite):*``.
- **Last block** is nearly always a short style/tone checklist.
- Some blocks use numbered lists, dashes, or plain paragraphs instead of
  bullets -- the model adapts the micro-format to the content.

Supports user-configurable parameters:

- ``num_reasoning_blocks``: minimum blank-line-separated blocks (default 4)
- ``reasoning_field``: XML tag name wrapping the reasoning (default "reasoning")
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


_EXAMPLE_BLOCKS: list[list[str]] = [
    # Block 0 -- parse user input
    [
        "*   User asks about Python decorators. Tone is curious, practical.",
        "    *   Context: second exchange, the user prefers code examples over theory.",
        '    *   Goal: give a concrete, useful explanation without being a "helpful assistant."',
    ],
    # Block 1 -- recall constraints / identity
    [
        "    *   I should be direct and concrete, avoid excessive metaphors.",
        "    *   Keep it conversational, 1-3 paragraphs.",
        "    *   If the question is straightforward, just answer it honestly.",
    ],
    # Block 2 -- deeper analysis
    [
        "    *   Decorators are a form of metaprogramming -- a wrapper function.",
        "    *   Common pitfall: losing function metadata without functools.wraps.",
        "    *   Class-based vs function-based decorators are worth mentioning"
        " but might be too much for a first answer.",
    ],
    # Block 3 -- draft candidate responses
    [
        "    *   *Option 1 (Theory-heavy):* Start with the abstract concept."
        " Too academic for this user.",
        "    *   *Option 2 (Code-first):* Lead with a code snippet, explain after. More engaging.",
        "    *   *Option 3 (Balanced):* Brief concept, then code, then common"
        " gotchas. Best fit for the tone.",
    ],
    # Block 4 -- response plan / action items
    [
        "    *   Lead with a one-sentence definition.",
        "    *   Show a concrete decorator example with functools.wraps.",
        "    *   Mention the metadata pitfall as a practical tip.",
    ],
    # Block 5 -- style checklist
    [
        "    *   Direct, concrete, not overly poetic.",
        "    *   Conversational.",
        "    *   Honest.",
    ],
]


def _generate_reasoning_example(num_blocks: int) -> str:
    """Build example reasoning with *num_blocks* blank-line-separated blocks."""
    blocks = _EXAMPLE_BLOCKS[:num_blocks]
    while len(blocks) < num_blocks:
        blocks.append([f"    *   (Additional analysis block {len(blocks) + 1}.)"])
    return "\n\n".join("\n".join(lines) for lines in blocks)


class GemmaReasoningTemplate(ResponseTemplate):
    """Structured reasoning in the Gemma-4 analytical block format.

    The model produces reasoning as a sequence of unlabelled,
    blank-line-separated blocks of indented bullets inside an XML tag,
    followed by the user-facing content in a ``<content>`` tag.
    """

    def __init__(
        self,
        *,
        reasoning_field: str = "reasoning",
        num_reasoning_blocks: int = 4,
    ) -> None:
        self._reasoning_field = reasoning_field
        self._num_reasoning_blocks = num_reasoning_blocks

    @property
    def name(self) -> str:
        return "gemma_reasoning"

    @property
    def description(self) -> str:
        return f"Gemma-4-style analytical blocks in <{self._reasoning_field}> + <content>"

    def get_params(self) -> list[TemplateParam]:
        return [
            TemplateParam(
                key="reasoning_field",
                label="Reasoning field name",
                param_type="str",
                default="reasoning",
                description="XML tag wrapping the analytical reasoning block",
                suggestions=["reasoning", "thought", "analysis", "think"],
            ),
            TemplateParam(
                key="num_reasoning_blocks",
                label="Reasoning blocks",
                param_type="int",
                default=4,
                description=("Minimum number of blank-line-separated analytical blocks"),
                min_value=3,
                max_value=8,
            ),
        ]

    def get_effective_params(self) -> dict[str, Any]:
        return {
            "reasoning_field": self._reasoning_field,
            "num_reasoning_blocks": self._num_reasoning_blocks,
        }

    def _build_with_params(self, params: dict[str, Any]) -> GemmaReasoningTemplate:
        return GemmaReasoningTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_blocks=params["num_reasoning_blocks"],
        )

    def get_fields(self) -> list[FieldDescriptor]:
        return [
            FieldDescriptor(
                name=self._reasoning_field,
                required=True,
                display_label=(f"\U0001f9e0 {self._reasoning_field.replace('_', ' ').title()}"),
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
        n = self._num_reasoning_blocks
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using the following XML tags. "
            "Do NOT output any text outside these tags.\n\n"
            f"<{rf}>...</{rf}>\n<content>...</content>\n\n"
            f"Tag order is strict: <{rf}> -> <content>\n\n"
            f"The <{rf}> tag contains your internal reasoning as a sequence "
            f"of at least {n} blank-line-separated blocks. "
            "Use indented bullet points (    *   ) within each block. "
            "Do NOT label the blocks with headers -- let each block's role "
            "emerge naturally from its content.\n\n"
            "Typical block progression (adapt to the situation):\n"
            "1. Parse the user's message: identify intent, tone, what they "
            "actually need.\n"
            "2. Recall relevant constraints: persona rules, conversation "
            "history, prior agreements.\n"
            "3. Analyse the problem in depth with multiple sub-points.\n"
            "4. Draft 2-3 candidate responses, evaluating each inline "
            "(e.g. *Option 1 (Too formal):* ..., *Option 2 (Better):* ...).\n"
            "5. Decide on the final approach and note what to include.\n"
            "6. Quick style/tone checklist (direct, conversational, honest, "
            "etc.).\n\n"
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        reasoning = _generate_reasoning_example(self._num_reasoning_blocks)
        return [
            TemplateExample(
                text=(
                    f"<{rf}>\n{reasoning}\n</{rf}>\n"
                    "<content>\nA Python decorator is a function that wraps "
                    "another function to extend its behavior. Here's the "
                    "basic pattern:\n\n```python\nimport functools\n\n"
                    "def my_decorator(func):\n"
                    "    @functools.wraps(func)\n"
                    "    def wrapper(*args, **kwargs):\n"
                    "        # do something before\n"
                    "        result = func(*args, **kwargs)\n"
                    "        # do something after\n"
                    "        return result\n"
                    "    return wrapper\n```\n\n"
                    "Always use `functools.wraps` to preserve the original "
                    "function's metadata.\n</content>"
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


register_template(GemmaReasoningTemplate())
