"""Gemma-4-style structured reasoning template.

Produces reasoning in the native Gemma-4 analytical format: a sequence of
distinct cognitive steps (situation analysis, context recall, deep analysis,
option drafting, final refinement) expressed as indented bullet-point
sections.  This is fundamentally different from paragraph-based reasoning --
each step represents a separate phase of the thinking process.

The format is modelled on real reasoning_content produced by
``google/gemma-4-31b-it`` across 100+ conversation turns.

Supports user-configurable parameters:
- num_reasoning_steps: minimum distinct analytical steps (default 4)
- reasoning_field: XML tag name wrapping the reasoning block (default "reasoning")
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

_STEP_LABELS = [
    "Situation Analysis",
    "Context & Constraints",
    "Deep Analysis",
    "Options & Drafts",
    "Evaluation & Refinement",
    "Final Plan",
]

_STEP_DESCRIPTIONS = {
    "Situation Analysis": (
        "Parse the user's message: identify their intent, emotional tone, "
        "and what kind of response they expect."
    ),
    "Context & Constraints": (
        "Recall relevant constraints: persona rules, conversation history, "
        "prior agreements, or system instructions that apply."
    ),
    "Deep Analysis": (
        "Break down the core problem with multiple sub-points. "
        "Consider different angles, implications, and connections."
    ),
    "Options & Drafts": (
        "Generate 2-3 candidate responses or approaches. "
        "Briefly evaluate each for tone, accuracy, and alignment with your role."
    ),
    "Evaluation & Refinement": (
        "Select the best option. Check for style, conciseness, honesty, "
        "and whether it matches the persona and conversation flow."
    ),
    "Final Plan": (
        "Summarize the chosen approach in 1-2 sentences: "
        "what to say, how to say it, and what to avoid."
    ),
}


def _generate_step_example(num_steps: int) -> str:
    """Build example reasoning with exactly *num_steps* analytical steps."""
    steps = _STEP_LABELS[:num_steps]
    while len(steps) < num_steps:
        steps.append(f"Additional Step {len(steps) + 1}")

    lines: list[str] = []
    for step in steps:
        desc = _STEP_DESCRIPTIONS.get(step)
        lines.append(f"    *   **{step}:**")
        if step == "Situation Analysis":
            lines.append(
                "    *   The user asks about Python decorators. Tone is curious, not adversarial."
            )
            lines.append("    *   They seem to want a practical explanation, not just theory.")
        elif step == "Context & Constraints":
            lines.append("    *   I should be direct and concrete, avoid excessive metaphors.")
            lines.append("    *   Keep it conversational, 1-3 paragraphs.")
        elif step == "Deep Analysis":
            lines.append(
                "    *   Decorators are a form of metaprogramming. "
                "The basic pattern is a wrapper function."
            )
            lines.append(
                "    *   Common pitfall: losing function metadata without functools.wraps."
            )
            lines.append("    *   Class-based vs function-based decorators are worth mentioning.")
        elif step == "Options & Drafts":
            lines.append(
                "    *   *Option 1 (Theory-heavy):* Start with the abstract concept. Too academic."
            )
            lines.append(
                "    *   *Option 2 (Code-first):* Lead with a code "
                "snippet, explain after. More engaging."
            )
            lines.append(
                "    *   *Option 3 (Balanced):* Brief concept, then "
                "code, then common gotchas. Best fit."
            )
        elif step == "Evaluation & Refinement":
            lines.append("    *   Option 3 is best: practical, direct, covers the key points.")
            lines.append("    *   Keep the code snippet concise. Mention functools.wraps as a tip.")
        elif step == "Final Plan":
            lines.append(
                "    *   Lead with a one-sentence definition, "
                "show a concrete example, end with the wraps tip."
            )
        else:
            lines.append(f"    *   ({desc or 'Further analysis.'})")
        lines.append("")
    return "\n".join(lines).rstrip()


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


_STEP_BLOCK_RE = re.compile(
    r"(?:^|\n)\s{0,4}\*\s{1,4}(?:\*{1,2}[^*]+\*{1,2}:?|\*[^*]+\*:?)",
)


def _count_reasoning_steps(text: str) -> int:
    """Count distinct top-level analytical steps in reasoning text."""
    return len(_STEP_BLOCK_RE.findall(text))


class GemmaReasoningTemplate(ResponseTemplate):
    """Structured reasoning in the Gemma-4 analytical step format.

    The model produces reasoning as a sequence of labelled analytical
    steps (situation analysis, context recall, deep analysis, option
    drafting, refinement) wrapped in an XML tag, followed by the
    user-facing content.
    """

    def __init__(
        self,
        *,
        reasoning_field: str = "reasoning",
        num_reasoning_steps: int = 4,
    ) -> None:
        self._reasoning_field = reasoning_field
        self._num_reasoning_steps = num_reasoning_steps

    @property
    def name(self) -> str:
        return "gemma_reasoning"

    @property
    def description(self) -> str:
        return f"Gemma-4-style analytical steps in <{self._reasoning_field}> + <content>"

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
                key="num_reasoning_steps",
                label="Reasoning steps",
                param_type="int",
                default=4,
                description="Minimum number of distinct analytical steps",
                min_value=2,
                max_value=6,
            ),
        ]

    def get_effective_params(self) -> dict[str, Any]:
        return {
            "reasoning_field": self._reasoning_field,
            "num_reasoning_steps": self._num_reasoning_steps,
        }

    def _build_with_params(self, params: dict[str, Any]) -> GemmaReasoningTemplate:
        return GemmaReasoningTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_steps=params["num_reasoning_steps"],
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
        n = self._num_reasoning_steps
        steps_spec = "\n".join(
            f"    {i + 1}. **{label}** -- {_STEP_DESCRIPTIONS[label]}"
            for i, label in enumerate(_STEP_LABELS[:n])
        )
        return (
            "\n\n--- RESPONSE FORMAT ---\n"
            "You MUST structure EVERY response using the following XML tags. "
            "Do NOT output any text outside these tags.\n\n"
            f"<{rf}>...</{rf}>\n<content>...</content>\n\n"
            f"Tag order is strict: <{rf}> -> <content>\n\n"
            f"The <{rf}> tag contains your internal reasoning as a sequence "
            f"of at least {n} distinct analytical steps. "
            "Use indented bullet points (* ) for each step, with bold labels "
            "(**Step Name:**). Each step should represent a separate cognitive "
            "phase, not just a continuation of the previous thought.\n\n"
            "Recommended step types (adapt as needed):\n"
            f"{steps_spec}\n\n"
            "Within each step, use sub-bullets for individual points. "
            "When evaluating options, label them (*Option 1:*, *Option 2:*, etc.) "
            "and briefly assess each.\n\n"
            "The <content> tag contains your response to the human."
        )

    def examples(self) -> list[TemplateExample]:
        rf = self._reasoning_field
        reasoning = _generate_step_example(self._num_reasoning_steps)
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
