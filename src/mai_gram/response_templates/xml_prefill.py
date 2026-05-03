"""XML prefill template -- inherits from XmlTemplate, adds assistant prefill.

The prefill primes the model with the opening reasoning tag so it begins
structured output from the first token.  This also disables native reasoning
on providers that support it (e.g. Google Gemini).
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_template import XmlTemplate


class XmlPrefillTemplate(XmlTemplate):
    """XML template with assistant prefill for the opening reasoning tag."""

    @property
    def name(self) -> str:
        return "xml_prefill"

    @property
    def description(self) -> str:
        return f"XML + prefill: <{self._reasoning_field}> + <content>"

    def _build_with_params(self, params: dict[str, Any]) -> XmlPrefillTemplate:
        return XmlPrefillTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
        )

    def assistant_prefill(self) -> str:
        return f"<{self._reasoning_field}>"


register_template(XmlPrefillTemplate())
