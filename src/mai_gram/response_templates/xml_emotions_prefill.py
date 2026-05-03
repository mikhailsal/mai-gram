"""XML-with-emotions prefill -- adds assistant prefill to XmlWithEmotionsTemplate.

The prefill primes the model with the opening reasoning tag, same as the
plain XML prefill, since the reasoning tag comes first in the field order.
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.registry import register_template
from mai_gram.response_templates.xml_with_emotions import XmlWithEmotionsTemplate


class XmlWithEmotionsPrefillTemplate(XmlWithEmotionsTemplate):
    """XML-with-emotions template with assistant prefill for the opening reasoning tag."""

    @property
    def name(self) -> str:
        return "xml_emotions_prefill"

    @property
    def description(self) -> str:
        rf = self._reasoning_field
        ef = self._emotions_field
        return f"XML + prefill: <{rf}> + <{ef}> + <content>"

    def _build_with_params(self, params: dict[str, Any]) -> XmlWithEmotionsPrefillTemplate:
        return XmlWithEmotionsPrefillTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
            emotions_field=params["emotions_field"],
            num_emotions=params["num_emotions"],
        )

    def assistant_prefill(self) -> str:
        return f"<{self._reasoning_field}>"


register_template(XmlWithEmotionsPrefillTemplate())
