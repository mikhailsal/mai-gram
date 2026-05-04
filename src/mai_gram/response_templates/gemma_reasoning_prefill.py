"""Gemma-4-style reasoning prefill -- adds assistant prefill.

The prefill primes the model with the opening reasoning tag and the first
bullet marker so it begins the analytical block sequence from the first
token.  This also disables native reasoning on providers that support it
(e.g. Google Gemini).
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.gemma_reasoning_template import (
    GemmaReasoningTemplate,
)
from mai_gram.response_templates.registry import register_template


class GemmaReasoningPrefillTemplate(GemmaReasoningTemplate):
    """Gemma-4 analytical reasoning template with assistant prefill."""

    @property
    def name(self) -> str:
        return "gemma_reasoning_prefill"

    @property
    def description(self) -> str:
        return (
            f"Gemma-4-style + prefill: analytical blocks in <{self._reasoning_field}> + <content>"
        )

    def _build_with_params(self, params: dict[str, Any]) -> GemmaReasoningPrefillTemplate:
        return GemmaReasoningPrefillTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_blocks=params["num_reasoning_blocks"],
        )

    def assistant_prefill(self) -> str:
        return f"<{self._reasoning_field}>\n*   "


register_template(GemmaReasoningPrefillTemplate())
