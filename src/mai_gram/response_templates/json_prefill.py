"""JSON prefill template -- inherits from JsonTemplate, adds assistant prefill.

The prefill primes the model with the beginning of the JSON object so it
starts structured output immediately.
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.json_template import JsonTemplate
from mai_gram.response_templates.registry import register_template


class JsonPrefillTemplate(JsonTemplate):
    """JSON template with assistant prefill for the opening JSON structure."""

    @property
    def name(self) -> str:
        return "json_prefill"

    @property
    def description(self) -> str:
        return f'JSON + prefill: {{"{self._reasoning_field}", "content"}}'

    def _build_with_params(self, params: dict[str, Any]) -> JsonPrefillTemplate:
        return JsonPrefillTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
        )

    def assistant_prefill(self) -> str:
        return '{"' + self._reasoning_field + '": "'


register_template(JsonPrefillTemplate())
