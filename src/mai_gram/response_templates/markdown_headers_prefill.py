"""Markdown-headers prefill -- adds assistant prefill to MarkdownHeadersTemplate.

The prefill primes the model with the first markdown section header so it
begins structured output immediately.
"""

from __future__ import annotations

from typing import Any

from mai_gram.response_templates.markdown_headers_template import MarkdownHeadersTemplate
from mai_gram.response_templates.registry import register_template


class MarkdownHeadersPrefillTemplate(MarkdownHeadersTemplate):
    """Markdown headers template with assistant prefill for the opening section."""

    @property
    def name(self) -> str:
        return "markdown_headers_prefill"

    @property
    def description(self) -> str:
        return f"Markdown + prefill: ## {self._reasoning_field} + ## Content"

    def _build_with_params(self, params: dict[str, Any]) -> MarkdownHeadersPrefillTemplate:
        return MarkdownHeadersPrefillTemplate(
            reasoning_field=params["reasoning_field"],
            num_reasoning_paragraphs=params["num_reasoning_paragraphs"],
        )

    def assistant_prefill(self) -> str:
        return f"## {self._reasoning_field}\n"


register_template(MarkdownHeadersPrefillTemplate())
