"""Token usage and approximate cost tracking for LLM sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mai_gram.llm.provider import TokenUsage

# Approximate USD price per 1K tokens.
DEFAULT_MODEL_PRICING_PER_1K: dict[str, dict[str, float]] = {
    "openai/gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "openai/gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
}


def _resolve_pricing_key(model_name: str | None) -> str | None:
    if not model_name:
        return None
    normalized = model_name.strip().lower()
    if normalized in DEFAULT_MODEL_PRICING_PER_1K:
        return normalized
    if "/" in normalized:
        suffix = normalized.split("/", maxsplit=1)[1]
        if suffix in DEFAULT_MODEL_PRICING_PER_1K:
            return suffix
    return None


@dataclass(frozen=True)
class CostBreakdown:
    """One-shot cost estimation result for a token usage sample."""

    model: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class SessionCostTracker:
    """Accumulates token usage and computes approximate USD costs."""

    def __init__(
        self,
        *,
        pricing_per_1k: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._pricing = {
            key.lower(): value
            for key, value in (pricing_per_1k or DEFAULT_MODEL_PRICING_PER_1K).items()
        }
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._total_cost_usd = 0.0

    def _estimate_cost_usd(self, usage: TokenUsage, model_name: str | None) -> float:
        pricing_key = _resolve_pricing_key(model_name)
        if pricing_key is None:
            return 0.0
        rates = self._pricing.get(pricing_key)
        if rates is None:
            return 0.0
        prompt_rate = float(rates.get("prompt", 0.0))
        completion_rate = float(rates.get("completion", 0.0))
        return (usage.prompt_tokens / 1000.0) * prompt_rate + (
            usage.completion_tokens / 1000.0
        ) * completion_rate

    def record(self, usage: TokenUsage, *, model_name: str | None) -> CostBreakdown:
        """Record one LLM call and return the per-call estimate."""
        self._calls += 1
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._total_tokens += usage.total_tokens
        call_cost = self._estimate_cost_usd(usage, model_name)
        self._total_cost_usd += call_cost
        return CostBreakdown(
            model=model_name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=call_cost,
        )

    def stats(self) -> dict[str, int | float]:
        return {
            "calls": self._calls,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": self._total_cost_usd,
        }
