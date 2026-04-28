"""Tests for session cost tracking helpers."""

from __future__ import annotations

import pytest

from mai_gram.debug.cost_tracker import SessionCostTracker, _resolve_pricing_key
from mai_gram.llm.provider import TokenUsage


def test_resolve_pricing_key_handles_direct_suffix_and_unknown_models() -> None:
    assert _resolve_pricing_key(None) is None
    assert _resolve_pricing_key(" openai/gpt-4o ") == "openai/gpt-4o"
    assert _resolve_pricing_key("custom/gpt-4o-mini") == "gpt-4o-mini"
    assert _resolve_pricing_key("unknown/model") is None


def test_session_cost_tracker_records_costs_and_stats() -> None:
    tracker = SessionCostTracker()
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    breakdown = tracker.record(usage, model_name="custom/gpt-4o-mini")
    stats = tracker.stats()

    assert breakdown.model == "custom/gpt-4o-mini"
    assert breakdown.total_tokens == 1500
    assert breakdown.estimated_cost_usd == pytest.approx(0.00045)
    assert stats["calls"] == 1
    assert stats["prompt_tokens"] == 1000
    assert stats["completion_tokens"] == 500
    assert stats["total_tokens"] == 1500
    assert stats["estimated_cost_usd"] == pytest.approx(0.00045)


def test_session_cost_tracker_returns_zero_for_missing_pricing() -> None:
    tracker = SessionCostTracker(pricing_per_1k={"gpt-4o-mini": {"prompt": 0.1}})
    usage = TokenUsage(prompt_tokens=200, completion_tokens=300, total_tokens=500)

    missing_model = tracker.record(usage, model_name="other/model")
    partial_rates = tracker.record(usage, model_name="custom/gpt-4o-mini")

    assert missing_model.estimated_cost_usd == 0.0
    assert partial_rates.estimated_cost_usd == pytest.approx(0.02)
