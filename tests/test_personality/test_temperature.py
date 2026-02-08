"""Tests for personality/temperature.py -- trait-to-temperature mapping."""

from __future__ import annotations

from mai_companion.personality.temperature import (
    TEMPERATURE_ADJUSTMENTS,
    compute_temperature,
    describe_temperature,
)
from mai_companion.personality.traits import TRAIT_DEFINITIONS

# ---------------------------------------------------------------------------
# Temperature formula
# ---------------------------------------------------------------------------

class TestComputeTemperature:
    """Verify the temperature computation formula."""

    def test_all_medium_returns_base(self) -> None:
        """All traits at 0.5 should produce the base temperature (0.7)."""
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        assert compute_temperature(traits) == 0.7

    def test_high_humor_raises_temperature(self) -> None:
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        traits["humor"] = 0.9
        temp = compute_temperature(traits)
        assert temp > 0.7

    def test_low_humor_lowers_temperature(self) -> None:
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        traits["humor"] = 0.1
        temp = compute_temperature(traits)
        assert temp < 0.7

    def test_high_directness_lowers_temperature(self) -> None:
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        traits["directness"] = 0.9
        temp = compute_temperature(traits)
        assert temp < 0.7

    def test_clamps_to_min(self) -> None:
        """Even with extreme values, temperature should not go below 0.3."""
        traits = {
            "humor": 0.0,
            "directness": 1.0,
            "patience": 1.0,
            "laziness": 1.0,
        }
        temp = compute_temperature(traits)
        assert temp >= 0.3

    def test_clamps_to_max(self) -> None:
        """Even with extreme values, temperature should not exceed 1.4."""
        traits = {
            "humor": 1.0,
            "directness": 0.0,
            "patience": 0.0,
            "laziness": 0.0,
        }
        temp = compute_temperature(traits)
        assert temp <= 1.4

    def test_warmth_does_not_affect_temperature(self) -> None:
        """Warmth is not in TEMPERATURE_ADJUSTMENTS."""
        base_traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        base_temp = compute_temperature(base_traits)

        warm_traits = dict(base_traits)
        warm_traits["warmth"] = 0.9
        warm_temp = compute_temperature(warm_traits)

        assert base_temp == warm_temp

    def test_mood_volatility_does_not_affect_temperature(self) -> None:
        """mood_volatility is not in TEMPERATURE_ADJUSTMENTS."""
        base_traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        base_temp = compute_temperature(base_traits)

        volatile_traits = dict(base_traits)
        volatile_traits["mood_volatility"] = 0.9
        volatile_temp = compute_temperature(volatile_traits)

        assert base_temp == volatile_temp

    def test_unknown_traits_ignored(self) -> None:
        """Traits not in TEMPERATURE_ADJUSTMENTS are gracefully ignored."""
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        traits["future_trait"] = 0.9
        temp = compute_temperature(traits)
        assert temp == 0.7

    def test_empty_traits_returns_base(self) -> None:
        assert compute_temperature({}) == 0.7

    def test_witty_sidekick_higher_than_scholar(self) -> None:
        """Witty sidekick (high humor) should get higher temp than scholar."""
        witty = {
            "warmth": 0.6, "humor": 0.9, "patience": 0.3,
            "directness": 0.7, "laziness": 0.4, "mood_volatility": 0.6,
        }
        scholar = {
            "warmth": 0.5, "humor": 0.2, "patience": 0.9,
            "directness": 0.6, "laziness": 0.1, "mood_volatility": 0.2,
        }
        assert compute_temperature(witty) > compute_temperature(scholar)


# ---------------------------------------------------------------------------
# Temperature adjustments registry
# ---------------------------------------------------------------------------

class TestTemperatureAdjustments:
    """Verify the adjustments registry."""

    def test_wave1_adjustments_count(self) -> None:
        """Wave 1 has 4 traits with temperature adjustments."""
        assert len(TEMPERATURE_ADJUSTMENTS) == 4

    def test_expected_traits_in_adjustments(self) -> None:
        expected = {"humor", "directness", "patience", "laziness"}
        assert set(TEMPERATURE_ADJUSTMENTS.keys()) == expected


# ---------------------------------------------------------------------------
# describe_temperature
# ---------------------------------------------------------------------------

class TestDescribeTemperature:
    """Verify human-readable temperature descriptions."""

    def test_low_temperature(self) -> None:
        desc = describe_temperature(0.3)
        assert "focused" in desc.lower() or "deterministic" in desc.lower()

    def test_base_temperature(self) -> None:
        desc = describe_temperature(0.7)
        assert "balanced" in desc.lower()

    def test_high_temperature(self) -> None:
        desc = describe_temperature(1.0)
        assert "creative" in desc.lower()

    def test_format_includes_value(self) -> None:
        desc = describe_temperature(0.85)
        assert "0.85" in desc
