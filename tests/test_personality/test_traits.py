"""Tests for personality/traits.py -- Wave 1 trait definitions, presets, validation."""

from __future__ import annotations

import pytest

from mai_companion.personality.traits import (
    PRESETS,
    TRAIT_BEHAVIORAL_INSTRUCTIONS,
    TRAIT_DEFINITIONS,
    TraitLevel,
    TraitName,
    detect_extreme_config,
    generate_random_traits,
    get_registered_traits,
    validate_traits,
    value_to_level,
)

# ---------------------------------------------------------------------------
# Trait definitions
# ---------------------------------------------------------------------------

class TestTraitDefinitions:
    """Verify the Wave 1 trait registry."""

    def test_wave1_has_6_traits(self) -> None:
        assert len(TRAIT_DEFINITIONS) == 6

    def test_wave1_trait_names(self) -> None:
        expected = {
            TraitName.WARMTH,
            TraitName.HUMOR,
            TraitName.PATIENCE,
            TraitName.DIRECTNESS,
            TraitName.LAZINESS,
            TraitName.MOOD_VOLATILITY,
        }
        assert set(TRAIT_DEFINITIONS.keys()) == expected

    def test_get_registered_traits_returns_wave1(self) -> None:
        registered = get_registered_traits()
        assert len(registered) == 6
        assert all(isinstance(t, TraitName) for t in registered)

    def test_each_definition_has_required_fields(self) -> None:
        for trait_name, defn in TRAIT_DEFINITIONS.items():
            assert defn.name == trait_name
            assert defn.display_name
            assert defn.description
            assert defn.low_label
            assert defn.high_label

    def test_enum_has_all_13_traits(self) -> None:
        """The enum defines all 13 traits (roadmap), even though only 6 are registered."""
        assert len(TraitName) == 13

    def test_unregistered_traits_not_in_definitions(self) -> None:
        """Wave 2/3 traits exist in enum but not in TRAIT_DEFINITIONS."""
        wave2_3 = {
            TraitName.ASSERTIVENESS,
            TraitName.CURIOSITY,
            TraitName.EMOTIONAL_DEPTH,
            TraitName.INDEPENDENCE,
            TraitName.HELPFULNESS,
            TraitName.PROACTIVENESS,
            TraitName.SPECIAL_SPEECH,
        }
        for trait in wave2_3:
            assert trait not in TRAIT_DEFINITIONS


# ---------------------------------------------------------------------------
# Behavioral instructions
# ---------------------------------------------------------------------------

class TestBehavioralInstructions:
    """Verify the trait behavioral instruction templates."""

    def test_30_instruction_templates(self) -> None:
        """Wave 1: 6 traits x 5 levels = 30 templates."""
        assert len(TRAIT_BEHAVIORAL_INSTRUCTIONS) == 30

    def test_every_registered_trait_has_all_5_levels(self) -> None:
        for trait_name in TRAIT_DEFINITIONS:
            for level in TraitLevel:
                key = (trait_name, level)
                assert key in TRAIT_BEHAVIORAL_INSTRUCTIONS, (
                    f"Missing instruction for ({trait_name.value}, {level.name})"
                )

    def test_instructions_are_non_empty_strings(self) -> None:
        for key, instruction in TRAIT_BEHAVIORAL_INSTRUCTIONS.items():
            assert isinstance(instruction, str)
            assert len(instruction) > 20, f"Instruction for {key} is too short"


# ---------------------------------------------------------------------------
# TraitLevel
# ---------------------------------------------------------------------------

class TestTraitLevel:
    """Verify TraitLevel enum and value_to_level mapping."""

    def test_five_levels(self) -> None:
        assert len(TraitLevel) == 5

    def test_level_values(self) -> None:
        assert TraitLevel.VERY_LOW.value == 0.1
        assert TraitLevel.LOW.value == 0.3
        assert TraitLevel.MEDIUM.value == 0.5
        assert TraitLevel.HIGH.value == 0.7
        assert TraitLevel.VERY_HIGH.value == 0.9

    def test_level_labels(self) -> None:
        assert TraitLevel.VERY_LOW.label == "Very Low"
        assert TraitLevel.MEDIUM.label == "Medium"
        assert TraitLevel.VERY_HIGH.label == "Very High"

    @pytest.mark.parametrize(
        "value, expected_level",
        [
            (0.0, TraitLevel.VERY_LOW),
            (0.1, TraitLevel.VERY_LOW),
            (0.15, TraitLevel.VERY_LOW),
            (0.3, TraitLevel.LOW),
            (0.35, TraitLevel.LOW),
            (0.5, TraitLevel.MEDIUM),
            (0.55, TraitLevel.MEDIUM),
            (0.7, TraitLevel.HIGH),
            (0.75, TraitLevel.HIGH),
            (0.9, TraitLevel.VERY_HIGH),
            (1.0, TraitLevel.VERY_HIGH),
        ],
    )
    def test_value_to_level(self, value: float, expected_level: TraitLevel) -> None:
        assert value_to_level(value) == expected_level

    def test_value_to_level_midpoints_are_consistent(self) -> None:
        """Midpoints between levels go to one of the adjacent levels."""
        # 0.2 is equidistant between VERY_LOW(0.1) and LOW(0.3)
        result = value_to_level(0.2)
        assert result in (TraitLevel.VERY_LOW, TraitLevel.LOW)
        # 0.4 is equidistant between LOW(0.3) and MEDIUM(0.5)
        result = value_to_level(0.4)
        assert result in (TraitLevel.LOW, TraitLevel.MEDIUM)
        # 0.8 is equidistant between HIGH(0.7) and VERY_HIGH(0.9)
        result = value_to_level(0.8)
        assert result in (TraitLevel.HIGH, TraitLevel.VERY_HIGH)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    """Verify personality presets."""

    def test_six_presets(self) -> None:
        assert len(PRESETS) == 6

    def test_expected_preset_names(self) -> None:
        expected = {
            "thoughtful_scholar",
            "witty_sidekick",
            "caring_guide",
            "bold_challenger",
            "balanced_friend",
            "free_spirit",
        }
        assert set(PRESETS.keys()) == expected

    def test_presets_have_all_registered_traits(self) -> None:
        """Every preset must include values for all registered traits."""
        registered_keys = {t.value for t in TRAIT_DEFINITIONS}
        for preset_name, preset in PRESETS.items():
            preset_keys = set(preset.trait_values.keys())
            missing = registered_keys - preset_keys
            assert not missing, (
                f"Preset '{preset_name}' missing traits: {missing}"
            )

    def test_preset_values_in_range(self) -> None:
        for preset_name, preset in PRESETS.items():
            for trait, value in preset.trait_values.items():
                assert 0.0 <= value <= 1.0, (
                    f"Preset '{preset_name}' trait '{trait}' = {value} out of range"
                )

    def test_presets_have_required_fields(self) -> None:
        for preset_name, preset in PRESETS.items():
            assert preset.name, f"Preset '{preset_name}' missing name"
            assert preset.tagline, f"Preset '{preset_name}' missing tagline"
            assert preset.description, f"Preset '{preset_name}' missing description"
            assert preset.example_line, f"Preset '{preset_name}' missing example_line"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Verify trait validation logic."""

    def test_validate_fills_missing_traits(self) -> None:
        result = validate_traits({})
        assert len(result) == 6
        assert all(v == 0.5 for v in result.values())

    def test_validate_clamps_values(self) -> None:
        result = validate_traits({"warmth": 1.5, "humor": -0.5})
        assert result["warmth"] == 1.0
        assert result["humor"] == 0.0

    def test_validate_preserves_valid_values(self) -> None:
        result = validate_traits({"warmth": 0.3, "humor": 0.8})
        assert result["warmth"] == 0.3
        assert result["humor"] == 0.8

    def test_validate_ignores_unknown_traits(self) -> None:
        result = validate_traits({"warmth": 0.5, "unknown_trait": 0.9})
        assert "unknown_trait" not in result

    def test_validate_returns_only_registered_traits(self) -> None:
        result = validate_traits({"warmth": 0.3})
        registered_keys = {t.value for t in TRAIT_DEFINITIONS}
        assert set(result.keys()) == registered_keys


# ---------------------------------------------------------------------------
# Extreme config detection
# ---------------------------------------------------------------------------

class TestExtremeDetection:
    """Verify extreme personality detection."""

    def test_balanced_config_no_warning(self) -> None:
        traits = {t.value: 0.5 for t in TRAIT_DEFINITIONS}
        assert detect_extreme_config(traits) is None

    def test_cold_direct_impatient_warns(self) -> None:
        traits = {
            "warmth": 0.1,
            "directness": 0.9,
            "patience": 0.1,
            "humor": 0.5,
            "laziness": 0.5,
            "mood_volatility": 0.5,
        }
        warning = detect_extreme_config(traits)
        assert warning is not None
        w = warning.lower()
        assert "cold" in w or "blunt" in w or "direct" in w

    def test_very_lazy_very_impatient_warns(self) -> None:
        traits = {
            "warmth": 0.5,
            "directness": 0.5,
            "patience": 0.1,
            "humor": 0.5,
            "laziness": 0.9,
            "mood_volatility": 0.5,
        }
        warning = detect_extreme_config(traits)
        assert warning is not None
        assert "lazy" in warning.lower()

    def test_cold_lazy_humorless_warns(self) -> None:
        traits = {
            "warmth": 0.1,
            "directness": 0.5,
            "patience": 0.5,
            "humor": 0.1,
            "laziness": 0.9,
            "mood_volatility": 0.5,
        }
        warning = detect_extreme_config(traits)
        assert warning is not None
        assert "brick wall" in warning.lower()


# ---------------------------------------------------------------------------
# Random trait generation
# ---------------------------------------------------------------------------

class TestRandomTraits:
    """Verify random trait generation."""

    def test_generates_all_registered_traits(self) -> None:
        traits = generate_random_traits()
        registered_keys = {t.value for t in TRAIT_DEFINITIONS}
        assert set(traits.keys()) == registered_keys

    def test_values_in_range(self) -> None:
        for _ in range(20):
            traits = generate_random_traits()
            for key, value in traits.items():
                assert 0.1 <= value <= 0.9, f"Trait '{key}' = {value} out of range"

    def test_generates_varied_values(self) -> None:
        """Multiple random configs should not be identical."""
        configs = [generate_random_traits() for _ in range(10)]
        # At least some variation
        warmth_values = [c["warmth"] for c in configs]
        assert len(set(warmth_values)) > 1


# ---------------------------------------------------------------------------
# Extensibility
# ---------------------------------------------------------------------------

class TestExtensibility:
    """Verify that the architecture supports adding new traits."""

    def test_adding_trait_to_registry_is_picked_up_by_validate(self) -> None:
        """Simulate adding a new trait -- validate_traits should include it."""
        from mai_companion.personality.traits import TraitDefinition

        # Temporarily add a fake trait
        fake_trait = TraitName.ASSERTIVENESS  # exists in enum, not in defs
        fake_def = TraitDefinition(
            name=fake_trait,
            display_name="Assertiveness",
            description="Test",
            low_label="Low",
            high_label="High",
        )
        TRAIT_DEFINITIONS[fake_trait] = fake_def
        try:
            result = validate_traits({})
            assert "assertiveness" in result
            assert result["assertiveness"] == 0.5
        finally:
            # Clean up
            del TRAIT_DEFINITIONS[fake_trait]

    def test_validate_still_works_after_cleanup(self) -> None:
        """After removing the fake trait, validate returns 6 traits again."""
        result = validate_traits({})
        assert len(result) == 6
