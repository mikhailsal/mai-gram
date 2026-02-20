"""Tests for personality/character.py -- CharacterBuilder and system prompt generation."""

from __future__ import annotations

import json

import pytest

from mai_companion.personality.character import (
    CharacterBuilder,
    CharacterConfig,
    CommunicationStyle,
    Gender,
    Verbosity,
    generate_system_prompt,
    regenerate_system_prompt_from_companion,
)

# ---------------------------------------------------------------------------
# CharacterConfig
# ---------------------------------------------------------------------------

class TestGenderEnum:
    """Verify Gender enum."""

    def test_three_genders(self) -> None:
        assert len(Gender) == 3

    def test_gender_values(self) -> None:
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NEUTRAL.value == "neutral"


class TestCharacterConfig:
    """Verify CharacterConfig dataclass."""

    def test_default_values(self) -> None:
        config = CharacterConfig(
            name="Test",
            language="English",
            traits={"warmth": 0.5},
        )
        assert config.communication_style == CommunicationStyle.BALANCED
        assert config.verbosity == Verbosity.NORMAL
        assert config.speech_variant is None
        assert config.appearance_description is None
        assert config.preset_name is None
        assert config.gender == Gender.NEUTRAL

    def test_gender_can_be_set(self) -> None:
        config = CharacterConfig(
            name="Test",
            language="English",
            traits={"warmth": 0.5},
            gender=Gender.FEMALE,
        )
        assert config.gender == Gender.FEMALE


# ---------------------------------------------------------------------------
# CharacterBuilder
# ---------------------------------------------------------------------------

class TestCharacterBuilder:
    """Verify CharacterBuilder factory methods."""

    def test_from_preset_valid(self) -> None:
        config = CharacterBuilder.from_preset("Aria", "witty_sidekick")
        assert config.name == "Aria"
        assert config.preset_name == "witty_sidekick"
        assert config.language == "English"
        assert len(config.traits) == 6

    def test_from_preset_with_language(self) -> None:
        config = CharacterBuilder.from_preset(
            "Aria", "caring_guide", language="Russian"
        )
        assert config.language == "Russian"

    def test_from_preset_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            CharacterBuilder.from_preset("Aria", "nonexistent_preset")

    def test_from_traits(self) -> None:
        traits = {"warmth": 0.8, "humor": 0.3}
        config = CharacterBuilder.from_traits("Kai", traits, language="Spanish")
        assert config.name == "Kai"
        assert config.language == "Spanish"
        # Validate fills in missing traits
        assert len(config.traits) == 6
        assert config.traits["warmth"] == 0.8
        assert config.traits["humor"] == 0.3

    def test_from_traits_clamps(self) -> None:
        traits = {"warmth": 2.0, "humor": -1.0}
        config = CharacterBuilder.from_traits("Kai", traits)
        assert config.traits["warmth"] == 1.0
        assert config.traits["humor"] == 0.0

    def test_random_config(self) -> None:
        config = CharacterBuilder.random_config("Random")
        assert config.name == "Random"
        assert len(config.traits) == 6
        for value in config.traits.values():
            assert 0.0 <= value <= 1.0

    def test_random_configs_differ(self) -> None:
        configs = [CharacterBuilder.random_config("R") for _ in range(10)]
        warmth_values = [c.traits["warmth"] for c in configs]
        assert len(set(warmth_values)) > 1

    def test_config_gender_can_be_set_after_creation(self) -> None:
        """Gender is set on the config after creation (as done in onboarding)."""
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        assert config.gender == Gender.NEUTRAL  # default
        config.gender = Gender.FEMALE
        assert config.gender == Gender.FEMALE

    def test_get_presets(self) -> None:
        presets = CharacterBuilder.get_presets()
        assert len(presets) == 6
        assert "witty_sidekick" in presets

    def test_get_extreme_warning_balanced(self) -> None:
        config = CharacterBuilder.from_preset("Test", "balanced_friend")
        assert CharacterBuilder.get_extreme_warning(config) is None

    def test_get_extreme_warning_extreme(self) -> None:
        traits = {
            "warmth": 0.1,
            "directness": 0.9,
            "patience": 0.1,
            "humor": 0.5,
            "laziness": 0.5,
            "mood_volatility": 0.5,
        }
        config = CharacterBuilder.from_traits("Harsh", traits)
        warning = CharacterBuilder.get_extreme_warning(config)
        assert warning is not None


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

class TestSystemPromptGeneration:
    """Verify generate_system_prompt output."""

    def test_prompt_contains_name(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        prompt = generate_system_prompt(config)
        assert "Luna" in prompt

    def test_prompt_contains_language_section(self) -> None:
        config = CharacterBuilder.from_preset(
            "Luna", "caring_guide", language="Japanese"
        )
        prompt = generate_system_prompt(config)
        assert "Japanese" in prompt
        assert "## Language" in prompt

    def test_prompt_contains_personality_section(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        prompt = generate_system_prompt(config)
        assert "## Personality" in prompt

    def test_prompt_contains_communication_style(self) -> None:
        config = CharacterBuilder.from_preset(
            "Luna", "caring_guide",
            style=CommunicationStyle.FORMAL,
            verbosity=Verbosity.DETAILED,
        )
        prompt = generate_system_prompt(config)
        assert "## Communication style" in prompt
        assert "polished" in prompt.lower() or "formal" in prompt.lower()
        assert "detailed" in prompt.lower() or "thorough" in prompt.lower()

    def test_prompt_contains_ethical_floor(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        prompt = generate_system_prompt(config)
        assert "## Ethical boundaries" in prompt
        assert "self-harm" in prompt

    def test_prompt_contains_gender_instructions_male(self) -> None:
        config = CharacterBuilder.from_preset("Max", "bold_challenger")
        config.gender = Gender.MALE
        prompt = generate_system_prompt(config)
        assert "male" in prompt.lower() or "masculine" in prompt.lower()

    def test_prompt_contains_gender_instructions_female(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        config.gender = Gender.FEMALE
        prompt = generate_system_prompt(config)
        assert "female" in prompt.lower() or "feminine" in prompt.lower()

    def test_prompt_contains_gender_instructions_neutral(self) -> None:
        config = CharacterBuilder.from_preset("Alex", "balanced_friend")
        config.gender = Gender.NEUTRAL
        prompt = generate_system_prompt(config)
        assert "neutral" in prompt.lower() or "gender-neutral" in prompt.lower()

    def test_prompt_contains_mood_placeholder(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        prompt = generate_system_prompt(config)
        assert "{mood_section}" in prompt

    def test_prompt_contains_relationship_placeholder(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        prompt = generate_system_prompt(config)
        assert "{relationship_section}" in prompt

    def test_prompt_personality_varies_by_preset(self) -> None:
        """Different presets should produce different personality sections."""
        config_caring = CharacterBuilder.from_preset("A", "caring_guide")
        config_bold = CharacterBuilder.from_preset("A", "bold_challenger")
        prompt_caring = generate_system_prompt(config_caring)
        prompt_bold = generate_system_prompt(config_bold)
        # The personality sections should differ
        assert prompt_caring != prompt_bold

    def test_prompt_loops_over_registered_traits(self) -> None:
        """The personality block should have content for each registered trait."""
        config = CharacterBuilder.from_preset("Luna", "witty_sidekick")
        prompt = generate_system_prompt(config)
        # The prompt should contain personality-related text for each trait
        # We can check that the personality section has multiple paragraphs
        personality_start = prompt.index("## Personality")
        personality_end = prompt.index("## Communication style")
        personality_section = prompt[personality_start:personality_end]
        # 6 traits should produce 6 paragraphs
        paragraphs = [p for p in personality_section.split("\n\n") if p.strip()]
        # First paragraph is the header, then 6 trait paragraphs
        assert len(paragraphs) >= 6

    def test_prompt_contains_language_style_when_set(self) -> None:
        """Language style should be included in the prompt when specified."""
        config = CharacterBuilder.from_preset("Татьяна", "caring_guide", language="Russian")
        config.language_style = "pre-revolutionary orthography"
        prompt = generate_system_prompt(config)
        assert "pre-revolutionary orthography" in prompt
        assert "CRITICAL LANGUAGE STYLE REQUIREMENT" in prompt

    def test_prompt_no_language_style_section_when_none(self) -> None:
        """Language style section should not appear when style is None."""
        config = CharacterBuilder.from_preset("Luna", "caring_guide", language="Russian")
        config.language_style = None
        prompt = generate_system_prompt(config)
        assert "CRITICAL LANGUAGE STYLE REQUIREMENT" not in prompt


# ---------------------------------------------------------------------------
# create_companion_record
# ---------------------------------------------------------------------------

class TestCreateCompanionRecord:
    """Verify create_companion_record output."""

    def test_returns_correct_fields(self) -> None:
        config = CharacterBuilder.from_preset(
            "Luna", "caring_guide", language="Russian"
        )
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["name"] == "Luna"
        assert record["human_language"] == "Russian"
        assert record["temperature"] == 0.65
        assert record["relationship_stage"] == "getting_to_know"
        assert record["mood_volatility"] == config.traits["mood_volatility"]
        assert record["communication_style"] == "balanced"
        assert record["verbosity"] == "normal"
        # personality_traits should be valid JSON
        traits = json.loads(record["personality_traits"])
        assert isinstance(traits, dict)
        assert len(traits) == 6
        # system_prompt should be non-empty
        assert len(record["system_prompt"]) > 100

    def test_stores_communication_style_and_verbosity(self) -> None:
        config = CharacterBuilder.from_preset(
            "Luna", "caring_guide",
            style=CommunicationStyle.FORMAL,
            verbosity=Verbosity.DETAILED,
        )
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["communication_style"] == "formal"
        assert record["verbosity"] == "detailed"

    def test_includes_gender_field(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        config.gender = Gender.FEMALE
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["gender"] == "female"

    def test_gender_defaults_to_neutral(self) -> None:
        config = CharacterBuilder.from_preset("Alex", "balanced_friend")
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["gender"] == "neutral"

    def test_includes_language_style_field(self) -> None:
        config = CharacterBuilder.from_preset("Luna", "caring_guide", language="Russian")
        config.language_style = "pre-revolutionary orthography"
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["language_style"] == "pre-revolutionary orthography"

    def test_language_style_defaults_to_none(self) -> None:
        config = CharacterBuilder.from_preset("Alex", "balanced_friend")
        record = CharacterBuilder.create_companion_record(config, temperature=0.65)
        assert record["language_style"] is None


# ---------------------------------------------------------------------------
# regenerate_system_prompt_from_companion
# ---------------------------------------------------------------------------

class _FakeCompanion:
    """Minimal stand-in for a Companion ORM object (no DB needed)."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestRegenerateSystemPrompt:
    """Verify regenerate_system_prompt_from_companion produces correct output."""

    def test_basic_regeneration(self) -> None:
        companion = _FakeCompanion(
            name="TestBot",
            gender="neutral",
            human_language="English",
            language_style=None,
            personality_traits=json.dumps({
                "warmth": 0.5, "humor": 0.5, "directness": 0.5,
                "patience": 0.5, "laziness": 0.5, "mood_volatility": 0.5,
            }),
            communication_style="balanced",
            verbosity="normal",
        )
        prompt = regenerate_system_prompt_from_companion(companion)
        assert "TestBot" in prompt
        assert "## Personality" in prompt
        assert "## Ethical boundaries" in prompt
        assert "{mood_section}" in prompt
        assert "{relationship_section}" in prompt

    def test_regeneration_matches_generate(self) -> None:
        """Output from regenerate should match generate for the same config."""
        config = CharacterBuilder.from_preset(
            "Luna", "caring_guide", language="Spanish",
            style=CommunicationStyle.FORMAL,
            verbosity=Verbosity.DETAILED,
        )
        config.gender = Gender.FEMALE
        config.language_style = "Andalusian dialect"
        direct_prompt = generate_system_prompt(config)

        companion = _FakeCompanion(
            name="Luna",
            gender="female",
            human_language="Spanish",
            language_style="Andalusian dialect",
            personality_traits=json.dumps(config.traits),
            communication_style="formal",
            verbosity="detailed",
        )
        regenerated_prompt = regenerate_system_prompt_from_companion(companion)
        assert regenerated_prompt == direct_prompt

    def test_regeneration_with_defaults_for_missing_fields(self) -> None:
        """Old companions without communication_style/verbosity should get defaults."""
        companion = _FakeCompanion(
            name="OldBot",
            gender="male",
            human_language="English",
            personality_traits=json.dumps({
                "warmth": 0.5, "humor": 0.5, "directness": 0.5,
                "patience": 0.5, "laziness": 0.5, "mood_volatility": 0.5,
            }),
        )
        # Simulate missing attributes (old DB schema)
        # regenerate should fall back to defaults
        prompt = regenerate_system_prompt_from_companion(companion)
        assert "OldBot" in prompt
        assert "## Ethical boundaries" in prompt

    def test_regeneration_preserves_gender(self) -> None:
        companion = _FakeCompanion(
            name="Мария",
            gender="female",
            human_language="Russian",
            language_style=None,
            personality_traits=json.dumps({
                "warmth": 0.8, "humor": 0.5, "directness": 0.5,
                "patience": 0.5, "laziness": 0.5, "mood_volatility": 0.5,
            }),
            communication_style="casual",
            verbosity="concise",
        )
        prompt = regenerate_system_prompt_from_companion(companion)
        assert "feminine" in prompt.lower() or "female" in prompt.lower()


class TestCharacterConfigLanguageStyle:
    """Verify CharacterConfig language_style field."""

    def test_language_style_default_is_none(self) -> None:
        config = CharacterConfig(
            name="Test",
            language="English",
            traits={"warmth": 0.5},
        )
        assert config.language_style is None

    def test_language_style_can_be_set(self) -> None:
        config = CharacterConfig(
            name="Test",
            language="Russian",
            traits={"warmth": 0.5},
            language_style="pre-revolutionary orthography",
        )
        assert config.language_style == "pre-revolutionary orthography"
