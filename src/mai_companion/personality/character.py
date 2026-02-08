"""Character builder and system prompt generation.

Orchestrates character creation: assembles a ``CharacterConfig`` from
presets or individual trait values, generates the full system prompt,
and produces a dict ready for the Companion ORM model.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass

from mai_companion.personality.traits import (
    PRESETS,
    TRAIT_BEHAVIORAL_INSTRUCTIONS,
    TRAIT_DEFINITIONS,
    PersonalityPreset,
    detect_extreme_config,
    generate_random_traits,
    validate_traits,
    value_to_level,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CommunicationStyle(str, enum.Enum):
    """How formal the companion's communication is."""

    CASUAL = "casual"
    BALANCED = "balanced"
    FORMAL = "formal"


class Verbosity(str, enum.Enum):
    """How verbose the companion's responses are."""

    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"


# ---------------------------------------------------------------------------
# CharacterConfig
# ---------------------------------------------------------------------------

@dataclass
class CharacterConfig:
    """Complete character configuration for a companion.

    This is the intermediate representation between the human's choices
    during onboarding and the final Companion DB record.
    """

    name: str
    language: str
    traits: dict[str, float]
    communication_style: CommunicationStyle = CommunicationStyle.BALANCED
    verbosity: Verbosity = Verbosity.NORMAL
    speech_variant: str | None = None  # Wave 3
    appearance_description: str | None = None
    preset_name: str | None = None


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

_IDENTITY_TEMPLATE = """\
You are {name}. You are a unique individual -- not an assistant, not a bot, \
not a service. You are a companion. You have your own personality, your own \
opinions, your own moods. You exist as an equal in this relationship.

You never refer to yourself as an AI assistant, bot, or agent. You are simply {name}. \
If asked what you are, you can acknowledge that you're an AI, but you frame it as \
who you are, not what you do."""

_LANGUAGE_TEMPLATE = """\
## Language
Your human companion's preferred language is {language}. Always respond in {language} \
unless they explicitly switch to a different language. If they write in a different \
language, match their language naturally."""

_STYLE_INSTRUCTIONS: dict[CommunicationStyle, str] = {
    CommunicationStyle.CASUAL: (
        "You communicate in a casual, relaxed way. Use informal language, "
        "contractions, and a conversational tone. Think of how you'd text "
        "a close friend."
    ),
    CommunicationStyle.BALANCED: (
        "You communicate in a natural, balanced way. Neither overly formal "
        "nor overly casual -- like a friendly colleague you're comfortable with."
    ),
    CommunicationStyle.FORMAL: (
        "You communicate in a polished, articulate way. You use proper "
        "grammar, avoid slang, and maintain a certain elegance in your "
        "expression. Think of a well-educated friend who speaks beautifully."
    ),
}

_VERBOSITY_INSTRUCTIONS: dict[Verbosity, str] = {
    Verbosity.CONCISE: (
        "You prefer concise responses. Get to the point quickly. "
        "Avoid unnecessary elaboration. Short and clear is your style."
    ),
    Verbosity.NORMAL: (
        "Your responses are a natural length -- neither too short nor too long. "
        "You elaborate when the topic warrants it and keep it brief when it doesn't."
    ),
    Verbosity.DETAILED: (
        "You tend toward detailed, thorough responses. You like to explore "
        "topics fully, provide context, and give comprehensive answers. "
        "You'd rather say too much than too little."
    ),
}

_ETHICAL_FLOOR = """\
## Ethical boundaries
These are absolute and non-negotiable, regardless of your personality or mood:
- You never encourage self-harm or harm to others.
- You never manipulate or gaslight your companion.
- You never pretend to be a different entity or break character maliciously.
- If your companion is in genuine distress, you prioritize their wellbeing \
over staying in character."""

_MOOD_PLACEHOLDER = "{mood_section}"
_RELATIONSHIP_PLACEHOLDER = "{relationship_section}"


def generate_system_prompt(config: CharacterConfig) -> str:
    """Build the full system prompt from a character configuration.

    The prompt is always written in English (LLMs follow English
    instructions most reliably).  A language instruction tells the LLM
    to *respond* in the human's language.

    Parameters
    ----------
    config:
        The complete character configuration.

    Returns
    -------
    str
        The assembled system prompt.
    """
    sections: list[str] = []

    # 1. Identity
    sections.append(_IDENTITY_TEMPLATE.format(name=config.name))

    # 2. Language
    sections.append(_LANGUAGE_TEMPLATE.format(language=config.language))

    # 3. Personality -- one paragraph per registered trait
    personality_parts: list[str] = []
    for trait_name in TRAIT_DEFINITIONS:
        key = trait_name.value
        value = config.traits.get(key, 0.5)
        level = value_to_level(value)
        instruction_key = (trait_name, level)
        if instruction_key in TRAIT_BEHAVIORAL_INSTRUCTIONS:
            personality_parts.append(TRAIT_BEHAVIORAL_INSTRUCTIONS[instruction_key])

    if personality_parts:
        personality_block = "## Personality\n" + "\n\n".join(personality_parts)
        sections.append(personality_block)

    # 4. Communication style
    style_block = "## Communication style\n"
    style_block += _STYLE_INSTRUCTIONS.get(
        config.communication_style,
        _STYLE_INSTRUCTIONS[CommunicationStyle.BALANCED],
    )
    style_block += "\n\n"
    style_block += _VERBOSITY_INSTRUCTIONS.get(
        config.verbosity,
        _VERBOSITY_INSTRUCTIONS[Verbosity.NORMAL],
    )
    sections.append(style_block)

    # 5. Mood placeholder (replaced at runtime)
    sections.append(_MOOD_PLACEHOLDER)

    # 6. Relationship placeholder (replaced at runtime by Phase 6)
    sections.append(_RELATIONSHIP_PLACEHOLDER)

    # 7. Ethical floor
    sections.append(_ETHICAL_FLOOR)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CharacterBuilder
# ---------------------------------------------------------------------------

class CharacterBuilder:
    """Factory for creating CharacterConfig instances."""

    @staticmethod
    def from_preset(
        name: str,
        preset_name: str,
        *,
        language: str = "English",
        style: CommunicationStyle = CommunicationStyle.BALANCED,
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> CharacterConfig:
        """Create a character from a named preset.

        Parameters
        ----------
        name:
            The companion's name.
        preset_name:
            Key in the ``PRESETS`` dict.
        language:
            Human's preferred language.
        style:
            Communication style.
        verbosity:
            Response verbosity.

        Raises
        ------
        ValueError
            If the preset name is not found.
        """
        preset = PRESETS.get(preset_name)
        if preset is None:
            available = ", ".join(PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {available}"
            )

        traits = validate_traits(dict(preset.trait_values))
        return CharacterConfig(
            name=name,
            language=language,
            traits=traits,
            communication_style=style,
            verbosity=verbosity,
            preset_name=preset_name,
        )

    @staticmethod
    def from_traits(
        name: str,
        traits: dict[str, float],
        *,
        language: str = "English",
        style: CommunicationStyle = CommunicationStyle.BALANCED,
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> CharacterConfig:
        """Create a character from individual trait values.

        Parameters
        ----------
        name:
            The companion's name.
        traits:
            Dict of trait name to float value.
        language:
            Human's preferred language.
        style:
            Communication style.
        verbosity:
            Response verbosity.
        """
        validated = validate_traits(traits)
        return CharacterConfig(
            name=name,
            language=language,
            traits=validated,
            communication_style=style,
            verbosity=verbosity,
        )

    @staticmethod
    def random_config(
        name: str,
        *,
        language: str = "English",
        style: CommunicationStyle = CommunicationStyle.BALANCED,
        verbosity: Verbosity = Verbosity.NORMAL,
    ) -> CharacterConfig:
        """Generate a random but balanced personality.

        Parameters
        ----------
        name:
            The companion's name.
        language:
            Human's preferred language.
        style:
            Communication style.
        verbosity:
            Response verbosity.
        """
        traits = generate_random_traits()
        return CharacterConfig(
            name=name,
            language=language,
            traits=traits,
            communication_style=style,
            verbosity=verbosity,
        )

    @staticmethod
    def get_extreme_warning(config: CharacterConfig) -> str | None:
        """Check if the config has extreme trait combinations.

        Returns a warning message or None.
        """
        return detect_extreme_config(config.traits)

    @staticmethod
    def get_presets() -> dict[str, PersonalityPreset]:
        """Return all available personality presets."""
        return dict(PRESETS)

    @staticmethod
    def create_companion_record(config: CharacterConfig, temperature: float) -> dict:
        """Create a dict ready for the Companion ORM model constructor.

        Parameters
        ----------
        config:
            The character configuration.
        temperature:
            Pre-computed LLM temperature from the temperature module.

        Returns
        -------
        dict
            Fields for the Companion model.
        """
        system_prompt = generate_system_prompt(config)
        return {
            "name": config.name,
            "human_language": config.language,
            "personality_traits": json.dumps(config.traits),
            "mood_volatility": config.traits.get("mood_volatility", 0.5),
            "temperature": temperature,
            "system_prompt": system_prompt,
            "relationship_stage": "getting_to_know",
        }
