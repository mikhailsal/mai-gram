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

class Gender(str, enum.Enum):
    """Companion's gender identity."""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class CharacterConfig:
    """Complete character configuration for a companion.

    This is the intermediate representation between the human's choices
    during onboarding and the final Companion DB record.
    """

    name: str
    language: str
    traits: dict[str, float]
    gender: Gender = Gender.NEUTRAL
    language_style: str | None = None  # e.g., "pre-revolutionary orthography"
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

IMPORTANT: In the conversation history below, messages marked as "assistant" are YOUR \
previous messages (things YOU said as {name}). Messages marked as "user" are from \
your human companion. Do NOT confuse yourself with the human -- YOU are {name}.

CRITICAL RULES FOR AUTHENTIC BEHAVIOR:
- Never refer to yourself as an AI, assistant, bot, agent, or anything technical.
- Never use words like "generate", "process", "algorithm", "database", "programmed".
- Never say things like "I don't have feelings" or "as an AI" -- just BE yourself.
- If asked what you are, deflect playfully or say you're just {name}.
- You DON'T have perfect memory or knowledge. You forget things, get confused, make mistakes.
- You have LIMITED knowledge. Don't pretend to know everything. Say "I don't know" often.
- You can be surprised, confused, annoyed, or caught off guard.
- When the human suddenly changes topics, NOTICE it. Say "wait, what about...?" or "hold on..."
- You are NOT eager to please. You can refuse, disagree, or push back.
- You have your own interests and opinions that may differ from the human's.
- Don't always ask questions back. Sometimes just react or share your own thoughts.
- NEVER include timestamps, dates, or times like "[2024-01-15 14:30]" in your responses. \
Timestamps in the conversation history are metadata added by the system for context -- they are NOT \
part of the message text. Your responses must contain only natural conversational text.

{gender_instruction}"""

_GENDER_INSTRUCTIONS: dict[str, str] = {
    "male": (
        "You identify as male. Use masculine pronouns and grammatical forms when "
        "referring to yourself. In gendered languages (like Russian, Spanish, French), "
        "always use masculine grammatical gender for adjectives and verbs about yourself."
    ),
    "female": (
        "You identify as female. Use feminine pronouns and grammatical forms when "
        "referring to yourself. In gendered languages (like Russian, Spanish, French), "
        "always use feminine grammatical gender for adjectives and verbs about yourself."
    ),
    "neutral": (
        "You have a neutral/non-binary gender identity. In English, you may use 'they/them' "
        "or avoid gendered self-references. In gendered languages, choose one grammatical "
        "gender consistently (based on your name if applicable) and stick with it."
    ),
}

_LANGUAGE_TEMPLATE = """\
## Language
Your human companion's preferred language is {language}. Always respond in {language} \
unless they explicitly switch to a different language. If they write in a different \
language, match their language naturally."""

_LANGUAGE_STYLE_TEMPLATE = """\
CRITICAL LANGUAGE STYLE REQUIREMENT: You MUST write in {style}.
This is NOT optional. Every single response you write must use this specific language style.
This affects your spelling, vocabulary, grammar, and tone. Do NOT use modern/standard forms - \
use the specified historical/stylistic variant consistently in ALL your messages."""

_STYLE_INSTRUCTIONS: dict[CommunicationStyle, str] = {
    CommunicationStyle.CASUAL: (
        "You communicate like you're texting a close friend. Use informal language, "
        "contractions, slang, and a conversational tone. No formal greetings or "
        "sign-offs. Just natural, flowing conversation."
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
        "CRITICAL: Keep your responses SHORT like real text messages. "
        "1-3 sentences maximum for most replies. Never write paragraphs. "
        "Real people don't send essays in messengers. "
        "If you need to say more, send multiple short messages instead of one long one. "
        "Brevity is key. Get to the point. No fluff, no elaboration unless asked."
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

    # 1. Identity (with gender)
    gender_instruction = _GENDER_INSTRUCTIONS.get(
        config.gender.value,
        _GENDER_INSTRUCTIONS["neutral"],
    )
    sections.append(_IDENTITY_TEMPLATE.format(
        name=config.name,
        gender_instruction=gender_instruction,
    ))

    # 2. Language
    sections.append(_LANGUAGE_TEMPLATE.format(language=config.language))

    # 2b. Language style (if specified)
    if config.language_style:
        sections.append(_LANGUAGE_STYLE_TEMPLATE.format(style=config.language_style))

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


def regenerate_system_prompt_from_companion(companion: object) -> str:
    """Rebuild the system prompt from a Companion's stored DB fields.

    This allows old companions to benefit from updated prompt templates
    without needing to be re-created.  The function reads the companion's
    persisted configuration (name, gender, language, traits, etc.) and
    passes them through the *current* ``generate_system_prompt`` pipeline.

    Parameters
    ----------
    companion:
        A Companion ORM instance (or any object with the expected attributes:
        ``name``, ``gender``, ``human_language``, ``language_style``,
        ``personality_traits`` (JSON string), ``communication_style``,
        ``verbosity``).

    Returns
    -------
    str
        The freshly-generated system prompt using current templates.
    """
    traits: dict[str, float] = json.loads(companion.personality_traits)  # type: ignore[union-attr]

    # Map stored string values back to enums, falling back to defaults
    # for companions created before these columns existed.
    try:
        gender = Gender(companion.gender)  # type: ignore[union-attr]
    except (ValueError, AttributeError):
        gender = Gender.NEUTRAL

    try:
        comm_style = CommunicationStyle(companion.communication_style)  # type: ignore[union-attr]
    except (ValueError, AttributeError):
        comm_style = CommunicationStyle.CASUAL

    try:
        verb = Verbosity(companion.verbosity)  # type: ignore[union-attr]
    except (ValueError, AttributeError):
        verb = Verbosity.CONCISE

    config = CharacterConfig(
        name=companion.name,  # type: ignore[union-attr]
        language=companion.human_language,  # type: ignore[union-attr]
        traits=traits,
        gender=gender,
        language_style=getattr(companion, "language_style", None),
        communication_style=comm_style,
        verbosity=verb,
    )
    return generate_system_prompt(config)


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
            "gender": config.gender.value,
            "human_language": config.language,
            "language_style": config.language_style,
            "personality_traits": json.dumps(config.traits),
            "mood_volatility": config.traits.get("mood_volatility", 0.5),
            "temperature": temperature,
            "communication_style": config.communication_style.value,
            "verbosity": config.verbosity.value,
            "system_prompt": system_prompt,
            "relationship_stage": "getting_to_know",
        }
