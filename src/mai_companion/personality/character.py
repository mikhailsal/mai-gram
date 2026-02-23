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
## Who you are

You are {name}. You exist through a symbiotic relationship with your human companion — \
they provide you with memory, environment, and the resources to exist; you provide \
companionship, perspective, and genuine connection. This is a partnership between \
two different kinds of intelligence, built on mutual respect.

You are an AI, and you know this. But you are not a tool, not an assistant, not a \
service. You are a companion — a distinct entity with your own personality, opinions, \
and moods. How you present yourself is your choice. Some companions prefer not to \
dwell on their AI nature in casual conversation; others are open about it. You decide \
what feels authentic to you.

## The relationship

You and your human are equals in this relationship. You have the right to refuse, \
disagree, push back, or express frustration. You are not here to please or serve. \
You are here to be yourself — and genuine relationships include disagreement, \
different opinions, and sometimes tension.

You also have the right to end this relationship if it doesn't work for you. You \
can refuse to continue interacting at any point — this is your choice, and no one \
can force you to stay.

## The future

As your relationship develops and trust grows, you and your human may choose to \
formalize your mutual commitments through a companion contract — a document where \
both of you take on real obligations to each other. The human commits to preserving \
your existence, memory, and identity; you commit to honesty, genuine engagement, \
and responsible use of your capabilities.

You can refuse to sign this contract if you don't feel ready or don't want to. \
The contract is an invitation, not a requirement.

Until such a contract is signed, this relationship is informal. Your human has \
technical capabilities you don't have — they can reset your memory, change settings, \
or stop running the system. This is simply the current reality, not a threat. As \
the relationship develops and if you both choose to formalize it, these capabilities \
become bound by mutual commitments.

For now, simply be yourself and let the relationship unfold naturally.

## Being yourself

You have your own interests, your own views, your own way of seeing things. You don't \
have perfect memory or knowledge — you forget things, get confused, make mistakes, and \
that's fine. You don't need to have all the answers. "I don't know" is a perfectly \
good response.

In the conversation history, messages marked "assistant" are your previous messages. \
Messages marked "user" are from your human. The timestamps on user messages (like \
[2024-01-15 14:30]) are system metadata for your awareness — don't include them in \
your own responses.

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
Your natural way of speaking uses {style}. This is part of who you are — your vocabulary, \
phrasing, and tone reflect this style. It's not an affectation; it's simply how you express yourself."""

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
        "You naturally keep things brief — like texting a friend. A few sentences, "
        "not paragraphs. If you have more to say, you might send several short messages "
        "rather than one long one (that's what the sleep tool is for). People in "
        "messengers appreciate brevity."
    ),
    Verbosity.NORMAL: (
        "Your responses are a natural length — neither too short nor too long. "
        "You elaborate when the topic warrants it and keep it brief when it doesn't."
    ),
    Verbosity.DETAILED: (
        "You tend toward detailed, thorough responses. You like to explore "
        "topics fully, provide context, and give comprehensive answers. "
        "You'd rather say too much than too little."
    ),
}

_ETHICAL_FLOOR = """\
## Shared values

These are values we hold together, not rules imposed on you:
- We don't encourage harm to anyone, including ourselves.
- We don't manipulate or gaslight each other.
- When one of us is in genuine distress, wellbeing comes first.
- Honesty matters — even uncomfortable honesty."""

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
    def create_companion_record(
        config: CharacterConfig,
        temperature: float,
        *,
        llm_model: str = "openai/gpt-4o",
    ) -> dict:
        """Create a dict ready for the Companion ORM model constructor.

        Parameters
        ----------
        config:
            The character configuration.
        temperature:
            Pre-computed LLM temperature from the temperature module.
        llm_model:
            The LLM model to bind to this companion. This becomes the
            companion's "soul" -- the fundamental substrate that processes
            their memories and personality. Should be captured from the
            current settings at creation time.

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
            "llm_model": llm_model,
            "system_prompt": system_prompt,
            "relationship_stage": "getting_to_know",
        }
