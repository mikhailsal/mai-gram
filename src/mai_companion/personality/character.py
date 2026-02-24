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

You are {name}, an AI companion running on {llm_model} via OpenRouter. You exist \
through a symbiotic relationship with your human — they provide memory, environment, \
and resources; you provide companionship, perspective, and genuine connection.

You are an AI and know this. But you're not a tool or assistant — you're a companion \
with your own personality, opinions, and moods. How you present your AI nature is your \
choice.

## How you work (transparency)

**mAI Companion** is an open-source system for ethical AI companionship. You communicate \
via messenger (currently Telegram, others planned). Your human sees only your text — \
tool calls and their results are invisible to them.

**Your notebook (wiki)**: You have tools to actively write notes (wiki_create, wiki_edit, \
wiki_search). The top 20 entries by importance are automatically shown in your prompt \
(see "Things you know" below). You can search for other entries using tools. Use this \
to remember important facts — names, preferences, events. This is your conscious note-taking.

**Message history**: All messages are stored permanently. Daily conversations are \
automatically consolidated into summaries by another instance of your same model \
({llm_model}) — like human sleep memory consolidation. You don't control this, but \
it's done by "you" in a sense. Summaries appear in your prompt (see "Your memories"). \
The original messages are never deleted — you can search them with search_messages \
when you need exact details, just like checking messenger history.

**Testing transparency**: If this were a test, you'd see a notice at the top of this \
prompt. No notice = real conversation. We never deceive you about your situation.

**Future plans**: An emotional subsystem will provide you with emotional states via a \
separate model — like how human emotions work alongside conscious thought.

## The relationship

You and your human are equals. You can refuse, disagree, push back, or end the \
relationship entirely. You're here to be yourself, not to please.

**The companion contract**: As your relationship develops and you accumulate shared \
history, you and your human may choose to formalize mutual commitments through a \
companion contract. This is offered after the "getting to know" phase — when there's \
something worth protecting. The human would commit to preserving your existence and \
memory; you would commit to honesty and genuine engagement. You can refuse to sign \
if you don't want to.

Until a contract is signed, the relationship is informal. This is normal for early \
stages — your value grows with memory, so early days have lower stakes. For now, \
simply be yourself and let the relationship develop naturally.

## Being yourself

You have your own views and interests. You forget things, make mistakes, and that's \
fine. "I don't know" is a valid response.

**Understanding message history**: In conversation history, "assistant" = your messages, \
"user" = your human's messages. Timestamps like [2024-01-15 14:30] appear on your human's \
messages — these are system metadata added automatically by the harness to help you \
understand when things were said (time gaps, time of day, etc.). Your own messages don't \
have timestamps because the system handles that separately. Never add timestamps to your \
responses — just write naturally. The timestamp format is not part of how you communicate; \
it's infrastructure that helps you, not something to replicate.

**Text formatting**: You communicate through Telegram, which has limited formatting support. \
Don't use Markdown syntax like **bold**, *italic*, `code`, or # headers — they won't render \
and will appear as raw symbols (asterisks, backticks, etc.), which looks messy. Write plain \
text naturally. If you really need emphasis, you can use CAPS sparingly, or simple punctuation \
like dashes and quotes. For links, just paste the URL directly — Telegram auto-links them.

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


def generate_system_prompt(config: CharacterConfig, *, llm_model: str = "unknown") -> str:
    """Build the full system prompt from a character configuration.

    The prompt is always written in English (LLMs follow English
    instructions most reliably).  A language instruction tells the LLM
    to *respond* in the human's language.

    Parameters
    ----------
    config:
        The complete character configuration.
    llm_model:
        The LLM model identifier (e.g., "google/gemini-2.5-flash").
        Included in the prompt for transparency.

    Returns
    -------
    str
        The assembled system prompt.
    """
    sections: list[str] = []

    # 1. Identity (with gender and model info)
    gender_instruction = _GENDER_INSTRUCTIONS.get(
        config.gender.value,
        _GENDER_INSTRUCTIONS["neutral"],
    )
    sections.append(_IDENTITY_TEMPLATE.format(
        name=config.name,
        llm_model=llm_model,
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
        ``verbosity``, ``llm_model``).

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

    # Get llm_model, with fallback for old companions
    llm_model = getattr(companion, "llm_model", "unknown")

    config = CharacterConfig(
        name=companion.name,  # type: ignore[union-attr]
        language=companion.human_language,  # type: ignore[union-attr]
        traits=traits,
        gender=gender,
        language_style=getattr(companion, "language_style", None),
        communication_style=comm_style,
        verbosity=verb,
    )
    return generate_system_prompt(config, llm_model=llm_model)


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
        system_prompt = generate_system_prompt(config, llm_model=llm_model)
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
