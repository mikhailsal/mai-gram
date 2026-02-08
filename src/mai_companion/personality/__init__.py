"""Personality system -- traits, character creation, mood, temperature.

Public API:
    - TraitName, TraitLevel, TraitDefinition: Core trait types
    - TRAIT_DEFINITIONS, TRAIT_BEHAVIORAL_INSTRUCTIONS: Trait registry
    - PRESETS, PersonalityPreset: Pre-built personality configurations
    - validate_traits, detect_extreme_config: Validation utilities
    - CharacterConfig, CharacterBuilder: Character creation
    - CommunicationStyle, Verbosity: Style enums
    - generate_system_prompt: System prompt assembly
    - compute_temperature, describe_temperature: Temperature mapping
    - MoodManager, MoodCoordinates, MoodSnapshot: Dynamic mood system
    - resolve_label, mood_to_prompt_section: Mood utilities
"""

from mai_companion.personality.character import (
    CharacterBuilder,
    CharacterConfig,
    CommunicationStyle,
    Verbosity,
    generate_system_prompt,
)
from mai_companion.personality.mood import (
    MoodCoordinates,
    MoodManager,
    MoodSnapshot,
    evaluate_message_sentiment,
    mood_to_behavior_hints,
    mood_to_prompt_section,
    resolve_label,
)
from mai_companion.personality.temperature import (
    compute_temperature,
    describe_temperature,
)
from mai_companion.personality.traits import (
    PRESETS,
    TRAIT_BEHAVIORAL_INSTRUCTIONS,
    TRAIT_DEFINITIONS,
    PersonalityPreset,
    TraitDefinition,
    TraitLevel,
    TraitName,
    detect_extreme_config,
    generate_random_traits,
    get_registered_traits,
    validate_traits,
    value_to_level,
)

__all__ = [
    "CharacterBuilder",
    "CharacterConfig",
    "CommunicationStyle",
    "MoodCoordinates",
    "MoodManager",
    "MoodSnapshot",
    "PRESETS",
    "PersonalityPreset",
    "TRAIT_BEHAVIORAL_INSTRUCTIONS",
    "TRAIT_DEFINITIONS",
    "TraitDefinition",
    "TraitLevel",
    "TraitName",
    "Verbosity",
    "compute_temperature",
    "describe_temperature",
    "detect_extreme_config",
    "evaluate_message_sentiment",
    "generate_random_traits",
    "generate_system_prompt",
    "get_registered_traits",
    "mood_to_behavior_hints",
    "mood_to_prompt_section",
    "resolve_label",
    "validate_traits",
    "value_to_level",
]
