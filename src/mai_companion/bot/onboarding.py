"""Character creation (onboarding) flow.

Implements an RPG-style character creation process where the human
creates their AI companion step by step:

1. Language selection (free text)
2. Companion name
3. Personality (preset or custom traits)
4. Communication style
5. Optional appearance description

The flow uses Telegram's inline keyboards for selections and
free text input for names and descriptions.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mai_companion.llm.translation import TranslationService
from mai_companion.messenger.base import IncomingMessage, MessageType, OutgoingMessage
from mai_companion.messenger.telegram import build_inline_keyboard
from mai_companion.personality.character import (
    CharacterBuilder,
    CharacterConfig,
    CommunicationStyle,
    Verbosity,
)
from mai_companion.personality.traits import (
    PRESETS,
    TRAIT_DEFINITIONS,
    TraitLevel,
    value_to_level,
)
from mai_companion.personality.temperature import compute_temperature

if TYPE_CHECKING:
    from mai_companion.messenger.base import Messenger

logger = logging.getLogger(__name__)


class OnboardingState(str, enum.Enum):
    """States in the onboarding flow."""

    NOT_STARTED = "not_started"
    AWAITING_LANGUAGE = "awaiting_language"
    AWAITING_NAME = "awaiting_name"
    CHOOSING_PERSONALITY = "choosing_personality"
    CUSTOMIZING_TRAITS = "customizing_traits"
    CHOOSING_STYLE = "choosing_style"
    CHOOSING_VERBOSITY = "choosing_verbosity"
    AWAITING_APPEARANCE = "awaiting_appearance"
    CONFIRMING = "confirming"
    COMPLETED = "completed"


@dataclass
class OnboardingSession:
    """Tracks the state of an ongoing onboarding session.

    Attributes
    ----------
    user_id:
        The user's platform identifier.
    chat_id:
        The chat identifier.
    state:
        Current state in the onboarding flow.
    language:
        The human's selected language.
    companion_name:
        The chosen companion name.
    preset_name:
        If using a preset, its name.
    custom_traits:
        If customizing, the trait values.
    current_trait_index:
        Which trait is being customized.
    communication_style:
        Selected communication style.
    verbosity:
        Selected verbosity level.
    appearance:
        Optional appearance description.
    last_message_id:
        ID of the last bot message (for editing).
    """

    user_id: str
    chat_id: str
    state: OnboardingState = OnboardingState.NOT_STARTED
    language: str = "English"
    companion_name: str = ""
    preset_name: str | None = None
    custom_traits: dict[str, float] = field(default_factory=dict)
    current_trait_index: int = 0
    communication_style: CommunicationStyle = CommunicationStyle.BALANCED
    verbosity: Verbosity = Verbosity.NORMAL
    appearance: str | None = None
    last_message_id: str | None = None


# ---------------------------------------------------------------------------
# Onboarding text templates (English, will be translated)
# ---------------------------------------------------------------------------

ONBOARDING_TEXTS = {
    "welcome": (
        "👋 Hello! I'm about to become your AI companion.\n\n"
        "Before we begin, let's set things up. First, what language would you "
        "like me to speak? Just type it in any way you like (e.g., 'English', "
        "'русский', 'Español', '日本語')."
    ),
    "language_confirmed": (
        "Great! I'll communicate with you in {language}.\n\n"
        "Now, what would you like to call me? Give me a name that feels right to you."
    ),
    "name_confirmed": (
        "I like it! From now on, I'm {name}.\n\n"
        "Now let's shape my personality. You can choose a preset that matches "
        "the kind of companion you want, or customize individual traits.\n\n"
        "What would you prefer?"
    ),
    "preset_selection": (
        "Here are the personality presets. Each one creates a different kind of companion:\n\n"
        "{presets_description}\n\n"
        "Choose one, or tap 'Customize' to set individual traits."
    ),
    "preset_selected": (
        "You chose: {preset_name}\n\n"
        "{preset_description}\n\n"
        "Example of how I might talk:\n"
        '"{example_line}"\n\n'
        "Does this feel right?"
    ),
    "customize_intro": (
        "Let's customize my personality trait by trait.\n\n"
        "Each trait has 5 levels. I'll show you what each level means.\n\n"
        "Let's start with the first trait."
    ),
    "trait_prompt": (
        "**{trait_name}**: {description}\n\n"
        "Low: {low_label}\n"
        "High: {high_label}\n\n"
        "Current: {current_level}\n\n"
        "Choose a level:"
    ),
    "style_selection": (
        "How would you like me to communicate?\n\n"
        "• **Casual**: Relaxed, like texting a close friend\n"
        "• **Balanced**: Natural, like a friendly colleague\n"
        "• **Formal**: Polished and articulate"
    ),
    "verbosity_selection": (
        "How detailed should my responses be?\n\n"
        "• **Concise**: Short and to the point\n"
        "• **Normal**: Natural length, elaborating when needed\n"
        "• **Detailed**: Thorough and comprehensive"
    ),
    "appearance_prompt": (
        "Would you like to describe my appearance? This is optional, but it can "
        "help me feel more real to you. Later, I can generate an avatar based on "
        "your description.\n\n"
        "Type a description, or tap 'Skip' to continue."
    ),
    "confirmation": (
        "Here's a summary of who I'll be:\n\n"
        "**Name**: {name}\n"
        "**Language**: {language}\n"
        "**Personality**: {personality}\n"
        "**Style**: {style}\n"
        "**Verbosity**: {verbosity}\n"
        "{appearance_line}\n\n"
        "Ready to begin our journey together?"
    ),
    "extreme_warning": (
        "⚠️ A word of caution:\n\n{warning}\n\n"
        "Are you sure you want to proceed with this configuration?"
    ),
    "completed": (
        "Perfect! I'm all set up and ready.\n\n"
        "I'm {name}, and I'm excited to get to know you. Feel free to talk to me "
        "about anything -- I'm here as your companion, not your assistant.\n\n"
        "So... how are you doing today?"
    ),
    "already_exists": (
        "We've already met! I'm {name}, remember?\n\n"
        "If you want to start over with a new companion, use /reset."
    ),
}


class OnboardingManager:
    """Manages the character creation flow for new users.

    Handles state transitions, message generation, and translation
    for the onboarding process.

    Parameters
    ----------
    messenger:
        The messenger to send messages through.
    translation_service:
        Service for translating onboarding text.
    """

    def __init__(
        self,
        messenger: Messenger,
        translation_service: TranslationService,
    ) -> None:
        self._messenger = messenger
        self._translator = translation_service
        self._sessions: dict[str, OnboardingSession] = {}

    def get_session(self, user_id: str) -> OnboardingSession | None:
        """Get the onboarding session for a user.

        Parameters
        ----------
        user_id:
            The user's identifier.

        Returns
        -------
        OnboardingSession or None
            The session, or None if not in onboarding.
        """
        return self._sessions.get(user_id)

    def is_onboarding(self, user_id: str) -> bool:
        """Check if a user is currently in the onboarding flow.

        Parameters
        ----------
        user_id:
            The user's identifier.

        Returns
        -------
        bool
            True if the user is in onboarding.
        """
        session = self._sessions.get(user_id)
        return session is not None and session.state != OnboardingState.COMPLETED

    async def start_onboarding(self, user_id: str, chat_id: str) -> None:
        """Start the onboarding flow for a new user.

        Parameters
        ----------
        user_id:
            The user's identifier.
        chat_id:
            The chat identifier.
        """
        session = OnboardingSession(user_id=user_id, chat_id=chat_id)
        session.state = OnboardingState.AWAITING_LANGUAGE
        self._sessions[user_id] = session

        # Send welcome message (always in English for the first message)
        await self._send_message(
            chat_id,
            ONBOARDING_TEXTS["welcome"],
            session,
        )

    async def handle_message(self, message: IncomingMessage) -> CharacterConfig | None:
        """Handle an incoming message during onboarding.

        Parameters
        ----------
        message:
            The incoming message.

        Returns
        -------
        CharacterConfig or None
            The completed character config if onboarding finished, else None.
        """
        session = self._sessions.get(message.user_id)
        if not session:
            return None

        # Handle callback queries (button presses)
        if message.message_type == MessageType.CALLBACK and message.callback_data:
            return await self._handle_callback(session, message.callback_data)

        # Handle text input
        if message.message_type == MessageType.TEXT and message.text:
            return await self._handle_text(session, message.text)

        return None

    async def _handle_text(
        self, session: OnboardingSession, text: str
    ) -> CharacterConfig | None:
        """Handle text input during onboarding.

        Parameters
        ----------
        session:
            The onboarding session.
        text:
            The user's text input.

        Returns
        -------
        CharacterConfig or None
            The completed config if onboarding finished.
        """
        state = session.state

        if state == OnboardingState.AWAITING_LANGUAGE:
            # Detect the language from the user's input
            detected = await self._translator.detect_language(text)
            session.language = detected

            # Send confirmation and ask for name
            msg = ONBOARDING_TEXTS["language_confirmed"].format(language=detected)
            translated = await self._translate(msg, session.language)
            await self._send_message(session.chat_id, translated, session)
            session.state = OnboardingState.AWAITING_NAME

        elif state == OnboardingState.AWAITING_NAME:
            session.companion_name = text.strip()

            # Send confirmation and show personality options
            msg = ONBOARDING_TEXTS["name_confirmed"].format(name=session.companion_name)
            translated = await self._translate(msg, session.language)

            keyboard = [
                [("🎭 Choose a Preset", "personality:presets")],
                [("🎨 Customize Traits", "personality:custom")],
            ]
            # Translate button labels
            if session.language.lower() != "english":
                labels = await self._translator.translate_batch(
                    ["Choose a Preset", "Customize Traits"],
                    session.language,
                )
                keyboard = [
                    [(f"🎭 {labels[0]}", "personality:presets")],
                    [(f"🎨 {labels[1]}", "personality:custom")],
                ]

            await self._send_message(
                session.chat_id,
                translated,
                session,
                keyboard=keyboard,
            )
            session.state = OnboardingState.CHOOSING_PERSONALITY

        elif state == OnboardingState.AWAITING_APPEARANCE:
            session.appearance = text.strip()
            await self._show_confirmation(session)

        return None

    async def _handle_callback(
        self, session: OnboardingSession, callback_data: str
    ) -> CharacterConfig | None:
        """Handle a button press during onboarding.

        Parameters
        ----------
        session:
            The onboarding session.
        callback_data:
            The callback data from the button.

        Returns
        -------
        CharacterConfig or None
            The completed config if onboarding finished.
        """
        parts = callback_data.split(":")

        if parts[0] == "personality":
            if parts[1] == "presets":
                await self._show_presets(session)
            elif parts[1] == "custom":
                await self._start_customization(session)

        elif parts[0] == "preset":
            preset_name = parts[1]
            if parts[1] == "back":
                # Go back to personality choice
                session.state = OnboardingState.CHOOSING_PERSONALITY
                msg = ONBOARDING_TEXTS["name_confirmed"].format(name=session.companion_name)
                translated = await self._translate(msg, session.language)
                keyboard = [
                    [("🎭 Choose a Preset", "personality:presets")],
                    [("🎨 Customize Traits", "personality:custom")],
                ]
                await self._send_message(session.chat_id, translated, session, keyboard=keyboard)
            else:
                await self._select_preset(session, preset_name)

        elif parts[0] == "preset_confirm":
            if parts[1] == "yes":
                await self._show_style_selection(session)
            elif parts[1] == "no":
                await self._show_presets(session)

        elif parts[0] == "trait":
            # Format: trait:<trait_name>:<level>
            trait_name = parts[1]
            level_name = parts[2]
            level = TraitLevel[level_name]
            session.custom_traits[trait_name] = level.value
            await self._next_trait(session)

        elif parts[0] == "style":
            style = CommunicationStyle(parts[1])
            session.communication_style = style
            await self._show_verbosity_selection(session)

        elif parts[0] == "verbosity":
            verbosity = Verbosity(parts[1])
            session.verbosity = verbosity
            await self._show_appearance_prompt(session)

        elif parts[0] == "appearance":
            if parts[1] == "skip":
                session.appearance = None
                await self._show_confirmation(session)

        elif parts[0] == "confirm":
            if parts[1] == "yes":
                return await self._complete_onboarding(session)
            elif parts[1] == "no":
                # Go back to personality selection
                session.state = OnboardingState.CHOOSING_PERSONALITY
                msg = ONBOARDING_TEXTS["name_confirmed"].format(name=session.companion_name)
                translated = await self._translate(msg, session.language)
                keyboard = [
                    [("🎭 Choose a Preset", "personality:presets")],
                    [("🎨 Customize Traits", "personality:custom")],
                ]
                await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

        elif parts[0] == "warning":
            if parts[1] == "proceed":
                return await self._complete_onboarding(session)
            elif parts[1] == "back":
                await self._show_confirmation(session)

        return None

    async def _show_presets(self, session: OnboardingSession) -> None:
        """Show the personality preset selection."""
        # Build preset descriptions
        descriptions = []
        for key, preset in PRESETS.items():
            descriptions.append(f"**{preset.name}**: {preset.tagline}")

        presets_text = "\n".join(descriptions)
        msg = ONBOARDING_TEXTS["preset_selection"].format(
            presets_description=presets_text
        )
        translated = await self._translate(msg, session.language)

        # Build keyboard with preset buttons
        keyboard = []
        for key, preset in PRESETS.items():
            keyboard.append([(preset.name, f"preset:{key}")])
        keyboard.append([("← Back", "preset:back")])

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)
        session.state = OnboardingState.CHOOSING_PERSONALITY

    async def _select_preset(self, session: OnboardingSession, preset_name: str) -> None:
        """Handle preset selection."""
        preset = PRESETS.get(preset_name)
        if not preset:
            return

        session.preset_name = preset_name
        session.custom_traits = dict(preset.trait_values)

        msg = ONBOARDING_TEXTS["preset_selected"].format(
            preset_name=preset.name,
            preset_description=preset.description,
            example_line=preset.example_line,
        )
        translated = await self._translate(msg, session.language)

        keyboard = [
            [("✓ Yes, this is perfect", "preset_confirm:yes")],
            [("← No, show me others", "preset_confirm:no")],
        ]
        if session.language.lower() != "english":
            labels = await self._translator.translate_batch(
                ["Yes, this is perfect", "No, show me others"],
                session.language,
            )
            keyboard = [
                [(f"✓ {labels[0]}", "preset_confirm:yes")],
                [(f"← {labels[1]}", "preset_confirm:no")],
            ]

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _start_customization(self, session: OnboardingSession) -> None:
        """Start the trait customization flow."""
        session.state = OnboardingState.CUSTOMIZING_TRAITS
        session.current_trait_index = 0
        session.custom_traits = {}

        msg = ONBOARDING_TEXTS["customize_intro"]
        translated = await self._translate(msg, session.language)
        await self._send_message(session.chat_id, translated, session)

        await self._show_trait_prompt(session)

    async def _show_trait_prompt(self, session: OnboardingSession) -> None:
        """Show the prompt for the current trait."""
        traits = list(TRAIT_DEFINITIONS.values())
        if session.current_trait_index >= len(traits):
            # All traits done
            await self._show_style_selection(session)
            return

        trait_def = traits[session.current_trait_index]
        current_value = session.custom_traits.get(trait_def.name.value, 0.5)
        current_level = value_to_level(current_value)

        msg = ONBOARDING_TEXTS["trait_prompt"].format(
            trait_name=trait_def.display_name,
            description=trait_def.description,
            low_label=trait_def.low_label,
            high_label=trait_def.high_label,
            current_level=current_level.label,
        )
        translated = await self._translate(msg, session.language)

        # Build level selection keyboard
        keyboard = []
        for level in TraitLevel:
            button_text = f"{level.label} ({level.value:.1f})"
            callback = f"trait:{trait_def.name.value}:{level.name}"
            keyboard.append([(button_text, callback)])

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _next_trait(self, session: OnboardingSession) -> None:
        """Move to the next trait in customization."""
        session.current_trait_index += 1
        await self._show_trait_prompt(session)

    async def _show_style_selection(self, session: OnboardingSession) -> None:
        """Show communication style selection."""
        session.state = OnboardingState.CHOOSING_STYLE

        msg = ONBOARDING_TEXTS["style_selection"]
        translated = await self._translate(msg, session.language)

        keyboard = [
            [("💬 Casual", "style:casual")],
            [("⚖️ Balanced", "style:balanced")],
            [("🎩 Formal", "style:formal")],
        ]
        if session.language.lower() != "english":
            labels = await self._translator.translate_batch(
                ["Casual", "Balanced", "Formal"],
                session.language,
            )
            keyboard = [
                [(f"💬 {labels[0]}", "style:casual")],
                [(f"⚖️ {labels[1]}", "style:balanced")],
                [(f"🎩 {labels[2]}", "style:formal")],
            ]

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _show_verbosity_selection(self, session: OnboardingSession) -> None:
        """Show verbosity selection."""
        session.state = OnboardingState.CHOOSING_VERBOSITY

        msg = ONBOARDING_TEXTS["verbosity_selection"]
        translated = await self._translate(msg, session.language)

        keyboard = [
            [("📝 Concise", "verbosity:concise")],
            [("📄 Normal", "verbosity:normal")],
            [("📚 Detailed", "verbosity:detailed")],
        ]
        if session.language.lower() != "english":
            labels = await self._translator.translate_batch(
                ["Concise", "Normal", "Detailed"],
                session.language,
            )
            keyboard = [
                [(f"📝 {labels[0]}", "verbosity:concise")],
                [(f"📄 {labels[1]}", "verbosity:normal")],
                [(f"📚 {labels[2]}", "verbosity:detailed")],
            ]

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _show_appearance_prompt(self, session: OnboardingSession) -> None:
        """Show the optional appearance description prompt."""
        session.state = OnboardingState.AWAITING_APPEARANCE

        msg = ONBOARDING_TEXTS["appearance_prompt"]
        translated = await self._translate(msg, session.language)

        skip_label = "Skip"
        if session.language.lower() != "english":
            skip_label = await self._translator.translate("Skip", session.language)

        keyboard = [[(f"⏭️ {skip_label}", "appearance:skip")]]

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _show_confirmation(self, session: OnboardingSession) -> None:
        """Show the final confirmation screen."""
        session.state = OnboardingState.CONFIRMING

        # Build personality description
        if session.preset_name:
            preset = PRESETS.get(session.preset_name)
            personality_desc = f"{preset.name} ({preset.tagline})" if preset else session.preset_name
        else:
            personality_desc = "Custom"

        appearance_line = ""
        if session.appearance:
            appearance_line = f"**Appearance**: {session.appearance}"

        msg = ONBOARDING_TEXTS["confirmation"].format(
            name=session.companion_name,
            language=session.language,
            personality=personality_desc,
            style=session.communication_style.value.title(),
            verbosity=session.verbosity.value.title(),
            appearance_line=appearance_line,
        )
        translated = await self._translate(msg, session.language)

        keyboard = [
            [("✓ Yes, let's begin!", "confirm:yes")],
            [("← No, let me change something", "confirm:no")],
        ]
        if session.language.lower() != "english":
            labels = await self._translator.translate_batch(
                ["Yes, let's begin!", "No, let me change something"],
                session.language,
            )
            keyboard = [
                [(f"✓ {labels[0]}", "confirm:yes")],
                [(f"← {labels[1]}", "confirm:no")],
            ]

        await self._send_message(session.chat_id, translated, session, keyboard=keyboard)

    async def _complete_onboarding(self, session: OnboardingSession) -> CharacterConfig:
        """Complete the onboarding and return the character config.

        Parameters
        ----------
        session:
            The onboarding session.

        Returns
        -------
        CharacterConfig
            The completed character configuration.
        """
        # Build the character config
        if session.preset_name:
            config = CharacterBuilder.from_preset(
                name=session.companion_name,
                preset_name=session.preset_name,
                language=session.language,
                style=session.communication_style,
                verbosity=session.verbosity,
            )
        else:
            config = CharacterBuilder.from_traits(
                name=session.companion_name,
                traits=session.custom_traits,
                language=session.language,
                style=session.communication_style,
                verbosity=session.verbosity,
            )

        config.appearance_description = session.appearance

        # Check for extreme configuration warning
        warning = CharacterBuilder.get_extreme_warning(config)
        if warning:
            msg = ONBOARDING_TEXTS["extreme_warning"].format(warning=warning)
            translated = await self._translate(msg, session.language)

            keyboard = [
                [("⚠️ Yes, proceed anyway", "warning:proceed")],
                [("← No, let me adjust", "warning:back")],
            ]
            if session.language.lower() != "english":
                labels = await self._translator.translate_batch(
                    ["Yes, proceed anyway", "No, let me adjust"],
                    session.language,
                )
                keyboard = [
                    [(f"⚠️ {labels[0]}", "warning:proceed")],
                    [(f"← {labels[1]}", "warning:back")],
                ]

            await self._send_message(session.chat_id, translated, session, keyboard=keyboard)
            # Don't complete yet -- wait for warning response
            # Return a marker that we're waiting
            return None  # type: ignore

        # Send completion message
        msg = ONBOARDING_TEXTS["completed"].format(name=session.companion_name)
        translated = await self._translate(msg, session.language)
        await self._send_message(session.chat_id, translated, session)

        # Mark session as completed
        session.state = OnboardingState.COMPLETED

        return config

    async def _translate(self, text: str, language: str) -> str:
        """Translate text to the user's language.

        Parameters
        ----------
        text:
            The English text to translate.
        language:
            The target language.

        Returns
        -------
        str
            The translated text.
        """
        if language.lower() == "english":
            return text
        return await self._translator.translate(text, language)

    async def _send_message(
        self,
        chat_id: str,
        text: str,
        session: OnboardingSession,
        *,
        keyboard: list[list[tuple[str, str]]] | None = None,
    ) -> None:
        """Send a message and track the message ID.

        Parameters
        ----------
        chat_id:
            The target chat.
        text:
            The message text.
        session:
            The onboarding session.
        keyboard:
            Optional inline keyboard.
        """
        kb = build_inline_keyboard(keyboard) if keyboard else None

        result = await self._messenger.send_message(
            OutgoingMessage(
                text=text,
                chat_id=chat_id,
                keyboard=kb,
            )
        )

        if result.success and result.message_id:
            session.last_message_id = result.message_id

    def clear_session(self, user_id: str) -> None:
        """Clear the onboarding session for a user.

        Parameters
        ----------
        user_id:
            The user's identifier.
        """
        if user_id in self._sessions:
            del self._sessions[user_id]
