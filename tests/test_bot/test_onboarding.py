"""Tests for the onboarding flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mai_companion.bot.onboarding import (
    OnboardingManager,
    OnboardingResult,
    OnboardingSession,
    OnboardingState,
    ONBOARDING_TEXTS,
)
from mai_companion.messenger.base import IncomingMessage, MessageType, SendResult
from mai_companion.personality.character import CommunicationStyle, Verbosity


@pytest.fixture
def mock_messenger():
    """Create a mock messenger."""
    messenger = MagicMock()
    messenger.send_message = AsyncMock(
        return_value=SendResult(success=True, message_id="123")
    )
    return messenger


@pytest.fixture
def mock_translation_service():
    """Create a mock translation service."""
    service = MagicMock()
    # By default, return the original text (simulate English)
    service.detect_language = AsyncMock(return_value="English")
    # Accept optional context parameter (added for UI translations)
    service.translate = AsyncMock(side_effect=lambda text, lang, **kwargs: text)
    service.translate_batch = AsyncMock(side_effect=lambda texts, lang, **kwargs: texts)
    service.translate_ui_batch = AsyncMock(side_effect=lambda texts, lang, **kwargs: texts)
    # Mock the LLM provider for gender inference
    service.llm_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "female"
    service.llm_provider.generate = AsyncMock(return_value=mock_response)
    return service


@pytest.fixture
def onboarding_manager(mock_messenger, mock_translation_service):
    """Create an onboarding manager with mocks."""
    return OnboardingManager(mock_messenger, mock_translation_service)


class TestOnboardingResult:
    """Tests for OnboardingResult dataclass."""

    def test_result_creation(self):
        """Test creating an onboarding result."""
        from mai_companion.personality.character import CharacterBuilder

        config = CharacterBuilder.from_preset("Luna", "caring_guide")
        result = OnboardingResult(config=config)

        assert result.config == config


class TestOnboardingSession:
    """Tests for OnboardingSession dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = OnboardingSession(user_id="user1", chat_id="chat1")

        assert session.user_id == "user1"
        assert session.chat_id == "chat1"
        assert session.state == OnboardingState.NOT_STARTED
        assert session.language == "English"
        assert session.companion_name == ""
        assert session.preset_name is None
        assert session.custom_traits == {}

    def test_session_defaults(self):
        """Test session default values."""
        session = OnboardingSession(user_id="u", chat_id="c")

        # Note: defaults changed to CASUAL/CONCISE for messenger-style conversations
        assert session.communication_style == CommunicationStyle.CASUAL
        assert session.verbosity == Verbosity.CONCISE
        assert session.appearance is None
        assert session.last_message_id is None
        assert session.warning_acknowledged is False


class TestOnboardingState:
    """Tests for OnboardingState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        states = [
            OnboardingState.NOT_STARTED,
            OnboardingState.AWAITING_LANGUAGE,
            OnboardingState.AWAITING_NAME,
            OnboardingState.CHOOSING_PERSONALITY,
            OnboardingState.CUSTOMIZING_TRAITS,
            OnboardingState.CHOOSING_STYLE,
            OnboardingState.CHOOSING_VERBOSITY,
            OnboardingState.AWAITING_APPEARANCE,
            OnboardingState.CONFIRMING,
            OnboardingState.COMPLETED,
        ]
        assert len(states) == 10


class TestOnboardingManager:
    """Tests for OnboardingManager class."""

    async def test_start_onboarding(self, onboarding_manager, mock_messenger):
        """Test starting the onboarding flow."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        # Should create a session
        session = onboarding_manager.get_session("user1")
        assert session is not None
        assert session.state == OnboardingState.AWAITING_LANGUAGE

        # Should send welcome message
        mock_messenger.send_message.assert_called_once()

    async def test_is_onboarding(self, onboarding_manager):
        """Test checking if user is in onboarding."""
        # Not onboarding initially
        assert onboarding_manager.is_onboarding("user1") is False

        # Start onboarding
        await onboarding_manager.start_onboarding("user1", "chat1")
        assert onboarding_manager.is_onboarding("user1") is True

    async def test_language_detection(
        self, onboarding_manager, mock_messenger, mock_translation_service
    ):
        """Test language detection step."""
        mock_translation_service.detect_language = AsyncMock(return_value="Russian")

        await onboarding_manager.start_onboarding("user1", "chat1")

        # Send a language input
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.TEXT,
            text="русский",
        )
        await onboarding_manager.handle_message(msg)

        # Session should have detected language
        session = onboarding_manager.get_session("user1")
        assert session.language == "Russian"
        assert session.state == OnboardingState.AWAITING_NAME

    async def test_name_input(self, onboarding_manager, mock_messenger):
        """Test name input step."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        # Set state to awaiting name
        session = onboarding_manager.get_session("user1")
        session.state = OnboardingState.AWAITING_NAME

        # Send a name
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.TEXT,
            text="Aurora",
        )
        await onboarding_manager.handle_message(msg)

        # Session should have the name
        session = onboarding_manager.get_session("user1")
        assert session.companion_name == "Aurora"
        assert session.state == OnboardingState.CHOOSING_PERSONALITY

    async def test_preset_selection_callback(self, onboarding_manager, mock_messenger):
        """Test selecting a personality preset via callback."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        session = onboarding_manager.get_session("user1")
        session.state = OnboardingState.CHOOSING_PERSONALITY
        session.companion_name = "Aurora"

        # Select presets option
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.CALLBACK,
            callback_data="personality:presets",
        )
        await onboarding_manager.handle_message(msg)

        # Should still be in choosing personality (showing presets)
        session = onboarding_manager.get_session("user1")
        assert session.state == OnboardingState.CHOOSING_PERSONALITY

    async def test_preset_confirm(self, onboarding_manager, mock_messenger):
        """Test confirming a preset selection."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        session = onboarding_manager.get_session("user1")
        session.state = OnboardingState.CHOOSING_PERSONALITY
        session.companion_name = "Aurora"

        # Select a preset
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.CALLBACK,
            callback_data="preset:balanced_friend",
        )
        await onboarding_manager.handle_message(msg)

        # Confirm the preset
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="2",
            message_type=MessageType.CALLBACK,
            callback_data="preset_confirm:yes",
        )
        await onboarding_manager.handle_message(msg)

        # Should skip style/verbosity and go directly to appearance
        # (simplified flow for messenger-style conversations)
        session = onboarding_manager.get_session("user1")
        assert session.state == OnboardingState.AWAITING_APPEARANCE

    async def test_default_style_and_verbosity_used(self, onboarding_manager):
        """Test that defaults are CASUAL and CONCISE for messenger-style conversations.

        Note: Style and verbosity selection steps have been removed from the
        onboarding flow. The companion now always uses CASUAL style and CONCISE
        verbosity for an authentic messenger feel.
        """
        await onboarding_manager.start_onboarding("user1", "chat1")
        session = onboarding_manager.get_session("user1")

        # Verify the defaults match what we expect for messenger-style conversations
        assert session.communication_style == CommunicationStyle.CASUAL
        assert session.verbosity == Verbosity.CONCISE

    async def test_skip_appearance(self, onboarding_manager, mock_messenger):
        """Test skipping appearance description."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        session = onboarding_manager.get_session("user1")
        session.state = OnboardingState.AWAITING_APPEARANCE
        session.companion_name = "Aurora"
        session.preset_name = "balanced_friend"
        session.custom_traits = {"warmth": 0.5}

        # Skip appearance
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.CALLBACK,
            callback_data="appearance:skip",
        )
        await onboarding_manager.handle_message(msg)

        session = onboarding_manager.get_session("user1")
        assert session.appearance is None
        assert session.state == OnboardingState.CONFIRMING

    async def test_appearance_text_input(self, onboarding_manager, mock_messenger):
        """Test providing appearance description."""
        await onboarding_manager.start_onboarding("user1", "chat1")

        session = onboarding_manager.get_session("user1")
        session.state = OnboardingState.AWAITING_APPEARANCE
        session.companion_name = "Aurora"
        session.preset_name = "balanced_friend"
        session.custom_traits = {"warmth": 0.5}

        # Provide appearance
        msg = IncomingMessage(
            platform="telegram",
            chat_id="chat1",
            user_id="user1",
            message_id="1",
            message_type=MessageType.TEXT,
            text="A friendly face with warm eyes",
        )
        await onboarding_manager.handle_message(msg)

        session = onboarding_manager.get_session("user1")
        assert session.appearance == "A friendly face with warm eyes"
        assert session.state == OnboardingState.CONFIRMING

    async def test_clear_session(self, onboarding_manager):
        """Test clearing a session."""
        await onboarding_manager.start_onboarding("user1", "chat1")
        assert onboarding_manager.is_onboarding("user1") is True

        onboarding_manager.clear_session("user1")
        assert onboarding_manager.is_onboarding("user1") is False
        assert onboarding_manager.get_session("user1") is None

    async def test_clear_nonexistent_session(self, onboarding_manager):
        """Test clearing a session that doesn't exist."""
        # Should not raise
        onboarding_manager.clear_session("nonexistent")


class TestOnboardingTexts:
    """Tests for onboarding text templates."""

    def test_welcome_text_exists(self):
        """Test that welcome text exists."""
        assert "welcome" in ONBOARDING_TEXTS
        assert len(ONBOARDING_TEXTS["welcome"]) > 0

    def test_language_confirmed_has_placeholder(self):
        """Test that language_confirmed has the language placeholder."""
        text = ONBOARDING_TEXTS["language_confirmed"]
        assert "{language}" in text

    def test_name_confirmed_has_placeholder(self):
        """Test that name_confirmed has the name placeholder."""
        text = ONBOARDING_TEXTS["name_confirmed"]
        assert "{name}" in text

    def test_completed_has_placeholder(self):
        """Test that completed text has the name placeholder."""
        text = ONBOARDING_TEXTS["completed"]
        assert "{name}" in text

    def test_all_required_texts_exist(self):
        """Test that all required text templates exist."""
        # Note: style_selection and verbosity_selection were removed
        # as the flow now skips these steps for messenger-style conversations
        required_keys = [
            "welcome",
            "language_confirmed",
            "name_confirmed",
            "preset_selection",
            "preset_selected",
            "customize_intro",
            "trait_prompt",
            "appearance_prompt",
            "confirmation",
            "extreme_warning",
            "completed",
            "already_exists",
            "back_to_settings",  # Added for "go back" functionality
        ]

        for key in required_keys:
            assert key in ONBOARDING_TEXTS, f"Missing text template: {key}"

    def test_back_to_settings_text_exists(self):
        """Test that back_to_settings text is different from name_confirmed."""
        # This is important because we don't want to confuse users
        # when they go back to change settings
        assert "back_to_settings" in ONBOARDING_TEXTS
        assert ONBOARDING_TEXTS["back_to_settings"] != ONBOARDING_TEXTS["name_confirmed"]
        assert "go back" in ONBOARDING_TEXTS["back_to_settings"].lower() or \
               "adjust" in ONBOARDING_TEXTS["back_to_settings"].lower()
