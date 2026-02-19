"""Tests for the bot handler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mai_companion.bot.handler import BotHandler
from mai_companion.llm.provider import ChatMessage, LLMResponse, MessageRole
from mai_companion.messenger.base import IncomingMessage, MessageType, SendResult


@pytest.fixture
def mock_messenger():
    """Create a mock messenger."""
    messenger = MagicMock()
    messenger.send_message = AsyncMock(
        return_value=SendResult(success=True, message_id="123")
    )
    messenger.send_typing_indicator = AsyncMock()
    messenger.register_command_handler = MagicMock()
    messenger.register_message_handler = MagicMock()
    messenger.register_callback_handler = MagicMock()
    return messenger


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=MagicMock(content="Hello! How can I help you?")
    )
    return provider


@pytest.fixture
def bot_handler(mock_messenger, mock_llm_provider):
    """Create a bot handler with mocks (no access control)."""
    with patch("mai_companion.bot.handler.get_settings") as mock_settings:
        mock_settings.return_value.get_allowed_user_ids.return_value = set()
        return BotHandler(mock_messenger, mock_llm_provider)


@pytest.fixture
def bot_handler_with_access_control(mock_messenger, mock_llm_provider):
    """Create a bot handler with access control enabled."""
    with patch("mai_companion.bot.handler.get_settings") as mock_settings:
        mock_settings.return_value.get_allowed_user_ids.return_value = {"allowed_user"}
        return BotHandler(mock_messenger, mock_llm_provider)


class TestBotHandlerInit:
    """Tests for BotHandler initialization."""

    def test_registers_start_command(self, mock_messenger, mock_llm_provider):
        """Test that /start command is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        # Check that register_command_handler was called with "start"
        calls = mock_messenger.register_command_handler.call_args_list
        command_names = [call[0][0] for call in calls]
        assert "start" in command_names

    def test_registers_reset_command(self, mock_messenger, mock_llm_provider):
        """Test that /reset command is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        calls = mock_messenger.register_command_handler.call_args_list
        command_names = [call[0][0] for call in calls]
        assert "reset" in command_names

    def test_registers_mood_command(self, mock_messenger, mock_llm_provider):
        """Test that /mood command is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        calls = mock_messenger.register_command_handler.call_args_list
        command_names = [call[0][0] for call in calls]
        assert "mood" in command_names

    def test_registers_help_command(self, mock_messenger, mock_llm_provider):
        """Test that /help command is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        calls = mock_messenger.register_command_handler.call_args_list
        command_names = [call[0][0] for call in calls]
        assert "help" in command_names

    def test_registers_message_handler(self, mock_messenger, mock_llm_provider):
        """Test that message handler is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        mock_messenger.register_message_handler.assert_called_once()

    def test_registers_callback_handler(self, mock_messenger, mock_llm_provider):
        """Test that callback handler is registered."""
        BotHandler(mock_messenger, mock_llm_provider)

        mock_messenger.register_callback_handler.assert_called_once()


class TestBotHandlerCommands:
    """Tests for bot command handlers."""

    @pytest.fixture
    def start_message(self):
        """Create a /start command message."""
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.COMMAND,
            text="/start",
            command="start",
        )

    @pytest.fixture
    def reset_message(self):
        """Create a /reset command message."""
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.COMMAND,
            text="/reset",
            command="reset",
        )

    @pytest.fixture
    def help_message(self):
        """Create a /help command message."""
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.COMMAND,
            text="/help",
            command="help",
        )

    async def test_start_command_no_companion(
        self, bot_handler, start_message, mock_messenger
    ):
        """Test /start when no companion exists."""
        with patch("mai_companion.bot.handler.get_session") as mock_get_session:
            # Mock the database session
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            # Mock no companion found
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            await bot_handler._handle_start(start_message)

            # Should start onboarding
            assert bot_handler._onboarding.is_onboarding(start_message.user_id)

    async def test_help_command_sends_help_text(
        self, bot_handler, help_message, mock_messenger
    ):
        """Test /help sends help text."""
        with patch("mai_companion.bot.handler.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            await bot_handler._handle_help(help_message)

            # Should send a message
            mock_messenger.send_message.assert_called()

            # Check that the message contains command information
            call_args = mock_messenger.send_message.call_args
            message = call_args[0][0]
            assert "/start" in message.text
            assert "/reset" in message.text
            assert "/mood" in message.text
            assert "/help" in message.text


class TestBotHandlerConversation:
    """Tests for conversation handling."""

    @pytest.fixture
    def text_message(self):
        """Create a text message."""
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.TEXT,
            text="Hello!",
        )

    async def test_message_without_companion_prompts_start(
        self, bot_handler, text_message, mock_messenger
    ):
        """Test that messaging without a companion prompts /start."""
        with patch("mai_companion.bot.handler.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            await bot_handler._handle_conversation(text_message)

            # Should send a message prompting to use /start
            mock_messenger.send_message.assert_called()
            call_args = mock_messenger.send_message.call_args
            message = call_args[0][0]
            assert "/start" in message.text


class TestRateLimitCallback:
    """Tests for rate limit callback."""

    async def test_rate_limit_sends_message(
        self, bot_handler, mock_messenger
    ):
        """Test that rate limiting sends a message to the user."""
        with patch("mai_companion.bot.handler.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            await bot_handler._handle_rate_limited("user1", "chat1")

            # Should send a rate limit message
            mock_messenger.send_message.assert_called()


class TestAccessControl:
    """Tests for access control functionality."""

    @pytest.fixture
    def allowed_user_message(self):
        """Create a message from an allowed user."""
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="allowed_user",
            message_id="111",
            message_type=MessageType.TEXT,
            text="Hello!",
        )

    @pytest.fixture
    def denied_user_message(self):
        """Create a message from a denied user."""
        return IncomingMessage(
            platform="telegram",
            chat_id="99999",
            user_id="denied_user",
            message_id="222",
            message_type=MessageType.TEXT,
            text="Hello!",
        )

    async def test_access_granted_for_allowed_user(
        self, bot_handler_with_access_control, allowed_user_message
    ):
        """Test that allowed users can access the bot."""
        result = await bot_handler_with_access_control._check_access(
            allowed_user_message
        )
        assert result is True

    async def test_access_denied_for_unknown_user(
        self, bot_handler_with_access_control, denied_user_message, mock_messenger
    ):
        """Test that unknown users are denied access."""
        result = await bot_handler_with_access_control._check_access(
            denied_user_message
        )
        assert result is False

        # Should send access denied message
        mock_messenger.send_message.assert_called()
        call_args = mock_messenger.send_message.call_args
        message = call_args[0][0]
        assert "Access denied" in message.text
        assert "denied_user" in message.text  # User ID should be shown

    async def test_no_access_control_allows_everyone(
        self, bot_handler, denied_user_message
    ):
        """Test that when no allowed_users is set, everyone is allowed."""
        result = await bot_handler._check_access(denied_user_message)
        assert result is True

    async def test_start_command_blocked_for_denied_user(
        self, bot_handler_with_access_control, mock_messenger
    ):
        """Test that /start is blocked for denied users."""
        start_message = IncomingMessage(
            platform="telegram",
            chat_id="99999",
            user_id="denied_user",
            message_id="222",
            message_type=MessageType.COMMAND,
            text="/start",
            command="start",
        )

        await bot_handler_with_access_control._handle_start(start_message)

        # Should send access denied message (not onboarding)
        mock_messenger.send_message.assert_called()
        call_args = mock_messenger.send_message.call_args
        message = call_args[0][0]
        assert "Access denied" in message.text

    async def test_message_blocked_for_denied_user(
        self, bot_handler_with_access_control, denied_user_message, mock_messenger
    ):
        """Test that regular messages are blocked for denied users."""
        await bot_handler_with_access_control._handle_message(denied_user_message)

        # Should send access denied message
        mock_messenger.send_message.assert_called()
        call_args = mock_messenger.send_message.call_args
        message = call_args[0][0]
        assert "Access denied" in message.text


class TestBotHandlerPhase5Integration:
    @pytest.fixture
    def text_message(self):
        return IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.TEXT,
            text="Hello!",
        )

    async def test_conversation_uses_prompt_builder(self, bot_handler, text_message):
        companion = MagicMock(
            id="12345",
            personality_traits="{}",
            temperature=0.7,
            human_language="English",
            system_prompt="{mood_section}\n{relationship_section}",
            relationship_stage="getting_to_know",
        )
        built_context = [ChatMessage(role=MessageRole.SYSTEM, content="system")]
        mood = MagicMock()

        with (
            patch("mai_companion.bot.handler.get_session") as mock_get_session,
            patch("mai_companion.bot.handler.PromptBuilder") as mock_builder_cls,
            patch("mai_companion.bot.handler.run_with_tools", new=AsyncMock(return_value=LLMResponse(content="ok", model="mock"))),
            patch("mai_companion.bot.handler.MemoryManager") as mock_memory_cls,
            patch("mai_companion.bot.handler.MoodManager") as mock_mood_cls,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = companion
            mock_session.execute = AsyncMock(return_value=mock_result)

            mock_builder = MagicMock()
            mock_builder.build_context = AsyncMock(return_value=built_context)
            mock_builder_cls.return_value = mock_builder

            mock_memory = MagicMock()
            mock_memory.save_message = AsyncMock()
            mock_memory.run_forgetting_cycle = AsyncMock()
            mock_memory_cls.return_value = mock_memory

            mock_mood = MagicMock()
            mock_mood.get_current_mood = AsyncMock(return_value=mood)
            mock_mood_cls.return_value = mock_mood

            await bot_handler._handle_conversation(text_message)

            assert mock_builder.build_context.await_count == 1
            called_companion, called_mood = mock_builder.build_context.await_args.args
            assert called_companion is companion
            assert called_mood is mood
            assert "clock" in mock_builder.build_context.await_args.kwargs

    async def test_conversation_uses_mcp_bridge(self, bot_handler, text_message):
        companion = MagicMock(
            id="12345",
            personality_traits="{}",
            temperature=0.7,
            human_language="English",
            system_prompt="{mood_section}\n{relationship_section}",
            relationship_stage="getting_to_know",
        )

        with (
            patch("mai_companion.bot.handler.get_session") as mock_get_session,
            patch("mai_companion.bot.handler.PromptBuilder") as mock_builder_cls,
            patch("mai_companion.bot.handler.run_with_tools", new=AsyncMock(return_value=LLMResponse(content="ok", model="mock"))) as mock_run,
            patch("mai_companion.bot.handler.MemoryManager") as mock_memory_cls,
            patch("mai_companion.bot.handler.MoodManager") as mock_mood_cls,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = companion
            mock_session.execute = AsyncMock(return_value=mock_result)

            mock_builder = MagicMock()
            mock_builder.build_context = AsyncMock(
                return_value=[ChatMessage(role=MessageRole.SYSTEM, content="system")]
            )
            mock_builder_cls.return_value = mock_builder

            mock_memory = MagicMock()
            mock_memory.save_message = AsyncMock()
            mock_memory.run_forgetting_cycle = AsyncMock()
            mock_memory_cls.return_value = mock_memory

            mock_mood = MagicMock()
            mock_mood.get_current_mood = AsyncMock(return_value=MagicMock())
            mock_mood_cls.return_value = mock_mood

            await bot_handler._handle_conversation(text_message)

            assert mock_run.await_count == 1

    async def test_conversation_saves_messages_via_memory_manager(self, bot_handler, text_message):
        companion = MagicMock(
            id="12345",
            personality_traits="{}",
            temperature=0.7,
            human_language="English",
            system_prompt="{mood_section}\n{relationship_section}",
            relationship_stage="getting_to_know",
        )

        with (
            patch("mai_companion.bot.handler.get_session") as mock_get_session,
            patch("mai_companion.bot.handler.PromptBuilder") as mock_builder_cls,
            patch("mai_companion.bot.handler.run_with_tools", new=AsyncMock(return_value=LLMResponse(content="ok", model="mock"))),
            patch("mai_companion.bot.handler.MemoryManager") as mock_memory_cls,
            patch("mai_companion.bot.handler.MoodManager") as mock_mood_cls,
        ):
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = companion
            mock_session.execute = AsyncMock(return_value=mock_result)

            mock_builder = MagicMock()
            mock_builder.build_context = AsyncMock(
                return_value=[ChatMessage(role=MessageRole.SYSTEM, content="system")]
            )
            mock_builder_cls.return_value = mock_builder

            mock_memory = MagicMock()
            mock_memory.save_message = AsyncMock()
            mock_memory.run_forgetting_cycle = AsyncMock()
            mock_memory_cls.return_value = mock_memory

            mock_mood = MagicMock()
            mock_mood.get_current_mood = AsyncMock(return_value=MagicMock())
            mock_mood_cls.return_value = mock_mood

            await bot_handler._handle_conversation(text_message)

            assert mock_memory.save_message.await_count == 2
