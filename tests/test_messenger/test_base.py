"""Tests for the messenger base classes."""

from datetime import datetime, timezone

import pytest

from mai_companion.messenger.base import (
    IncomingMessage,
    MessageType,
    OutgoingMessage,
    SendResult,
)


class TestIncomingMessage:
    """Tests for IncomingMessage dataclass."""

    def test_text_message_creation(self):
        """Test creating a text message."""
        msg = IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.TEXT,
            text="Hello, world!",
        )

        assert msg.platform == "telegram"
        assert msg.chat_id == "12345"
        assert msg.user_id == "67890"
        assert msg.message_id == "111"
        assert msg.message_type == MessageType.TEXT
        assert msg.text == "Hello, world!"
        assert msg.command is None
        assert msg.callback_data is None

    def test_command_message_creation(self):
        """Test creating a command message."""
        msg = IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.COMMAND,
            text="/start",
            command="start",
            command_args="some args",
        )

        assert msg.message_type == MessageType.COMMAND
        assert msg.command == "start"
        assert msg.command_args == "some args"

    def test_callback_message_creation(self):
        """Test creating a callback message."""
        msg = IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.CALLBACK,
            callback_data="button:pressed",
        )

        assert msg.message_type == MessageType.CALLBACK
        assert msg.callback_data == "button:pressed"

    def test_message_with_timestamp(self):
        """Test message with timestamp."""
        now = datetime.now(timezone.utc)
        msg = IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.TEXT,
            text="Hello",
            timestamp=now,
        )

        assert msg.timestamp == now

    def test_message_is_frozen(self):
        """Test that IncomingMessage is immutable."""
        msg = IncomingMessage(
            platform="telegram",
            chat_id="12345",
            user_id="67890",
            message_id="111",
            message_type=MessageType.TEXT,
        )

        with pytest.raises(AttributeError):
            msg.text = "new text"  # type: ignore


class TestOutgoingMessage:
    """Tests for OutgoingMessage dataclass."""

    def test_simple_message(self):
        """Test creating a simple outgoing message."""
        msg = OutgoingMessage(
            text="Hello!",
            chat_id="12345",
        )

        assert msg.text == "Hello!"
        assert msg.chat_id == "12345"
        assert msg.reply_to is None
        assert msg.parse_mode is None
        assert msg.keyboard is None

    def test_message_with_options(self):
        """Test creating a message with all options."""
        msg = OutgoingMessage(
            text="Hello!",
            chat_id="12345",
            reply_to="111",
            parse_mode="HTML",
            keyboard=[["Button 1", "Button 2"]],
        )

        assert msg.reply_to == "111"
        assert msg.parse_mode == "HTML"
        assert msg.keyboard == [["Button 1", "Button 2"]]

    def test_message_with_photo(self):
        """Test creating a message with a photo."""
        msg = OutgoingMessage(
            text="Check this out!",
            chat_id="12345",
            photo_path="/path/to/photo.jpg",
        )

        assert msg.photo_path == "/path/to/photo.jpg"


class TestSendResult:
    """Tests for SendResult dataclass."""

    def test_successful_result(self):
        """Test a successful send result."""
        result = SendResult(success=True, message_id="12345")

        assert result.success is True
        assert result.message_id == "12345"
        assert result.error is None

    def test_failed_result(self):
        """Test a failed send result."""
        result = SendResult(success=False, error="Network error")

        assert result.success is False
        assert result.message_id is None
        assert result.error == "Network error"

    def test_result_is_frozen(self):
        """Test that SendResult is immutable."""
        result = SendResult(success=True)

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types(self):
        """Test all message types exist."""
        assert MessageType.TEXT.value == "text"
        assert MessageType.COMMAND.value == "command"
        assert MessageType.CALLBACK.value == "callback"
        assert MessageType.PHOTO.value == "photo"
        assert MessageType.VOICE.value == "voice"
        assert MessageType.DOCUMENT.value == "document"
        assert MessageType.OTHER.value == "other"
