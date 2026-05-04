"""Tests for the messenger base classes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from mai_gram.messenger.base import (
    IncomingMessage,
    InlineKeyboardSpec,
    MessageHandler,
    MessageType,
    Messenger,
    OutgoingMessage,
    SendResult,
)


class _ConcreteMessenger(Messenger):
    """Minimal concrete Messenger for testing default method implementations."""

    @property
    def platform_name(self) -> str:
        return "test"

    @property
    def max_message_length(self) -> int:
        return 4000

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_message(self, message: OutgoingMessage) -> SendResult:
        return SendResult(success=True, message_id="sent_1")

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        return True

    async def send_typing_indicator(self, chat_id: str) -> None:
        pass

    def register_message_handler(self, handler: MessageHandler) -> None:
        pass

    def register_command_handler(
        self, command: str, handler: MessageHandler, *, description: str = ""
    ) -> None:
        pass

    def register_callback_handler(self, handler: MessageHandler) -> None:
        pass


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


class TestMessengerDefaults:
    """Tests for default method implementations on the Messenger ABC."""

    def setup_method(self) -> None:
        self.messenger = _ConcreteMessenger()

    def test_register_document_handler_noop(self) -> None:
        async def _handler(msg: IncomingMessage) -> None:
            pass

        result = self.messenger.register_document_handler(_handler)
        assert result is None

    def test_build_inline_keyboard(self) -> None:
        buttons: InlineKeyboardSpec = [
            [("Yes", "cb:yes"), ("No", "cb:no")],
            [("Cancel", "cb:cancel")],
        ]
        kb = self.messenger.build_inline_keyboard(buttons)
        assert kb == {
            "inline_keyboard": [
                [
                    {"text": "Yes", "callback_data": "cb:yes"},
                    {"text": "No", "callback_data": "cb:no"},
                ],
                [{"text": "Cancel", "callback_data": "cb:cancel"}],
            ]
        }

    async def test_download_file_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="does not support file downloads"):
            await self.messenger.download_file("file_123")

    def test_get_callback_source_message_returns_none(self) -> None:
        msg = IncomingMessage(
            platform="test",
            chat_id="c1",
            user_id="u1",
            message_id="m1",
            message_type=MessageType.CALLBACK,
            callback_data="cb:test",
        )
        assert self.messenger.get_callback_source_message(msg) is None

    async def test_delete_callback_source_message_no_source(self) -> None:
        msg = IncomingMessage(
            platform="test",
            chat_id="c1",
            user_id="u1",
            message_id="m1",
            message_type=MessageType.CALLBACK,
            callback_data="cb:test",
        )
        result = await self.messenger.delete_callback_source_message(msg)
        assert result is False

    async def test_set_profile_photo_default(self) -> None:
        result = await self.messenger.set_profile_photo("/home/user/photo.jpg")
        assert result is False

    async def test_set_profile_name_default(self) -> None:
        result = await self.messenger.set_profile_name("New Name")
        assert result is False

    def test_max_message_length_default(self) -> None:
        assert self.messenger.max_message_length == 4000
