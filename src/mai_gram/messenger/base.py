"""Abstract messenger interface.

Defines the contract that all messaging platform implementations
(Telegram, Discord, Matrix, etc.) must follow. This allows the
conversation engine to be platform-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


class MessageType(str, Enum):
    """Types of messages that can be received."""

    TEXT = "text"
    COMMAND = "command"
    CALLBACK = "callback"  # Button press, inline keyboard, etc.
    PHOTO = "photo"
    VOICE = "voice"
    DOCUMENT = "document"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A message received from the human.

    Attributes
    ----------
    platform:
        Identifier of the messaging platform (e.g., "telegram", "discord").
    chat_id:
        Platform-specific chat/conversation identifier.
    user_id:
        Platform-specific user identifier.
    message_id:
        Platform-specific message identifier.
    message_type:
        Type of the message (text, command, callback, etc.).
    text:
        The text content of the message. Empty for non-text messages.
    command:
        If message_type is COMMAND, the command name without the leading slash.
    command_args:
        Arguments passed to the command, if any.
    callback_data:
        If message_type is CALLBACK, the callback data string.
    timestamp:
        When the message was sent.
    bot_id:
        Identifier of the bot that received this message (e.g., Telegram bot username).
        Used to distinguish companions created via different bots by the same human.
    document_file_id:
        Platform-specific file identifier for uploaded documents.
    document_file_name:
        Original filename of the uploaded document.
    document_mime_type:
        MIME type of the uploaded document.
    document_file_size:
        Size of the uploaded document in bytes.
    raw:
        The original platform-specific message object for advanced use cases.
    """

    platform: str
    chat_id: str
    user_id: str
    message_id: str
    message_type: MessageType
    text: str = ""
    command: str | None = None
    command_args: str | None = None
    callback_data: str | None = None
    timestamp: datetime | None = None
    bot_id: str = ""
    document_file_id: str | None = None
    document_file_name: str | None = None
    document_mime_type: str | None = None
    document_file_size: int | None = None
    raw: Any = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class OutgoingMessage:
    """A message to be sent to the human.

    Attributes
    ----------
    text:
        The text content to send.
    chat_id:
        Target chat/conversation identifier.
    reply_to:
        Optional message ID to reply to.
    parse_mode:
        Text formatting mode (platform-specific, e.g., "Markdown", "HTML").
    keyboard:
        Optional keyboard/buttons configuration.
    photo_path:
        Optional path to a photo to send.
    photo_url:
        Optional URL of a photo to send.
    """

    text: str
    chat_id: str
    reply_to: str | None = None
    parse_mode: str | None = None
    keyboard: Any = None
    photo_path: str | None = None
    photo_url: str | None = None


@dataclass(frozen=True, slots=True)
class SendResult:
    """Result of sending a message.

    Attributes
    ----------
    success:
        Whether the message was sent successfully.
    message_id:
        The platform-specific ID of the sent message, if successful.
    error:
        Error message if the send failed.
    """

    success: bool
    message_id: str | None = None
    error: str | None = None


class MessengerError(Exception):
    """Base exception for messenger-related errors."""


# Type alias for message handlers
MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, None]]


class Messenger(ABC):
    """Abstract base class for messaging platform implementations.

    Subclasses implement the platform-specific logic for sending and
    receiving messages. The conversation engine interacts with this
    interface without knowing which platform is being used.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the name of this messaging platform."""

    @abstractmethod
    async def start(self) -> None:
        """Start the messenger (connect, authenticate, begin polling/webhooks).

        This method should be called once at application startup.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop the messenger gracefully.

        This method should be called during application shutdown.
        """

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> SendResult:
        """Send a message to the human.

        Parameters
        ----------
        message:
            The message to send.

        Returns
        -------
        SendResult
            The result of the send operation.
        """

    @abstractmethod
    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        """Edit a previously sent message.

        Parameters
        ----------
        chat_id:
            The chat containing the message.
        message_id:
            The ID of the message to edit.
        new_text:
            The new text content.
        **kwargs:
            Platform-specific options (parse_mode, keyboard, etc.).

        Returns
        -------
        SendResult
            The result of the edit operation.
        """

    @abstractmethod
    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message.

        Parameters
        ----------
        chat_id:
            The chat containing the message.
        message_id:
            The ID of the message to delete.

        Returns
        -------
        bool
            True if the message was deleted successfully.
        """

    @abstractmethod
    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show a "typing..." indicator in the chat.

        Parameters
        ----------
        chat_id:
            The chat to show the indicator in.
        """

    @abstractmethod
    def register_message_handler(self, handler: MessageHandler) -> None:
        """Register a handler for incoming messages.

        Parameters
        ----------
        handler:
            Async function that will be called for each incoming message.
        """

    @abstractmethod
    def register_command_handler(
        self,
        command: str,
        handler: MessageHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a handler for a specific command.

        Parameters
        ----------
        command:
            The command name (without leading slash).
        handler:
            Async function that will be called when the command is received.
        description:
            Human-readable description shown in command autocomplete menus
            (e.g. Telegram's ``/`` menu). Ignored on platforms that don't
            support it.
        """

    @abstractmethod
    def register_callback_handler(self, handler: MessageHandler) -> None:
        """Register a handler for callback queries (button presses).

        Parameters
        ----------
        handler:
            Async function that will be called for callback queries.
        """

    def register_document_handler(self, handler: MessageHandler) -> None:  # noqa: B027
        """Register a handler for incoming document uploads.

        Not all platforms support document uploads. The default is a no-op.

        Parameters
        ----------
        handler:
            Async function that will be called for document messages.
        """

    async def download_file(self, file_id: str) -> bytes:
        """Download a file by its platform-specific file ID.

        Parameters
        ----------
        file_id:
            The platform-specific file identifier.

        Returns
        -------
        bytes
            The raw file content.
        """
        raise NotImplementedError("This platform does not support file downloads")

    async def set_profile_photo(self, photo_path: str) -> bool:
        """Set the bot's profile photo.

        Parameters
        ----------
        photo_path:
            Path to the photo file.

        Returns
        -------
        bool
            True if the photo was set successfully.
        """
        # Default implementation does nothing (not all platforms support this)
        return False

    async def set_profile_name(self, name: str) -> bool:
        """Set the bot's display name.

        Parameters
        ----------
        name:
            The new display name.

        Returns
        -------
        bool
            True if the name was set successfully.
        """
        # Default implementation does nothing
        return False
