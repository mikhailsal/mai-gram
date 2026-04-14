"""Telegram messenger implementation using python-telegram-bot.

This module implements the Messenger interface for Telegram,
handling all Telegram-specific logic including message conversion,
keyboard building, and bot lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler as TGMessageHandler,
    filters,
)

from mai_gram.messenger.base import (
    IncomingMessage,
    MessageHandler,
    Messenger,
    MessengerError,
    MessageType,
    OutgoingMessage,
    SendResult,
)

logger = logging.getLogger(__name__)


def _convert_update_to_message(update: Update, *, bot_id: str = "") -> IncomingMessage | None:
    """Convert a Telegram Update to our IncomingMessage format.

    Parameters
    ----------
    update:
        The Telegram update object.
    bot_id:
        Identifier of the bot that received this update.

    Returns
    -------
    IncomingMessage or None
        The converted message, or None if the update is not a message.
    """
    # Handle callback queries (button presses)
    if update.callback_query:
        query = update.callback_query
        return IncomingMessage(
            platform="telegram",
            chat_id=str(query.message.chat_id) if query.message else "",
            user_id=str(query.from_user.id) if query.from_user else "",
            message_id=str(query.id),
            message_type=MessageType.CALLBACK,
            callback_data=query.data,
            timestamp=datetime.now(timezone.utc),
            bot_id=bot_id,
            raw=update,
        )

    # Handle regular messages
    message = update.message or update.edited_message
    if not message:
        return None

    # Determine message type
    if message.text and message.text.startswith("/"):
        msg_type = MessageType.COMMAND
        # Parse command and arguments
        parts = message.text.split(maxsplit=1)
        command = parts[0][1:].split("@")[0]  # Remove / and @botname
        command_args = parts[1] if len(parts) > 1 else None
    elif message.text:
        msg_type = MessageType.TEXT
        command = None
        command_args = None
    elif message.photo:
        msg_type = MessageType.PHOTO
        command = None
        command_args = None
    elif message.voice:
        msg_type = MessageType.VOICE
        command = None
        command_args = None
    elif message.document:
        msg_type = MessageType.DOCUMENT
        command = None
        command_args = None
    else:
        msg_type = MessageType.OTHER
        command = None
        command_args = None

    return IncomingMessage(
        platform="telegram",
        chat_id=str(message.chat_id),
        user_id=str(message.from_user.id) if message.from_user else "",
        message_id=str(message.message_id),
        message_type=msg_type,
        text=message.text or message.caption or "",
        command=command,
        command_args=command_args,
        timestamp=message.date,
        bot_id=bot_id,
        raw=update,
    )


def build_inline_keyboard(
    buttons: list[list[tuple[str, str]]]
) -> InlineKeyboardMarkup:
    """Build an inline keyboard from a list of button rows.

    Parameters
    ----------
    buttons:
        List of rows, where each row is a list of (text, callback_data) tuples.

    Returns
    -------
    InlineKeyboardMarkup
        The Telegram inline keyboard markup.
    """
    keyboard = [
        [InlineKeyboardButton(text, callback_data=data) for text, data in row]
        for row in buttons
    ]
    return InlineKeyboardMarkup(keyboard)


def build_reply_keyboard(
    buttons: list[list[str]],
    *,
    one_time: bool = True,
    resize: bool = True,
) -> ReplyKeyboardMarkup:
    """Build a reply keyboard from a list of button rows.

    Parameters
    ----------
    buttons:
        List of rows, where each row is a list of button texts.
    one_time:
        If True, the keyboard disappears after a button is pressed.
    resize:
        If True, the keyboard is resized to fit the buttons.

    Returns
    -------
    ReplyKeyboardMarkup
        The Telegram reply keyboard markup.
    """
    return ReplyKeyboardMarkup(
        buttons,
        one_time_keyboard=one_time,
        resize_keyboard=resize,
    )


class TelegramMessenger(Messenger):
    """Telegram implementation of the Messenger interface.

    Uses python-telegram-bot's Application for handling updates
    and sending messages.

    Parameters
    ----------
    token:
        The Telegram Bot API token from @BotFather.
    bot_id:
        A short identifier for this bot (e.g., the bot username).
        Used to distinguish companions created via different bots.
        If not provided, it will be resolved from the Telegram API on start.
    """

    def __init__(self, token: str, *, bot_id: str = "") -> None:
        if not token:
            raise MessengerError("Telegram bot token must not be empty")

        self._token = token
        self._bot_id = bot_id
        self._app: Application | None = None
        self._message_handlers: list[MessageHandler] = []
        self._callback_handlers: list[MessageHandler] = []
        self._command_handlers: dict[str, MessageHandler] = {}
        self._command_descriptions: dict[str, str] = {}

    @property
    def platform_name(self) -> str:
        """Return 'telegram' as the platform name."""
        return "telegram"

    @property
    def bot_id(self) -> str:
        """Return the bot identifier (username)."""
        return self._bot_id

    @property
    def bot(self) -> Bot:
        """Return the underlying Telegram Bot instance."""
        if self._app is None:
            raise MessengerError("Messenger not started")
        return self._app.bot

    async def start(self) -> None:
        """Start the Telegram bot and begin polling for updates."""
        logger.info("Starting Telegram messenger (bot_id=%s)...", self._bot_id or "(resolving)")

        # Build the application with increased timeouts for network resilience
        # Default timeouts are 5 seconds which is too aggressive for unstable networks
        self._app = (
            Application.builder()
            .token(self._token)
            .read_timeout(30.0)  # Wait up to 30s for response data
            .write_timeout(30.0)  # Wait up to 30s for sending data
            .connect_timeout(15.0)  # Wait up to 15s for connection
            .pool_timeout(10.0)  # Wait up to 10s for connection from pool
            .get_updates_read_timeout(30.0)  # Polling read timeout
            .get_updates_write_timeout(30.0)  # Polling write timeout
            .get_updates_connect_timeout(15.0)  # Polling connect timeout
            .build()
        )

        # Resolve bot_id from Telegram API if not provided
        if not self._bot_id:
            bot_info = await self._app.bot.get_me()
            self._bot_id = bot_info.username or str(bot_info.id)
            logger.info("Resolved bot_id: %s", self._bot_id)

        # Register command handlers
        for command, handler in self._command_handlers.items():
            self._app.add_handler(
                CommandHandler(command, self._make_command_wrapper(handler))
            )

        # Register callback query handler
        if self._callback_handlers:
            self._app.add_handler(
                CallbackQueryHandler(self._handle_callback_query)
            )

        # Register general message handler (must be last)
        if self._message_handlers:
            self._app.add_handler(
                TGMessageHandler(
                    filters.TEXT & ~filters.COMMAND,
                    self._handle_message,
                )
            )

        # Initialize the application
        await self._app.initialize()
        await self._app.start()

        # Register commands with Telegram so they appear in the "/" menu
        if self._command_descriptions:
            from telegram import BotCommand

            bot_commands = [
                BotCommand(cmd, desc)
                for cmd, desc in self._command_descriptions.items()
            ]
            try:
                await self._app.bot.set_my_commands(bot_commands)
                logger.info(
                    "Registered %d command(s) with Telegram",
                    len(bot_commands),
                )
            except TelegramError as e:
                logger.warning("Failed to set bot commands: %s", e)

        # Start polling in the background
        await self._app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram messenger started successfully")

    async def stop(self) -> None:
        """Stop the Telegram bot gracefully."""
        if self._app is None:
            return

        logger.info("Stopping Telegram messenger...")

        # Stop polling and shutdown
        if self._app.updater.running:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

        self._app = None
        logger.info("Telegram messenger stopped")

    async def send_message(
        self, message: OutgoingMessage, *, max_retries: int = 2
    ) -> SendResult:
        """Send a message via Telegram with retry logic for transient failures.
        
        Parameters
        ----------
        message:
            The message to send.
        max_retries:
            Number of retries on transient errors (default: 2).
        """
        if self._app is None:
            return SendResult(success=False, error="Messenger not started")

        # Determine parse mode
        parse_mode = None
        if message.parse_mode:
            if message.parse_mode.lower() == "markdown":
                parse_mode = ParseMode.MARKDOWN_V2
            elif message.parse_mode.lower() == "html":
                parse_mode = ParseMode.HTML

        # Build keyboard if provided
        reply_markup = None
        if message.keyboard is not None:
            if message.keyboard == "remove":
                reply_markup = ReplyKeyboardRemove()
            elif isinstance(message.keyboard, (InlineKeyboardMarkup, ReplyKeyboardMarkup)):
                reply_markup = message.keyboard
            elif isinstance(message.keyboard, list):
                # Assume it's inline keyboard data
                reply_markup = build_inline_keyboard(message.keyboard)

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):  # +2 because range is exclusive
            try:
                # Send photo if provided
                if message.photo_path:
                    with open(message.photo_path, "rb") as photo:
                        sent = await self._app.bot.send_photo(
                            chat_id=int(message.chat_id),
                            photo=photo,
                            caption=message.text if message.text else None,
                            parse_mode=parse_mode,
                            reply_markup=reply_markup,
                            reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                        )
                elif message.photo_url:
                    sent = await self._app.bot.send_photo(
                        chat_id=int(message.chat_id),
                        photo=message.photo_url,
                        caption=message.text if message.text else None,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                    )
                else:
                    # Send text message
                    sent = await self._app.bot.send_message(
                        chat_id=int(message.chat_id),
                        text=message.text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        reply_to_message_id=int(message.reply_to) if message.reply_to else None,
                    )

                return SendResult(success=True, message_id=str(sent.message_id))

            except TelegramError as e:
                last_error = e
                error_str = str(e).lower()
                # Retry on transient errors (timeouts, network issues)
                is_transient = "timed out" in error_str or "network" in error_str
                if is_transient and attempt <= max_retries:
                    logger.warning(
                        "Telegram send failed (attempt %d/%d): %s - retrying...",
                        attempt,
                        max_retries + 1,
                        e,
                    )
                    await asyncio.sleep(1.0 * attempt)  # Exponential backoff
                    continue
                # Non-transient error or max retries reached
                logger.error("Failed to send Telegram message: %s", e)
                return SendResult(success=False, error=str(e))

        # Should not reach here, but just in case
        logger.error("Failed to send Telegram message after retries: %s", last_error)
        return SendResult(success=False, error=str(last_error) if last_error else "Unknown error")

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        """Edit a previously sent Telegram message."""
        if self._app is None:
            return SendResult(success=False, error="Messenger not started")

        try:
            parse_mode = None
            if kwargs.get("parse_mode"):
                pm = kwargs["parse_mode"].lower()
                if pm == "markdown":
                    parse_mode = ParseMode.MARKDOWN_V2
                elif pm == "html":
                    parse_mode = ParseMode.HTML

            reply_markup = kwargs.get("keyboard")
            if isinstance(reply_markup, list):
                reply_markup = build_inline_keyboard(reply_markup)

            await self._app.bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(message_id),
                text=new_text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return SendResult(success=True, message_id=message_id)

        except TelegramError as e:
            logger.error("Failed to edit Telegram message: %s", e)
            return SendResult(success=False, error=str(e))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a Telegram message."""
        if self._app is None:
            return False

        try:
            await self._app.bot.delete_message(
                chat_id=int(chat_id),
                message_id=int(message_id),
            )
            return True
        except TelegramError as e:
            logger.error("Failed to delete Telegram message: %s", e)
            return False

    async def send_typing_indicator(self, chat_id: str) -> None:
        """Show the typing indicator in a Telegram chat."""
        if self._app is None:
            return

        try:
            await self._app.bot.send_chat_action(
                chat_id=int(chat_id),
                action=ChatAction.TYPING,
            )
        except TelegramError as e:
            logger.warning("Failed to send typing indicator: %s", e)

    def register_message_handler(self, handler: MessageHandler) -> None:
        """Register a handler for incoming text messages."""
        self._message_handlers.append(handler)

    def register_command_handler(
        self, command: str, handler: MessageHandler, *, description: str = "",
    ) -> None:
        """Register a handler for a specific command."""
        self._command_handlers[command] = handler
        if description:
            self._command_descriptions[command] = description

    def register_callback_handler(self, handler: MessageHandler) -> None:
        """Register a handler for callback queries (button presses)."""
        self._callback_handlers.append(handler)

    async def set_profile_photo(self, photo_path: str) -> bool:
        """Set the bot's profile photo."""
        if self._app is None:
            return False

        try:
            with open(photo_path, "rb") as photo:
                await self._app.bot.set_chat_photo(
                    chat_id=self._app.bot.id,
                    photo=photo,
                )
            return True
        except TelegramError as e:
            logger.error("Failed to set profile photo: %s", e)
            return False

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------

    def _make_command_wrapper(
        self, handler: MessageHandler
    ) -> Any:
        """Create a Telegram CommandHandler-compatible wrapper."""
        async def wrapper(
            update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            msg = _convert_update_to_message(update, bot_id=self._bot_id)
            if msg:
                await handler(msg)

        return wrapper

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages."""
        msg = _convert_update_to_message(update, bot_id=self._bot_id)
        if msg:
            for handler in self._message_handlers:
                await handler(msg)

    async def _handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback queries (button presses)."""
        msg = _convert_update_to_message(update, bot_id=self._bot_id)
        if msg:
            # Acknowledge the callback to remove the loading indicator
            if update.callback_query:
                try:
                    await update.callback_query.answer()
                except Exception as e:
                    # Network timeouts can happen - log and continue
                    logger.warning("Failed to answer callback query: %s", e)

            for handler in self._callback_handlers:
                await handler(msg)


async def answer_callback_query(update: Update, text: str | None = None) -> None:
    """Helper to answer a callback query with optional text.

    Parameters
    ----------
    update:
        The Telegram update containing the callback query.
    text:
        Optional text to show as a toast notification.
    """
    if update.callback_query:
        await update.callback_query.answer(text=text)
