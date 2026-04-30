"""Telegram messenger implementation using python-telegram-bot.

This module implements the Messenger interface for Telegram,
handling all Telegram-specific logic including message conversion,
keyboard building, and bot lifecycle management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram.constants import ChatAction
from telegram.error import TelegramError

from mai_gram.messenger.base import (
    CallbackSourceMessage,
    IncomingMessage,
    InlineKeyboardSpec,
    MessageHandler,
    Messenger,
    MessengerError,
    OutgoingMessage,
    SendResult,
)
from mai_gram.messenger.telegram_support import (
    build_application,
    build_inline_keyboard,
    build_reply_keyboard,
    build_reply_markup,
    convert_update_to_message,
    get_parse_mode,
    register_bot_commands,
    register_handlers,
    resolve_bot_id,
    send_message_with_retry,
)

if TYPE_CHECKING:
    from telegram import Bot, Update
    from telegram.ext import Application, ContextTypes

__all__ = [
    "TelegramMessenger",
    "answer_callback_query",
    "build_inline_keyboard",
    "build_reply_keyboard",
]

logger = logging.getLogger(__name__)


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
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None
        self._message_handlers: list[MessageHandler] = []
        self._callback_handlers: list[MessageHandler] = []
        self._document_handlers: list[MessageHandler] = []
        self._command_handlers: dict[str, MessageHandler] = {}
        self._command_descriptions: dict[str, str] = {}

    @property
    def platform_name(self) -> str:
        """Return 'telegram' as the platform name."""
        return "telegram"

    @property
    def max_message_length(self) -> int:
        return 4000

    @property
    def bot_id(self) -> str:
        """Return the bot identifier (username)."""
        return self._bot_id

    @property
    def bot(self) -> Bot:
        """Return the underlying Telegram Bot instance."""
        if self._app is None:
            raise MessengerError("Messenger not started")
        return self._app.bot  # type: ignore[no-any-return]

    async def start(self) -> None:
        """Start the Telegram bot and begin polling for updates."""
        logger.info("Starting Telegram messenger (bot_id=%s)...", self._bot_id or "(resolving)")
        self._app = build_application(self._token)
        self._bot_id = await resolve_bot_id(self._app, self._bot_id)
        logger.info("Resolved bot_id: %s", self._bot_id)
        register_handlers(
            self._app,
            command_handlers=self._command_handlers,
            callback_handlers=self._callback_handlers,
            document_handlers=self._document_handlers,
            message_handlers=self._message_handlers,
            make_command_wrapper=self._make_command_wrapper,
            handle_callback_query=self._handle_callback_query,
            handle_document=self._handle_document,
            handle_message=self._handle_message,
        )
        await self._app.initialize()
        await self._app.start()
        await register_bot_commands(self._app, self._command_descriptions, logger=logger)

        # Start polling in the background
        updater = self._app.updater
        if updater is None:
            raise MessengerError("Telegram application is missing an updater")
        await updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram messenger started successfully")

    async def stop(self) -> None:
        """Stop the Telegram bot gracefully."""
        if self._app is None:
            return

        logger.info("Stopping Telegram messenger...")

        # Stop polling and shutdown
        if self._app.updater and self._app.updater.running:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

        self._app = None
        logger.info("Telegram messenger stopped")

    async def send_message(self, message: OutgoingMessage, *, max_retries: int = 2) -> SendResult:
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
        return await send_message_with_retry(
            self._app.bot,
            message,
            max_retries=max_retries,
            logger=logger,
        )

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        """Edit a previously sent Telegram message."""
        if self._app is None:
            return SendResult(success=False, error="Messenger not started")

        try:
            parse_mode = get_parse_mode(kwargs.get("parse_mode"))
            reply_markup = build_reply_markup(kwargs.get("keyboard"))

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
        self,
        command: str,
        handler: MessageHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a handler for a specific command."""
        self._command_handlers[command] = handler
        if description:
            self._command_descriptions[command] = description

    def register_callback_handler(self, handler: MessageHandler) -> None:
        """Register a handler for callback queries (button presses)."""
        self._callback_handlers.append(handler)

    def register_document_handler(self, handler: MessageHandler) -> None:
        """Register a handler for incoming document uploads."""
        self._document_handlers.append(handler)

    def build_inline_keyboard(self, buttons: InlineKeyboardSpec) -> Any:
        """Build Telegram inline keyboard markup for shared workflows."""
        return build_inline_keyboard(buttons)

    async def download_file(self, file_id: str) -> bytes:
        """Download a file from Telegram servers by file_id."""
        if self._app is None:
            raise MessengerError("Messenger not started")

        tg_file = await self._app.bot.get_file(file_id)
        byte_array = await tg_file.download_as_bytearray()
        return bytes(byte_array)

    def get_callback_source_message(self, message: IncomingMessage) -> CallbackSourceMessage | None:
        callback_query = getattr(message.raw, "callback_query", None)
        callback_message = getattr(callback_query, "message", None)
        if callback_message is None:
            return None

        original_html = getattr(callback_message, "text_html", None)
        if original_html is not None:
            return CallbackSourceMessage(
                message_id=str(callback_message.message_id),
                text=original_html,
                parse_mode="html",
            )

        return CallbackSourceMessage(
            message_id=str(callback_message.message_id),
            text=getattr(callback_message, "text", "") or "",
            parse_mode=None,
        )

    async def set_profile_photo(self, photo_path: str) -> bool:
        """Set the bot's profile photo."""
        if self._app is None:
            return False

        try:
            with Path(photo_path).open("rb") as photo:
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

    def _make_command_wrapper(self, handler: MessageHandler) -> Any:
        """Create a Telegram CommandHandler-compatible wrapper."""

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            _ = context
            msg = convert_update_to_message(update, bot_id=self._bot_id)
            if msg:
                await handler(msg)

        return wrapper

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming document uploads."""
        _ = context
        msg = convert_update_to_message(update, bot_id=self._bot_id)
        if msg:
            for handler in self._document_handlers:
                await handler(msg)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        _ = context
        msg = convert_update_to_message(update, bot_id=self._bot_id)
        if msg:
            for handler in self._message_handlers:
                await handler(msg)

    async def _handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback queries (button presses)."""
        _ = context
        msg = convert_update_to_message(update, bot_id=self._bot_id)
        if msg:
            # Acknowledge the callback to remove the loading indicator
            if update.callback_query:
                try:
                    await update.callback_query.answer()
                except (TelegramError, OSError) as e:
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
