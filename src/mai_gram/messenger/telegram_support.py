"""Internal helpers for the Telegram messenger implementation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from mai_gram.messenger.base import (
    IncomingMessage,
    MessageType,
    OutgoingMessage,
    SendResult,
)
from mai_gram.messenger.base import (
    MessageHandler as IncomingMessageHandler,
)

if TYPE_CHECKING:
    import logging

TelegramApplication = Application[Any, Any, Any, Any, Any, Any]
ReplyMarkup = ReplyKeyboardRemove | InlineKeyboardMarkup | ReplyKeyboardMarkup | None
TelegramUpdateHandler = Callable[[Update, Any], Coroutine[Any, Any, Any]]

_RETRY_AFTER_RE = re.compile(r"retry in (\d+)")


def convert_update_to_message(update: Update, *, bot_id: str = "") -> IncomingMessage | None:
    """Convert a Telegram update into the shared incoming message model."""
    if update.callback_query:
        return _convert_callback_query(update, bot_id=bot_id)

    message = update.message or update.edited_message
    if not message:
        return None

    message_type, command, command_args = _classify_message(message)
    document_file_id, document_file_name, document_mime_type, document_file_size = (
        _extract_document_fields(message)
    )
    return IncomingMessage(
        platform="telegram",
        chat_id=str(message.chat_id),
        user_id=str(message.from_user.id) if message.from_user else "",
        message_id=str(message.message_id),
        message_type=message_type,
        text=message.text or message.caption or "",
        command=command,
        command_args=command_args,
        timestamp=message.date,
        bot_id=bot_id,
        document_file_id=document_file_id,
        document_file_name=document_file_name,
        document_mime_type=document_mime_type,
        document_file_size=document_file_size,
        raw=update,
    )


def build_inline_keyboard(buttons: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    """Build an inline keyboard from row-oriented button tuples."""
    keyboard = [
        [InlineKeyboardButton(text, callback_data=data) for text, data in row] for row in buttons
    ]
    return InlineKeyboardMarkup(keyboard)


def build_reply_keyboard(
    buttons: list[list[str]],
    *,
    one_time: bool = True,
    resize: bool = True,
) -> ReplyKeyboardMarkup:
    """Build a reply keyboard from row-oriented button text."""
    return ReplyKeyboardMarkup(
        buttons,
        one_time_keyboard=one_time,
        resize_keyboard=resize,
    )


def build_application(token: str) -> TelegramApplication:
    """Create the Telegram application with resilient network timeouts."""
    return (
        Application.builder()
        .token(token)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .connect_timeout(15.0)
        .pool_timeout(10.0)
        .get_updates_read_timeout(30.0)
        .get_updates_write_timeout(30.0)
        .get_updates_connect_timeout(15.0)
        .build()
    )


async def resolve_bot_id(app: TelegramApplication, bot_id: str) -> str:
    """Resolve the Telegram bot identifier when the caller did not supply one."""
    if bot_id:
        return bot_id
    bot_info = await app.bot.get_me()
    return bot_info.username or str(bot_info.id)


def register_handlers(
    app: TelegramApplication,
    *,
    command_handlers: dict[str, IncomingMessageHandler],
    callback_handlers: list[IncomingMessageHandler],
    document_handlers: list[IncomingMessageHandler],
    message_handlers: list[IncomingMessageHandler],
    make_command_wrapper: Callable[[IncomingMessageHandler], Any],
    handle_callback_query: TelegramUpdateHandler,
    handle_document: TelegramUpdateHandler,
    handle_message: TelegramUpdateHandler,
) -> None:
    """Register the active Telegram handlers onto the application."""
    for command, handler in command_handlers.items():
        app.add_handler(CommandHandler(command, make_command_wrapper(handler)))

    if callback_handlers:
        app.add_handler(CallbackQueryHandler(handle_callback_query))

    if document_handlers:
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    if message_handlers:
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


async def register_bot_commands(
    app: TelegramApplication,
    command_descriptions: dict[str, str],
    *,
    logger: logging.Logger,
) -> None:
    """Register slash-command descriptions with Telegram when configured."""
    if not command_descriptions:
        return

    bot_commands = [
        BotCommand(command, description) for command, description in command_descriptions.items()
    ]
    try:
        await app.bot.set_my_commands(bot_commands)
        logger.info("Registered %d command(s) with Telegram", len(bot_commands))
    except TelegramError as error:
        logger.warning("Failed to set bot commands: %s", error)


def get_parse_mode(parse_mode: str | None) -> str | None:
    """Map the generic parse mode strings onto Telegram parse modes."""
    if not parse_mode:
        return None
    normalized = parse_mode.lower()
    if normalized == "markdown":
        return ParseMode.MARKDOWN_V2
    if normalized == "html":
        return ParseMode.HTML
    return None


def build_reply_markup(keyboard: Any) -> ReplyMarkup:
    """Normalize the outgoing keyboard value into Telegram reply markup."""
    if keyboard is None:
        return None
    if keyboard == "remove":
        return ReplyKeyboardRemove()
    if isinstance(keyboard, (InlineKeyboardMarkup, ReplyKeyboardMarkup)):
        return keyboard
    if isinstance(keyboard, list):
        return build_inline_keyboard(keyboard)
    return None


async def send_message_with_retry(
    bot: Bot,
    message: OutgoingMessage,
    *,
    max_retries: int,
    logger: logging.Logger,
) -> SendResult:
    """Send a Telegram message with retry handling for transient failures."""
    parse_mode = get_parse_mode(message.parse_mode)
    reply_markup = build_reply_markup(message.keyboard)
    reply_to_message_id = int(message.reply_to) if message.reply_to else None
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        try:
            sent_message_id = await _send_once(
                bot,
                message,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                reply_to_message_id=reply_to_message_id,
            )
            return SendResult(success=True, message_id=str(sent_message_id))
        except TelegramError as error:
            last_error = error
            retry_result = await _handle_send_error(
                error,
                attempt=attempt,
                max_retries=max_retries,
                logger=logger,
            )
            if retry_result is None:
                return SendResult(success=False, error=str(error))
            if retry_result:
                continue
            logger.error("Failed to send Telegram message: %s", error)
            return SendResult(success=False, error=str(error))

    logger.error("Failed to send Telegram message after retries: %s", last_error)
    return SendResult(success=False, error=str(last_error) if last_error else "Unknown error")


def extract_retry_after(error_text: str) -> int:
    """Parse the retry-after duration from a Telegram flood-control error."""
    match = _RETRY_AFTER_RE.search(error_text)
    return int(match.group(1)) if match else 30


def should_retry_transient_error(error_text: str) -> bool:
    """Return whether the Telegram error text represents a transient failure."""
    return "timed out" in error_text or "network" in error_text


def is_flood_control_error(error_text: str) -> bool:
    """Return whether the Telegram error text indicates rate limiting."""
    return "flood control" in error_text or "too many requests" in error_text


def _convert_callback_query(update: Update, *, bot_id: str) -> IncomingMessage:
    query = update.callback_query
    assert query is not None
    return IncomingMessage(
        platform="telegram",
        chat_id=str(query.message.chat_id) if query.message else "",  # type: ignore[attr-defined]
        user_id=str(query.from_user.id) if query.from_user else "",
        message_id=str(query.id),
        message_type=MessageType.CALLBACK,
        callback_data=query.data,
        timestamp=datetime.now(timezone.utc),
        bot_id=bot_id,
        raw=update,
    )


def _classify_message(message: Any) -> tuple[MessageType, str | None, str | None]:
    if message.text and message.text.startswith("/"):
        parts = message.text.split(maxsplit=1)
        return MessageType.COMMAND, parts[0][1:].split("@")[0], parts[1] if len(parts) > 1 else None
    if message.text:
        return MessageType.TEXT, None, None
    if message.photo:
        return MessageType.PHOTO, None, None
    if message.voice:
        return MessageType.VOICE, None, None
    if message.document:
        return MessageType.DOCUMENT, None, None
    return MessageType.OTHER, None, None


def _extract_document_fields(
    message: Any,
) -> tuple[str | None, str | None, str | None, int | None]:
    if not message.document:
        return None, None, None, None
    return (
        message.document.file_id,
        message.document.file_name,
        message.document.mime_type,
        message.document.file_size,
    )


async def _send_once(
    bot: Bot,
    message: OutgoingMessage,
    *,
    parse_mode: str | None,
    reply_markup: ReplyMarkup,
    reply_to_message_id: int | None,
) -> int:
    if message.photo_path:
        with open(message.photo_path, "rb") as photo:
            sent = await bot.send_photo(
                chat_id=int(message.chat_id),
                photo=photo,
                caption=message.text if message.text else None,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                reply_to_message_id=reply_to_message_id,
            )
        return int(sent.message_id)

    if message.photo_url:
        sent = await bot.send_photo(
            chat_id=int(message.chat_id),
            photo=message.photo_url,
            caption=message.text if message.text else None,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            reply_to_message_id=reply_to_message_id,
        )
        return int(sent.message_id)

    sent = await bot.send_message(
        chat_id=int(message.chat_id),
        text=message.text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
        reply_to_message_id=reply_to_message_id,
    )
    return int(sent.message_id)


async def _handle_send_error(
    error: TelegramError,
    *,
    attempt: int,
    max_retries: int,
    logger: logging.Logger,
) -> bool | None:
    error_text = str(error).lower()
    if is_flood_control_error(error_text):
        retry_after = extract_retry_after(error_text)
        logger.warning(
            "Flood control hit (attempt %d/%d): retry in %ds",
            attempt,
            max_retries + 1,
            retry_after,
        )
        if attempt <= max_retries:
            await asyncio.sleep(retry_after)
            return True
        return None

    if should_retry_transient_error(error_text) and attempt <= max_retries:
        logger.warning(
            "Telegram send failed (attempt %d/%d): %s - retrying...",
            attempt,
            max_retries + 1,
            error,
        )
        await asyncio.sleep(1.0 * attempt)
        return True

    return False
