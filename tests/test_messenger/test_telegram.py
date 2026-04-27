"""Tests for Telegram messenger support helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import TelegramError
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

from mai_gram.messenger.base import MessageType, OutgoingMessage
from mai_gram.messenger.telegram_support import (
    build_inline_keyboard,
    build_reply_keyboard,
    build_reply_markup,
    convert_update_to_message,
    extract_retry_after,
    get_parse_mode,
    register_bot_commands,
    register_handlers,
    resolve_bot_id,
    send_message_with_retry,
)

if TYPE_CHECKING:
    from pathlib import Path

    from telegram import Update


def test_convert_update_to_message_parses_command_document_and_callback() -> None:
    callback_update = SimpleNamespace(
        callback_query=SimpleNamespace(
            message=SimpleNamespace(chat_id=42),
            from_user=SimpleNamespace(id=99),
            id="cb-1",
            data="pick:1",
        ),
        message=None,
        edited_message=None,
    )
    command_update = SimpleNamespace(
        callback_query=None,
        message=SimpleNamespace(
            chat_id=42,
            from_user=SimpleNamespace(id=99),
            message_id=7,
            text="/start hello",
            caption=None,
            photo=None,
            voice=None,
            document=None,
            date=datetime.now(timezone.utc),
        ),
        edited_message=None,
    )
    document_update = SimpleNamespace(
        callback_query=None,
        message=SimpleNamespace(
            chat_id=42,
            from_user=SimpleNamespace(id=99),
            message_id=8,
            text=None,
            caption="attached",
            photo=None,
            voice=None,
            document=SimpleNamespace(
                file_id="file-1",
                file_name="notes.txt",
                mime_type="text/plain",
                file_size=12,
            ),
            date=datetime.now(timezone.utc),
        ),
        edited_message=None,
    )

    callback_message = convert_update_to_message(cast("Update", callback_update), bot_id="bot-a")
    command_message = convert_update_to_message(cast("Update", command_update), bot_id="bot-a")
    document_message = convert_update_to_message(cast("Update", document_update), bot_id="bot-a")

    assert callback_message is not None
    assert callback_message.message_type is MessageType.CALLBACK
    assert callback_message.callback_data == "pick:1"
    assert command_message is not None
    assert command_message.message_type is MessageType.COMMAND
    assert command_message.command == "start"
    assert command_message.command_args == "hello"
    assert document_message is not None
    assert document_message.message_type is MessageType.DOCUMENT
    assert document_message.text == "attached"
    assert document_message.document_file_name == "notes.txt"


def test_build_reply_markup_and_parse_mode_helpers() -> None:
    inline_markup = build_reply_markup([[("One", "1")]])

    assert get_parse_mode("markdown") == "MarkdownV2"
    assert get_parse_mode("html") == "HTML"
    assert get_parse_mode(None) is None
    assert isinstance(inline_markup, InlineKeyboardMarkup)
    assert isinstance(build_reply_markup("remove"), ReplyKeyboardRemove)
    assert build_reply_markup(None) is None
    assert extract_retry_after("flood control: retry in 17 seconds") == 17


@pytest.mark.asyncio
async def test_resolve_bot_id_and_register_bot_commands() -> None:
    app = MagicMock()
    app.bot.get_me = AsyncMock(return_value=SimpleNamespace(username="mai_bot", id=123))
    app.bot.set_my_commands = AsyncMock()

    resolved = await resolve_bot_id(app, "")
    await register_bot_commands(app, {"start": "Start the bot"}, logger=MagicMock())

    assert resolved == "mai_bot"
    app.bot.set_my_commands.assert_awaited_once()


def test_register_handlers_adds_expected_handler_types() -> None:
    app = MagicMock()
    make_command_wrapper = MagicMock(return_value=AsyncMock())

    register_handlers(
        app,
        command_handlers={"start": AsyncMock()},
        callback_handlers=[AsyncMock()],
        document_handlers=[AsyncMock()],
        message_handlers=[AsyncMock()],
        make_command_wrapper=make_command_wrapper,
        handle_callback_query=AsyncMock(),
        handle_document=AsyncMock(),
        handle_message=AsyncMock(),
    )

    added_handlers = [call.args[0] for call in app.add_handler.call_args_list]
    assert len(added_handlers) == 4
    assert isinstance(added_handlers[0], CommandHandler)
    assert isinstance(added_handlers[1], CallbackQueryHandler)
    assert isinstance(added_handlers[2], MessageHandler)
    assert isinstance(added_handlers[3], MessageHandler)


@pytest.mark.asyncio
async def test_send_message_with_retry_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("mai_gram.messenger.telegram_support.asyncio.sleep", sleep)
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=[TelegramError("Timed out"), SimpleNamespace(message_id=42)]
    )

    result = await send_message_with_retry(
        bot,
        OutgoingMessage(text="hello", chat_id="7", parse_mode="html"),
        max_retries=2,
        logger=MagicMock(),
    )

    assert result.success is True
    assert result.message_id == "42"
    assert bot.send_message.await_count == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_send_message_with_retry_retries_flood_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("mai_gram.messenger.telegram_support.asyncio.sleep", sleep)
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=[
            TelegramError("Too many requests, retry in 3 seconds"),
            SimpleNamespace(message_id=9),
        ]
    )

    result = await send_message_with_retry(
        bot,
        OutgoingMessage(text="hello", chat_id="7", keyboard=[[("One", "1")]]),
        max_retries=1,
        logger=MagicMock(),
    )

    assert result.success is True
    assert result.message_id == "9"
    sleep.assert_awaited_once_with(3)


def test_build_inline_keyboard_preserves_rows() -> None:
    keyboard = build_inline_keyboard([[("One", "1")], [("Two", "2"), ("Three", "3")]])

    assert keyboard.inline_keyboard[0][0].text == "One"
    assert keyboard.inline_keyboard[1][1].callback_data == "3"


def test_convert_update_to_message_handles_text_voice_and_non_message() -> None:
    text_update = SimpleNamespace(
        callback_query=None,
        message=SimpleNamespace(
            chat_id=42,
            from_user=SimpleNamespace(id=99),
            message_id=9,
            text="hello",
            caption=None,
            photo=None,
            voice=None,
            document=None,
            date=datetime.now(timezone.utc),
        ),
        edited_message=None,
    )
    voice_update = SimpleNamespace(
        callback_query=None,
        message=SimpleNamespace(
            chat_id=42,
            from_user=SimpleNamespace(id=99),
            message_id=10,
            text=None,
            caption=None,
            photo=None,
            voice=object(),
            document=None,
            date=datetime.now(timezone.utc),
        ),
        edited_message=None,
    )
    empty_update = SimpleNamespace(callback_query=None, message=None, edited_message=None)

    text_message = convert_update_to_message(cast("Update", text_update), bot_id="bot-a")
    voice_message = convert_update_to_message(cast("Update", voice_update), bot_id="bot-a")

    assert text_message is not None
    assert text_message.message_type is MessageType.TEXT
    assert voice_message is not None
    assert voice_message.message_type is MessageType.VOICE
    assert convert_update_to_message(cast("Update", empty_update), bot_id="bot-a") is None


def test_build_reply_keyboard_and_markup_passthrough() -> None:
    reply_keyboard = build_reply_keyboard([["Yes", "No"]], one_time=False, resize=False)

    assert isinstance(reply_keyboard, ReplyKeyboardMarkup)
    assert build_reply_markup(reply_keyboard) is reply_keyboard
    assert build_reply_markup(object()) is None
    assert get_parse_mode("plain") is None


@pytest.mark.asyncio
async def test_register_bot_commands_handles_empty_and_error() -> None:
    app = MagicMock()
    app.bot.set_my_commands = AsyncMock(side_effect=TelegramError("boom"))
    logger = MagicMock()

    await register_bot_commands(app, {}, logger=logger)
    app.bot.set_my_commands.assert_not_awaited()

    await register_bot_commands(app, {"start": "Start the bot"}, logger=logger)

    logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_send_message_with_retry_sends_photo_path(tmp_path: Path) -> None:
    photo_path = tmp_path / "photo.jpg"
    photo_path.write_bytes(b"image-bytes")
    bot = MagicMock()
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=13))

    result = await send_message_with_retry(
        bot,
        OutgoingMessage(text="caption", chat_id="7", photo_path=str(photo_path)),
        max_retries=0,
        logger=MagicMock(),
    )

    assert result.success is True
    assert result.message_id == "13"
    bot.send_photo.assert_awaited_once()
