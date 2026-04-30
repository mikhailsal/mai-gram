"""Tests for Telegram messenger support helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.error import TelegramError
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

from mai_gram.messenger.base import (
    CallbackSourceMessage,
    IncomingMessage,
    MessageType,
    MessengerError,
    OutgoingMessage,
    SendResult,
)
from mai_gram.messenger.telegram import TelegramMessenger, answer_callback_query
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


@pytest.mark.asyncio
async def test_telegram_messenger_start_and_stop_manage_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(
        bot=SimpleNamespace(),
        updater=SimpleNamespace(running=True, start_polling=AsyncMock(), stop=AsyncMock()),
        initialize=AsyncMock(),
        start=AsyncMock(),
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    register_handlers_mock = MagicMock()
    register_bot_commands_mock = AsyncMock()
    monkeypatch.setattr(
        "mai_gram.messenger.telegram.build_application",
        MagicMock(return_value=app),
    )
    monkeypatch.setattr(
        "mai_gram.messenger.telegram.resolve_bot_id",
        AsyncMock(return_value="mai_bot"),
    )
    monkeypatch.setattr("mai_gram.messenger.telegram.register_handlers", register_handlers_mock)
    monkeypatch.setattr(
        "mai_gram.messenger.telegram.register_bot_commands",
        register_bot_commands_mock,
    )

    messenger = TelegramMessenger("token")
    messenger.register_message_handler(AsyncMock())
    messenger.register_command_handler("start", AsyncMock(), description="Start the bot")
    messenger.register_callback_handler(AsyncMock())
    messenger.register_document_handler(AsyncMock())

    await messenger.start()
    await messenger.stop()

    assert messenger.bot_id == "mai_bot"
    app.initialize.assert_awaited_once()
    app.start.assert_awaited_once()
    app.updater.start_polling.assert_awaited_once_with(drop_pending_updates=True)
    register_handlers_mock.assert_called_once()
    register_bot_commands_mock.assert_awaited_once_with(app, {"start": "Start the bot"}, logger=ANY)
    app.updater.stop.assert_awaited_once()
    app.stop.assert_awaited_once()
    app.shutdown.assert_awaited_once()
    assert messenger._app is None


@pytest.mark.asyncio
async def test_telegram_messenger_bot_property_and_send_message_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messenger = TelegramMessenger("token", bot_id="bot-a")

    with pytest.raises(MessengerError, match="Messenger not started"):
        _ = messenger.bot

    not_started_result = await messenger.send_message(OutgoingMessage(text="hello", chat_id="7"))
    assert not_started_result == SendResult(success=False, error="Messenger not started")

    fake_bot = SimpleNamespace()
    app_stub: Any = SimpleNamespace(bot=fake_bot)
    messenger._app = app_stub
    delegate = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    monkeypatch.setattr("mai_gram.messenger.telegram.send_message_with_retry", delegate)

    result = await messenger.send_message(OutgoingMessage(text="hello", chat_id="7"), max_retries=5)

    assert messenger.bot is fake_bot
    assert result == SendResult(success=True, message_id="42")
    delegate.assert_awaited_once_with(fake_bot, ANY, max_retries=5, logger=ANY)


@pytest.mark.asyncio
async def test_telegram_messenger_edit_delete_typing_download_and_photo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    messenger = TelegramMessenger("token")

    assert await messenger.edit_message("7", "9", "hi") == SendResult(
        success=False,
        error="Messenger not started",
    )
    assert await messenger.delete_message("7", "9") is False
    await messenger.send_typing_indicator("7")
    with pytest.raises(MessengerError, match="Messenger not started"):
        await messenger.download_file("file-1")
    assert await messenger.set_profile_photo(str(tmp_path / "missing.jpg")) is False

    download_file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"abc")))
    bot = SimpleNamespace(
        edit_message_text=AsyncMock(),
        delete_message=AsyncMock(),
        send_chat_action=AsyncMock(),
        get_file=AsyncMock(return_value=download_file),
        set_chat_photo=AsyncMock(),
        id=99,
    )
    app_stub: Any = SimpleNamespace(bot=bot, updater=None)
    messenger._app = app_stub
    monkeypatch.setattr(
        "mai_gram.messenger.telegram.get_parse_mode",
        MagicMock(return_value="HTML"),
    )
    monkeypatch.setattr(
        "mai_gram.messenger.telegram.build_reply_markup",
        MagicMock(return_value="markup"),
    )

    assert await messenger.edit_message("7", "9", "hi", parse_mode="html") == SendResult(
        success=True,
        message_id="9",
    )
    assert await messenger.delete_message("7", "9") is True
    await messenger.send_typing_indicator("7")
    assert await messenger.download_file("file-1") == b"abc"

    photo_path = tmp_path / "photo.jpg"
    photo_path.write_bytes(b"image-bytes")
    assert await messenger.set_profile_photo(str(photo_path)) is True

    bot.edit_message_text.side_effect = TelegramError("edit failed")
    bot.delete_message.side_effect = TelegramError("delete failed")
    bot.send_chat_action.side_effect = TelegramError("typing failed")
    bot.set_chat_photo.side_effect = TelegramError("photo failed")

    failed_edit = await messenger.edit_message("7", "9", "bye")
    failed_delete = await messenger.delete_message("7", "9")
    await messenger.send_typing_indicator("7")
    failed_photo = await messenger.set_profile_photo(str(photo_path))

    assert failed_edit == SendResult(success=False, error="edit failed")
    assert failed_delete is False
    assert failed_photo is False
    assert "Failed to send typing indicator" in caplog.text


@pytest.mark.asyncio
async def test_telegram_messenger_internal_handlers_and_callback_helper(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    messenger = TelegramMessenger("token", bot_id="bot-a")
    incoming = SimpleNamespace(message_type=MessageType.TEXT)
    convert_mock = MagicMock(return_value=incoming)
    monkeypatch.setattr("mai_gram.messenger.telegram.convert_update_to_message", convert_mock)

    command_handler = AsyncMock()
    document_handler = AsyncMock()
    message_handler = AsyncMock()
    callback_handler = AsyncMock()

    wrapper = messenger._make_command_wrapper(command_handler)
    messenger.register_document_handler(document_handler)
    messenger.register_message_handler(message_handler)
    messenger.register_callback_handler(callback_handler)

    callback_query = SimpleNamespace(answer=AsyncMock(side_effect=RuntimeError("boom")))
    update: Any = SimpleNamespace(callback_query=callback_query)

    await wrapper(update, None)
    await messenger._handle_document(update, None)
    await messenger._handle_message(update, None)
    await messenger._handle_callback_query(update, None)

    command_handler.assert_awaited_once_with(incoming)
    document_handler.assert_awaited_once_with(incoming)
    message_handler.assert_awaited_once_with(incoming)
    callback_handler.assert_awaited_once_with(incoming)
    assert "Failed to answer callback query" in caplog.text

    callback_query.answer = AsyncMock()
    await answer_callback_query(update, text="ok")
    callback_query.answer.assert_awaited_once_with(text="ok")

    empty_update: Any = SimpleNamespace(callback_query=None)
    await answer_callback_query(empty_update, text="noop")


def test_get_callback_source_message_prefers_html_snapshot() -> None:
    messenger = TelegramMessenger("token")
    callback_message = SimpleNamespace(message_id=77, text_html="<b>Hello</b>", text="Hello")
    incoming = IncomingMessage(
        platform="telegram",
        chat_id="42",
        user_id="7",
        message_id="cb-1",
        message_type=MessageType.CALLBACK,
        raw=SimpleNamespace(callback_query=SimpleNamespace(message=callback_message)),
    )

    result = messenger.get_callback_source_message(incoming)

    assert result == CallbackSourceMessage(message_id="77", text="<b>Hello</b>", parse_mode="html")


@pytest.mark.asyncio
async def test_delete_callback_source_message_deletes_origin_message() -> None:
    messenger = TelegramMessenger("token")
    messenger.delete_message = AsyncMock(return_value=True)
    callback_message = SimpleNamespace(message_id=88, text_html=None, text="Hello")
    incoming = IncomingMessage(
        platform="telegram",
        chat_id="42",
        user_id="7",
        message_id="cb-1",
        message_type=MessageType.CALLBACK,
        raw=SimpleNamespace(callback_query=SimpleNamespace(message=callback_message)),
    )

    result = await messenger.delete_callback_source_message(incoming)

    assert result is True
    messenger.delete_message.assert_awaited_once_with("42", "88")
