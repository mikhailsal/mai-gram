"""Tests for the console messenger implementation."""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock

from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage
from mai_gram.messenger.console import ConsoleMessenger, _extract_buttons


class _InlineKeyboardButton:
    def __init__(self, text: str | None, callback_data: str | None) -> None:
        self.text = text
        self.callback_data = callback_data


class _InlineKeyboardMarkup:
    def __init__(self, inline_keyboard: object) -> None:
        self.inline_keyboard = inline_keyboard


class TestConsoleMessenger:
    def test_extract_buttons_supports_keyboard_variants(self) -> None:
        assert _extract_buttons(None) == []
        assert _extract_buttons("not-a-keyboard") == []
        assert _extract_buttons([123]) == []
        assert _extract_buttons([("Direct", "direct:cb")]) == [("Direct", "direct:cb")]

        assert _extract_buttons(
            {"inline_keyboard": [[{"text": "Dict", "callback_data": "dict:cb"}]]}
        ) == [("Dict", "dict:cb")]

        assert _extract_buttons(
            _InlineKeyboardMarkup(
                [[_InlineKeyboardButton("Obj", "obj:cb"), _InlineKeyboardButton(None, None)]]
            )
        ) == [("Obj", "obj:cb")]

        tuple_row_buttons = _extract_buttons(
            [(_InlineKeyboardButton("One", "one:cb"), _InlineKeyboardButton("Two", "two:cb"))]
        )
        assert tuple_row_buttons == [("One", "one:cb"), ("Two", "two:cb")]

    async def test_send_message_renders_text_and_buttons(self, capsys) -> None:
        messenger = ConsoleMessenger()
        result = await messenger.send_message(
            OutgoingMessage(
                text="Hello from console!",
                chat_id="chat-1",
                keyboard=[[("Choose a Preset", "personality:presets")]],
            )
        )

        captured = capsys.readouterr().out
        assert result.success is True
        assert captured.startswith("--- AI Response ---")
        assert "Hello from console!" in captured
        assert "--- Buttons ---" in captured
        assert "[1] Choose a Preset  ->  personality:presets" in captured

    async def test_edit_message_renders_replacement_note(self, capsys) -> None:
        messenger = ConsoleMessenger(stream_debug=True)

        await messenger.edit_message(
            chat_id="chat-1",
            message_id="console-3",
            new_text="Updated response text",
        )

        captured = capsys.readouterr().out
        assert "replaces console-3" in captured
        assert "Updated response text" in captured

    async def test_edit_message_buffered_by_default(self, capsys) -> None:
        messenger = ConsoleMessenger()

        await messenger.edit_message(
            chat_id="chat-1",
            message_id="console-3",
            new_text="Buffered text",
        )

        captured = capsys.readouterr().out
        assert captured == ""

        messenger.flush_edits()
        captured = capsys.readouterr().out
        assert "Buffered text" in captured

    async def test_send_message_flushes_edits_and_renders_parse_mode(self) -> None:
        output = StringIO()
        messenger = ConsoleMessenger(output=output)

        await messenger.edit_message(
            chat_id="chat-1",
            message_id="console-4",
            new_text="Buffered HTML",
            parse_mode="HTML",
        )
        await messenger.send_message(
            OutgoingMessage(
                text="Fresh text",
                chat_id="chat-1",
                parse_mode="MarkdownV2",
                keyboard={"inline_keyboard": [[{"text": "Pick", "callback_data": "pick:1"}]]},
            )
        )

        rendered = output.getvalue()
        assert "final edit of console-4" in rendered
        assert "[parse_mode=HTML]" in rendered
        assert "[parse_mode=MarkdownV2]" in rendered
        assert "[1] Pick  ->  pick:1" in rendered

    async def test_start_stop_delete_and_document_registration_are_noops(self) -> None:
        output = StringIO()
        messenger = ConsoleMessenger(output=output)

        assert messenger.platform_name == "console"
        assert await messenger.start() is None
        assert await messenger.stop() is None
        assert await messenger.send_typing_indicator("chat-1") is None
        assert await messenger.delete_message("chat-1", "console-9") is True

        messenger.register_document_handler(AsyncMock())

        assert "--- Deleted Message --- console-9" in output.getvalue()

    async def test_dispatch_routes_to_registered_handlers(self) -> None:
        messenger = ConsoleMessenger()
        command_handler = AsyncMock()
        message_handler = AsyncMock()
        callback_handler = AsyncMock()

        messenger.register_command_handler("start", command_handler)
        messenger.register_message_handler(message_handler)
        messenger.register_callback_handler(callback_handler)

        await messenger.dispatch_message(
            IncomingMessage(
                platform="console",
                chat_id="chat-1",
                user_id="user-1",
                message_id="m1",
                message_type=MessageType.COMMAND,
                command="start",
                text="/start",
            )
        )
        await messenger.dispatch_message(
            IncomingMessage(
                platform="console",
                chat_id="chat-1",
                user_id="user-1",
                message_id="m2",
                message_type=MessageType.TEXT,
                text="hello",
            )
        )
        await messenger.dispatch_message(
            IncomingMessage(
                platform="console",
                chat_id="chat-1",
                user_id="user-1",
                message_id="m3",
                message_type=MessageType.CALLBACK,
                callback_data="preset:balanced",
            )
        )

        command_handler.assert_awaited_once()
        message_handler.assert_awaited_once()
        callback_handler.assert_awaited_once()

    async def test_dispatch_helpers_build_incoming_messages(self) -> None:
        messenger = ConsoleMessenger(output=StringIO())
        message_handler = AsyncMock()
        callback_handler = AsyncMock()

        messenger.register_message_handler(message_handler)
        messenger.register_callback_handler(callback_handler)

        await messenger.dispatch_text(chat_id="chat-1", user_id="user-1", text="hello")
        await messenger.dispatch_callback(
            chat_id="chat-1",
            user_id="user-1",
            callback_data="preset:balanced",
        )

        text_message = message_handler.await_args.args[0]
        callback_message = callback_handler.await_args.args[0]
        assert text_message.platform == "console"
        assert text_message.chat_id == "chat-1"
        assert text_message.user_id == "user-1"
        assert text_message.text == "hello"
        assert text_message.message_type == MessageType.TEXT
        assert text_message.message_id.startswith("in-")
        assert callback_message.platform == "console"
        assert callback_message.callback_data == "preset:balanced"
        assert callback_message.message_type == MessageType.CALLBACK
        assert callback_message.message_id.startswith("cb-")
