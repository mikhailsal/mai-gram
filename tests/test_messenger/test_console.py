"""Tests for the console messenger implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage
from mai_gram.messenger.console import ConsoleMessenger


class TestConsoleMessenger:
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
        messenger = ConsoleMessenger()

        await messenger.edit_message(
            chat_id="chat-1",
            message_id="console-3",
            new_text="Updated response text",
        )

        captured = capsys.readouterr().out
        assert "replaces message console-3" in captured
        assert "Updated response text" in captured

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
