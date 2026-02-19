"""Console messenger implementation for scripted/local testing."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from typing import Any, TextIO

from mai_companion.messenger.base import (
    IncomingMessage,
    MessageHandler,
    Messenger,
    MessageType,
    OutgoingMessage,
    SendResult,
)


def _extract_buttons(keyboard: Any) -> list[tuple[str, str]]:
    """Normalize different keyboard payload shapes into text/callback pairs."""
    if keyboard is None:
        return []

    # Telegram InlineKeyboardMarkup-like dict payload.
    if isinstance(keyboard, dict) and isinstance(keyboard.get("inline_keyboard"), list):
        keyboard = keyboard["inline_keyboard"]

    rows: list[Any]
    if isinstance(keyboard, list):
        rows = keyboard
    else:
        return []

    buttons: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, tuple) and len(row) == 2:
            text, callback = row
            buttons.append((str(text), str(callback)))
            continue

        if not isinstance(row, list):
            continue

        for item in row:
            if isinstance(item, tuple) and len(item) == 2:
                text, callback = item
                buttons.append((str(text), str(callback)))
                continue

            if isinstance(item, dict):
                text = item.get("text")
                callback = item.get("callback_data")
                if text is not None and callback is not None:
                    buttons.append((str(text), str(callback)))
    return buttons


class ConsoleMessenger(Messenger):
    """Non-interactive messenger that prints messages to stdout."""

    def __init__(self, *, output: TextIO | None = None) -> None:
        import sys

        self._output = output or sys.stdout
        self._message_handlers: list[MessageHandler] = []
        self._callback_handlers: list[MessageHandler] = []
        self._command_handlers: dict[str, MessageHandler] = {}
        self._id_counter = count(start=1)

    @property
    def platform_name(self) -> str:
        return "console"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_message(self, message: OutgoingMessage) -> SendResult:
        print("--- AI Response ---", file=self._output)
        print(message.text, file=self._output)

        buttons = _extract_buttons(message.keyboard)
        if buttons:
            print("", file=self._output)
            print("--- Buttons ---", file=self._output)
            for index, (text, callback_data) in enumerate(buttons, start=1):
                print(f"[{index}] {text}  ->  {callback_data}", file=self._output)

        message_id = f"console-{next(self._id_counter)}"
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        del chat_id
        print(
            f"--- Edited AI Response (replaces message {message_id}) ---",
            file=self._output,
        )
        print(new_text, file=self._output)

        buttons = _extract_buttons(kwargs.get("keyboard"))
        if buttons:
            print("", file=self._output)
            print("--- Buttons ---", file=self._output)
            for index, (text, callback_data) in enumerate(buttons, start=1):
                print(f"[{index}] {text}  ->  {callback_data}", file=self._output)

        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        del chat_id
        print(f"--- Deleted Message --- {message_id}", file=self._output)
        return True

    async def send_typing_indicator(self, chat_id: str) -> None:
        del chat_id
        return None

    def register_message_handler(self, handler: MessageHandler) -> None:
        self._message_handlers.append(handler)

    def register_command_handler(self, command: str, handler: MessageHandler) -> None:
        self._command_handlers[command] = handler

    def register_callback_handler(self, handler: MessageHandler) -> None:
        self._callback_handlers.append(handler)

    async def dispatch_message(self, message: IncomingMessage) -> None:
        """Route a synthetic incoming message through registered handlers."""
        if message.message_type == MessageType.COMMAND:
            if message.command and message.command in self._command_handlers:
                await self._command_handlers[message.command](message)
            return

        if message.message_type == MessageType.CALLBACK:
            for handler in self._callback_handlers:
                await handler(message)
            return

        for handler in self._message_handlers:
            await handler(message)

    async def dispatch_text(
        self,
        *,
        chat_id: str,
        user_id: str,
        text: str,
        message_id: str | None = None,
    ) -> None:
        """Convenience helper for injecting text input in scripts/tests."""
        incoming = IncomingMessage(
            platform=self.platform_name,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id or f"in-{next(self._id_counter)}",
            message_type=MessageType.TEXT,
            text=text,
            timestamp=datetime.now(timezone.utc),
        )
        await self.dispatch_message(incoming)

    async def dispatch_callback(
        self,
        *,
        chat_id: str,
        user_id: str,
        callback_data: str,
        message_id: str | None = None,
    ) -> None:
        """Convenience helper for injecting callback button events."""
        incoming = IncomingMessage(
            platform=self.platform_name,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id or f"cb-{next(self._id_counter)}",
            message_type=MessageType.CALLBACK,
            callback_data=callback_data,
            timestamp=datetime.now(timezone.utc),
        )
        await self.dispatch_message(incoming)
