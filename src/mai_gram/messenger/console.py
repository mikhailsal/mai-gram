"""Console messenger implementation for scripted/local testing."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from typing import Any, TextIO

from mai_gram.messenger.base import (
    IncomingMessage,
    MessageHandler,
    MessageType,
    Messenger,
    OutgoingMessage,
    SendResult,
)


def _extract_buttons(keyboard: Any) -> list[tuple[str, str]]:
    """Normalize different keyboard payload shapes into text/callback pairs."""
    if keyboard is None:
        return []

    # Handle Telegram InlineKeyboardMarkup object (has inline_keyboard attribute)
    if hasattr(keyboard, "inline_keyboard"):
        keyboard = keyboard.inline_keyboard

    # Telegram InlineKeyboardMarkup-like dict payload.
    if isinstance(keyboard, dict) and isinstance(keyboard.get("inline_keyboard"), list):
        keyboard = keyboard["inline_keyboard"]

    rows: list[Any]
    if isinstance(keyboard, (list, tuple)):
        rows = list(keyboard)
    else:
        return []

    buttons: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, tuple) and len(row) == 2:
            # Check if it's a (text, callback) tuple or a row of buttons
            first, second = row
            if isinstance(first, str) and isinstance(second, str):
                buttons.append((first, second))
                continue
            # Otherwise treat as a row of button objects
            row = list(row)

        if not isinstance(row, (list, tuple)):
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
                continue

            # Handle Telegram InlineKeyboardButton objects (have text and callback_data attrs)
            if hasattr(item, "text") and hasattr(item, "callback_data"):
                text = item.text
                callback = item.callback_data
                if text is not None and callback is not None:
                    buttons.append((str(text), str(callback)))
    return buttons


class ConsoleMessenger(Messenger):
    """Non-interactive messenger that prints messages to stdout.

    By default, intermediate streaming edits are buffered and only
    the final version of each message is printed (when a new message
    arrives or ``flush_edits()`` is called). Pass ``stream_debug=True``
    to print every edit for debugging the streaming behaviour itself.
    """

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        stream_debug: bool = False,
    ) -> None:
        import sys

        self._output = output or sys.stdout
        self._stream_debug = stream_debug
        self._message_handlers: list[MessageHandler] = []
        self._callback_handlers: list[MessageHandler] = []
        self._command_handlers: dict[str, MessageHandler] = {}
        self._id_counter = count(start=1)
        self._pending_edits: dict[str, tuple[str, dict[str, Any]]] = {}

    @property
    def platform_name(self) -> str:
        return "console"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def _print_message(self, header: str, text: str, kwargs: dict[str, Any] | None = None) -> None:
        """Print a message block with optional parse_mode and buttons."""
        print(header, file=self._output)
        parse_mode = (kwargs or {}).get("parse_mode")
        if parse_mode:
            print(f"[parse_mode={parse_mode}]", file=self._output)
        print(text, file=self._output)
        buttons = _extract_buttons((kwargs or {}).get("keyboard"))
        if buttons:
            print("", file=self._output)
            print("--- Buttons ---", file=self._output)
            for index, (bt, cb) in enumerate(buttons, start=1):
                print(f"[{index}] {bt}  ->  {cb}", file=self._output)

    def flush_edits(self) -> None:
        """Print all buffered edits. Call after the handler finishes."""
        for msg_id, (text, kwargs) in self._pending_edits.items():
            self._print_message(f"--- AI Response (final edit of {msg_id}) ---", text, kwargs)
        self._pending_edits.clear()

    async def send_message(self, message: OutgoingMessage) -> SendResult:
        self.flush_edits()
        self._print_message(
            "--- AI Response ---",
            message.text,
            {
                "parse_mode": message.parse_mode,
                "keyboard": message.keyboard,
            },
        )

        message_id = f"console-{next(self._id_counter)}"
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self, chat_id: str, message_id: str, new_text: str, **kwargs: Any
    ) -> SendResult:
        del chat_id
        if self._stream_debug:
            self._print_message(
                f"--- Edited AI Response (replaces {message_id}) ---",
                new_text,
                kwargs,
            )
        else:
            self._pending_edits[message_id] = (new_text, dict(kwargs))
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

    def register_command_handler(
        self,
        command: str,
        handler: MessageHandler,
        *,
        description: str = "",
    ) -> None:
        self._command_handlers[command] = handler

    def register_callback_handler(self, handler: MessageHandler) -> None:
        self._callback_handlers.append(handler)

    def register_document_handler(self, handler: MessageHandler) -> None:
        pass

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
