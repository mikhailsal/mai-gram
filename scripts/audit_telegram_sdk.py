"""Reproduce the TG-1 observations for python-telegram-bot 22.8 offline."""

# ruff: noqa: S101, T201

from __future__ import annotations

import asyncio
import json
import platform
import secrets
from inspect import signature
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import telegram
from telegram import Bot, Message, TextQuote, Update
from telegram.ext import Application, TypeHandler, Updater
from telegram.request import HTTPXRequest

from mai_gram.messenger.base import IncomingMessage, OutgoingMessage
from mai_gram.messenger.telegram import TelegramMessenger

CHAT = {"id": 100, "type": "private", "first_name": "Example"}
USER = {"id": 100, "is_bot": False, "first_name": "Example"}
BOT = {"id": 999, "is_bot": True, "first_name": "Audit", "username": "audit_bot"}
CELL = {"type": "bold", "text": "42"}
TABLE = {"type": "table", "cells": [[{"text": "Value", "is_header": True}], [{"text": CELL}]]}
RICH = {"blocks": [TABLE]}
MARKDOWN = {"markdown": "| Value |\n| --- |\n| 42 |"}
KEYBOARD = {"inline_keyboard": [[{"text": "Cut", "callback_data": "cut:42"}]]}
STOP = {"chat": CHAT, "message_thread_id": 3, "draft_id": 7}


def message(message_id: int, **fields: Any) -> dict[str, Any]:
    return {"chat": CHAT, "from": USER, "date": 1_700_000_000, "message_id": message_id, **fields}


def wire_updates() -> list[dict[str, Any]]:
    rich_message = message(1, rich_message=RICH)
    updates = [
        {"message": rich_message},
        {
            "message": message(
                2,
                text="What does this value mean?",
                reply_to_message=rich_message,
                quote={"text": "", "rich_texts": [CELL], "position": 6, "is_manual": True},
            )
        },
        {
            "message": message(
                3,
                text="What do you mean?",
                reply_to_message=message(30, text="Plain"),
                quote={"text": "Plain", "position": 0, "is_manual": True},
            )
        },
        {
            "message": message(
                4,
                caption="Photo caption",
                photo=[
                    {
                        "file_id": "photo",
                        "file_unique_id": "unique",
                        "width": 1,
                        "height": 1,
                    }
                ],
            )
        },
        {
            "message": message(
                5,
                text="/start",
                entities=[
                    {
                        "type": "bot_command",
                        "offset": 0,
                        "length": 6,
                    }
                ],
            )
        },
        {"stopped_message_generation": STOP},
        {
            "callback_query": {
                "id": "cut",
                "from": USER,
                "chat_instance": "demo",
                "data": "cut:42",
                "message": message(42, rich_message=RICH, reply_markup=KEYBOARD),
            }
        },
        {"message": message(8, text="Plain text")},
    ]
    return [{"update_id": index, **fields} for index, fields in enumerate(updates, start=1)]


def check_sdk_surface() -> None:
    assert hasattr(Bot, "send_message_draft")
    assert not hasattr(Bot, "send_rich_message") and not hasattr(Bot, "send_rich_message_draft")
    assert all(
        not hasattr(telegram, name)
        for name in (
            "RichMessage",
            "InputRichMessage",
            "MessageGenerationStopped",
        )
    )
    assert "stopped_message_generation" not in Update.ALL_TYPES
    assert not hasattr(Update, "stopped_message_generation")
    assert not hasattr(Message, "rich_message")
    assert not hasattr(TextQuote, "rich_texts")
    assert "can_stop" not in signature(Bot.send_message_draft).parameters
    assert "keep_on_stop" not in signature(Bot.send_message_draft).parameters
    assert "rich_message" not in signature(Bot.edit_message_text).parameters


async def check_inbound(
    app: Application, messenger: TelegramMessenger, received: list[IncomingMessage]
) -> None:
    stops = []

    async def inspect_stop(update: Update, _context: Any) -> None:
        if "stopped_message_generation" in update.api_kwargs:
            stops.append(update.api_kwargs["stopped_message_generation"])

    updates = await app.bot.get_updates(
        allowed_updates=["message", "callback_query", "stopped_message_generation"]
    )
    for update in updates:
        await app.process_update(update)
    assert [item.message_id for item in received] == ["2", "3", "4", "5", "cut", "8"]
    assert [item.text for item in received if item.message_id != "cut"] == [
        "What does this value mean?",
        "What do you mean?",
        "Photo caption",
        "/start",
        "Plain text",
    ]
    assert received[2].photo_file_id == "photo" and received[3].command == "start"
    assert updates[0].message.text is None and updates[0].message.caption is None
    assert updates[0].message.api_kwargs["rich_message"] == RICH
    assert updates[1].message.quote.api_kwargs["rich_texts"] == [CELL]
    assert updates[1].message.quote.text == "" and updates[1].message.quote.is_manual
    assert updates[2].message.quote.text == "Plain" and updates[2].message.quote.is_manual
    assert updates[5].api_kwargs["stopped_message_generation"] == STOP
    assert updates[5].effective_chat is None and updates[5].effective_message is None
    assert app.update_processor.max_concurrent_updates == 1
    app.add_handler(TypeHandler(Update, inspect_stop), group=1)
    await app.process_update(updates[5])
    assert stops == [STOP]
    source = messenger.get_callback_source_message(received[4])
    assert source is not None and source.message_id == "42" and source.text == ""
    assert updates[6].callback_query.message.reply_markup.to_dict() == KEYBOARD
    print(
        "Inbound: text, command, photo and callback dispatched. Raw rich fields and stop checked."
    )


def mock_transport(calls: list[tuple[str, dict[str, str]]]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "tg1.invalid"
        method = request.url.path.rsplit("/", 1)[-1]
        data = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
        calls.append((method, data))
        if method == "editMessageText":
            result = (
                message(42, rich_message=RICH, reply_markup=KEYBOARD)
                if "rich_message" in data
                else message(42, text=data["text"])
            )
        else:
            result = {
                "getMe": BOT,
                "getUpdates": wire_updates(),
                "answerCallbackQuery": True,
                "sendMessageDraft": True,
                "sendRichMessageDraft": True,
                "sendRichMessage": message(42, rich_message=RICH, reply_markup=KEYBOARD),
                "sendMessage": message(42, text="Existing delivery"),
            }[method]
        return httpx.Response(200, json={"ok": True, "result": result})

    return httpx.MockTransport(respond)


async def check_outbound(
    app: Application, messenger: TelegramMessenger, calls: list[tuple[str, dict[str, str]]]
) -> None:
    chat_id = CHAT["id"]
    bot = app.bot
    assert await bot.send_message_draft(chat_id, draft_id=7, text="Plain preview") is True
    assert (
        await bot.send_message_draft(
            chat_id,
            draft_id=7,
            text="Stoppable preview",
            api_kwargs={"can_stop": True, "keep_on_stop": True},
        )
        is True
    )
    draft = {"chat_id": chat_id, "draft_id": 7, "message_thread_id": 3, "rich_message": MARKDOWN}
    assert await bot.do_api_request("sendRichMessageDraft", api_kwargs=draft) is True
    stoppable = {**draft, "can_stop": True, "keep_on_stop": True}
    assert await bot.do_api_request("sendRichMessageDraft", api_kwargs=stoppable) is True
    reply = {"chat_id": chat_id, "rich_message": MARKDOWN, "reply_markup": KEYBOARD}
    published = await bot.do_api_request("sendRichMessage", api_kwargs=reply, return_type=Message)
    assert published.message_id == 42 and published.api_kwargs["rich_message"] == RICH
    assert published.reply_markup.to_dict() == KEYBOARD
    edited = await bot.do_api_request(
        "editMessageText", api_kwargs={**reply, "message_id": 42}, return_type=Message
    )
    assert edited.message_id == 42 and edited.api_kwargs["rich_message"] == RICH

    sent = await messenger.send_message(
        OutgoingMessage(
            "Existing delivery", str(chat_id), reply_to="8", keyboard=[[("Cut", "cut:42")]]
        )
    )
    assert sent.success and sent.message_id == "42"
    assert (await messenger.edit_message(str(chat_id), "42", "Existing edit")).success

    expected = [
        (
            "getUpdates",
            {"allowed_updates": ["message", "callback_query", "stopped_message_generation"]},
        ),
        ("sendMessageDraft", {"chat_id": chat_id, "draft_id": 7, "text": "Plain preview"}),
        (
            "sendMessageDraft",
            {
                "chat_id": chat_id,
                "draft_id": 7,
                "text": "Stoppable preview",
                "can_stop": True,
                "keep_on_stop": True,
            },
        ),
        ("sendRichMessageDraft", draft),
        ("sendRichMessageDraft", stoppable),
        ("sendRichMessage", reply),
        ("editMessageText", {**reply, "message_id": 42}),
        (
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "Existing delivery",
                "reply_markup": KEYBOARD,
                "reply_parameters": {"message_id": 8},
            },
        ),
        ("editMessageText", {"chat_id": chat_id, "text": "Existing edit", "message_id": 42}),
    ]
    operation_calls = [
        (name, fields) for name, fields in calls if name not in {"getMe", "answerCallbackQuery"}
    ]
    assert len(operation_calls) == len(expected)
    for (actual_name, actual), (name, fields) in zip(operation_calls, expected, strict=True):
        assert actual_name == name and actual.keys() == fields.keys(), (actual_name, actual)
        for key, value in fields.items():
            decoded = actual[key] if isinstance(value, str) else json.loads(actual[key])
            assert decoded == value, (name, key, decoded)
    print("Outbound: plain and rich drafts, publication, edits and explicit update list checked.")


async def main() -> None:
    if not __debug__:
        raise RuntimeError("Run the audit without Python optimization to enable assertions")
    print(
        f"Python {platform.python_version()}, PTB {telegram.__version__}, "
        f"SDK Bot API {telegram.__bot_api_version__}"
    )
    check_sdk_surface()
    calls: list[tuple[str, dict[str, str]]] = []
    transport = mock_transport(calls)
    # Every request uses the fake transport, including getUpdates. The fake host is reserved.
    token = f"999:{secrets.token_urlsafe(16)}"
    app = (
        Application.builder()
        .token(token)
        .base_url("https://tg1.invalid/bot")
        .request(HTTPXRequest(httpx_kwargs={"transport": transport}))
        .get_updates_request(HTTPXRequest(httpx_kwargs={"transport": transport}))
        .build()
    )
    messenger = TelegramMessenger(token)
    received: list[IncomingMessage] = []

    async def record(incoming: IncomingMessage) -> None:
        received.append(incoming)

    messenger.register_message_handler(record)
    messenger.register_photo_handler(record)
    messenger.register_callback_handler(record)
    messenger.register_command_handler("start", record)
    with (
        patch("mai_gram.messenger.telegram.build_application", return_value=app),
        patch.object(Updater, "start_polling"),
    ):
        await messenger.start()
    try:
        await check_inbound(app, messenger, received)
        await check_outbound(app, messenger, calls)
    finally:
        await messenger.stop()


if __name__ == "__main__":
    asyncio.run(main())
