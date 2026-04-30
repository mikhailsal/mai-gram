"""Tests for BotHandler coordinator behavior."""

from __future__ import annotations

import zoneinfo
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mai_gram.bot import handler as handler_module
from mai_gram.bot.handler import BotHandler, make_chat_id
from mai_gram.bot.handler_services import HandlerServices
from mai_gram.messenger.base import IncomingMessage, MessageType


def _make_message(
    *,
    user_id: str = "user-1",
    chat_id: str = "chat-1",
    bot_id: str = "",
    text: str = "hello",
    command: str | None = None,
    command_args: str | None = None,
    callback_data: str | None = None,
    message_type: MessageType = MessageType.TEXT,
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        bot_id=bot_id,
        message_id="msg-1",
        message_type=message_type,
        text=text,
        command=command,
        command_args=command_args,
        callback_data=callback_data,
    )


def _make_services() -> HandlerServices:
    return HandlerServices(
        response_renderer=MagicMock(),
        mcp_manager_factory=MagicMock(),
        assistant_turn_builder=MagicMock(),
        conversation_executor=MagicMock(),
        conversation_service=MagicMock(handle_message=AsyncMock(return_value=["resp-1"])),
        import_workflow=MagicMock(
            handle_import=AsyncMock(),
            handle_document=AsyncMock(),
        ),
        history_actions=MagicMock(
            get_message_preview=AsyncMock(return_value="preview"),
            handle_cut_above=AsyncMock(),
        ),
        regenerate_service=MagicMock(handle_regenerate=AsyncMock(return_value=["regen-1"])),
        resend_service=MagicMock(
            handle_resend=AsyncMock(
                return_value=SimpleNamespace(
                    replaced_previous=False,
                    sent_message_ids=["resent-1"],
                )
            )
        ),
        reset_workflow=MagicMock(handle_reset=AsyncMock()),
        setup_workflow=MagicMock(
            is_in_setup=MagicMock(return_value=False),
            get_setup_session=MagicMock(return_value=None),
            clear_setup_session=MagicMock(),
            handle_start=AsyncMock(),
            handle_setup_text=AsyncMock(),
        ),
        callback_router=MagicMock(handle_callback=AsyncMock()),
    )


def _make_handler(monkeypatch, *, bot_allowed_users: list[object] | None = None):
    messenger = MagicMock()
    messenger.register_command_handler = MagicMock()
    messenger.register_message_handler = MagicMock()
    messenger.register_callback_handler = MagicMock()
    messenger.register_document_handler = MagicMock()
    messenger.send_message = AsyncMock()
    messenger.build_inline_keyboard = MagicMock(return_value="keyboard")

    services = _make_services()
    access_control = MagicMock()
    access_control.check_access = AsyncMock(return_value=True)
    access_control.handle_rate_limited = AsyncMock()
    rate_limiter = MagicMock()
    rate_limiter.check_rate_limit = AsyncMock(return_value=True)
    allowed_users_seen: list[set[str]] = []

    settings = SimpleNamespace(
        memory_data_dir="./data",
        wiki_context_limit=20,
        short_term_limit=500,
        tool_max_iterations=5,
        get_allowed_user_ids=lambda: {"global-user"},
    )
    bot_config = (
        SimpleNamespace(allowed_users=bot_allowed_users) if bot_allowed_users is not None else None
    )

    monkeypatch.setattr(handler_module, "get_settings", lambda: settings)
    monkeypatch.setattr(handler_module, "build_handler_services", lambda *args, **kwargs: services)

    def fake_access_control(current_messenger, *, allowed_users):
        assert current_messenger is messenger
        allowed_users_seen.append(allowed_users)
        return access_control

    def fake_rate_limiter(rate_limit_config, *, on_rate_limited):
        del rate_limit_config
        assert on_rate_limited is access_control.handle_rate_limited
        return rate_limiter

    monkeypatch.setattr(handler_module, "AccessControl", fake_access_control)
    monkeypatch.setattr(handler_module, "RateLimiter", fake_rate_limiter)

    handler = BotHandler(messenger, MagicMock(), test_mode=True, bot_config=bot_config)
    return handler, messenger, services, access_control, rate_limiter, allowed_users_seen


@asynccontextmanager
async def _session_context(session):
    yield session


def _patch_session(monkeypatch, session) -> None:
    monkeypatch.setattr(handler_module, "get_session", lambda: _session_context(session))


def test_make_chat_id_combines_user_and_bot() -> None:
    assert make_chat_id("user-1", "bot-1") == "user-1@bot-1"


def test_init_registers_transport_handlers_and_bot_whitelist(monkeypatch) -> None:
    handler, messenger, _, _, _, allowed_users_seen = _make_handler(
        monkeypatch,
        bot_allowed_users=[123, "456"],
    )

    registered_commands = [
        call.args[0] for call in messenger.register_command_handler.call_args_list
    ]

    assert allowed_users_seen == [{"123", "456"}]
    assert registered_commands == [
        "start",
        "reset",
        "model",
        "help",
        "datetime",
        "timezone",
        "reasoning",
        "toolcalls",
        "import",
        "resend_last",
    ]
    messenger.register_message_handler.assert_called_once_with(handler._handle_message)
    messenger.register_callback_handler.assert_called_once_with(handler._handle_callback)
    messenger.register_document_handler.assert_called_once_with(handler._handle_document)


async def test_handle_start_reset_import_callback_and_document_delegate(monkeypatch) -> None:
    handler, _, services, _, _, _ = _make_handler(monkeypatch)
    services.setup_workflow.is_in_setup.return_value = True

    await handler._handle_start(_make_message(message_type=MessageType.COMMAND, command="start"))
    await handler._handle_reset(_make_message(message_type=MessageType.COMMAND, command="reset"))
    await handler._handle_import(_make_message(message_type=MessageType.COMMAND, command="import"))
    await handler._handle_callback(
        _make_message(message_type=MessageType.CALLBACK, callback_data="confirm")
    )
    await handler._handle_document(_make_message(message_type=MessageType.DOCUMENT))

    services.setup_workflow.handle_start.assert_awaited_once()
    services.reset_workflow.handle_reset.assert_awaited_once()
    services.import_workflow.handle_import.assert_awaited_once()
    assert services.import_workflow.handle_import.await_args.kwargs == {"in_setup": True}
    services.callback_router.handle_callback.assert_awaited_once()
    services.import_workflow.handle_document.assert_awaited_once()


async def test_handle_message_stops_when_access_is_denied(monkeypatch) -> None:
    handler, _, services, access_control, rate_limiter, _ = _make_handler(monkeypatch)
    access_control.check_access.return_value = False

    await handler._handle_message(_make_message())

    rate_limiter.check_rate_limit.assert_not_called()
    services.setup_workflow.handle_setup_text.assert_not_called()
    services.conversation_service.handle_message.assert_not_called()


async def test_handle_message_stops_when_rate_limited(monkeypatch) -> None:
    handler, _, services, _, rate_limiter, _ = _make_handler(monkeypatch)
    rate_limiter.check_rate_limit.return_value = False

    await handler._handle_message(_make_message())

    services.setup_workflow.handle_setup_text.assert_not_called()
    services.conversation_service.handle_message.assert_not_called()


async def test_handle_message_routes_setup_text_before_conversation(monkeypatch) -> None:
    handler, _, services, _, _, _ = _make_handler(monkeypatch)
    services.setup_workflow.is_in_setup.return_value = True

    await handler._handle_message(_make_message())

    services.setup_workflow.handle_setup_text.assert_awaited_once()
    services.conversation_service.handle_message.assert_not_called()


async def test_handle_message_routes_to_conversation_and_tracks_responses(monkeypatch) -> None:
    handler, _, services, _, _, _ = _make_handler(monkeypatch)
    services.conversation_service.handle_message.return_value = ["resp-1", "resp-2"]

    await handler._handle_message(_make_message(chat_id="chat-42"))

    services.conversation_service.handle_message.assert_awaited_once()
    assert handler._response_message_ids["chat-42"] == ["resp-1", "resp-2"]


async def test_handle_model_reports_missing_chat_and_current_model(monkeypatch) -> None:
    handler, messenger, _, _, _, _ = _make_handler(monkeypatch)
    session = MagicMock(commit=AsyncMock())
    _patch_session(monkeypatch, session)
    handler._get_chat = AsyncMock(side_effect=[None, SimpleNamespace(llm_model="openrouter/free")])
    message = _make_message(message_type=MessageType.COMMAND, command="model")

    await handler._handle_model(message)
    await handler._handle_model(message)

    sent_messages = [call.args[0].text for call in messenger.send_message.await_args_list]
    assert sent_messages[0] == "No chat exists yet. Use /start to create one."
    assert sent_messages[1].startswith("Current model: openrouter/free")


async def test_handle_help_sends_help_text(monkeypatch) -> None:
    handler, messenger, _, _, _, _ = _make_handler(monkeypatch)

    await handler._handle_help(_make_message(message_type=MessageType.COMMAND, command="help"))

    sent_text = messenger.send_message.await_args.args[0].text
    assert "/start - Set up a new chat" in sent_text
    assert "/resend_last - Re-send last AI message" in sent_text


async def test_handle_datetime_toggle_updates_chat_flag(monkeypatch) -> None:
    handler, messenger, _, _, _, _ = _make_handler(monkeypatch)
    session = MagicMock(commit=AsyncMock())
    chat = SimpleNamespace(send_datetime=False)
    _patch_session(monkeypatch, session)
    handler._get_chat = AsyncMock(return_value=chat)

    await handler._handle_datetime_toggle(
        _make_message(message_type=MessageType.COMMAND, command="datetime")
    )

    assert chat.send_datetime is True
    session.commit.assert_awaited_once_with()
    assert messenger.send_message.await_args.args[0].text == "Date/time in messages: ON"


async def test_handle_timezone_reports_current_invalid_and_valid_values(monkeypatch) -> None:
    handler, messenger, _, _, _, _ = _make_handler(monkeypatch)
    session = MagicMock(commit=AsyncMock())
    chat = SimpleNamespace(timezone="UTC")
    _patch_session(monkeypatch, session)
    handler._get_chat = AsyncMock(return_value=chat)
    monkeypatch.setattr(zoneinfo, "available_timezones", lambda: {"UTC", "Europe/Moscow"})

    await handler._handle_timezone(
        _make_message(message_type=MessageType.COMMAND, command="timezone", command_args=None)
    )
    await handler._handle_timezone(
        _make_message(
            message_type=MessageType.COMMAND,
            command="timezone",
            command_args="Mars/Phobos",
        )
    )
    await handler._handle_timezone(
        _make_message(
            message_type=MessageType.COMMAND,
            command="timezone",
            command_args="Europe/Moscow",
        )
    )

    sent_messages = [call.args[0].text for call in messenger.send_message.await_args_list]
    assert sent_messages[0].startswith("Current timezone: UTC")
    assert sent_messages[1].startswith("Unknown timezone: Mars/Phobos")
    assert sent_messages[2] == "Timezone set to: Europe/Moscow"
    assert chat.timezone == "Europe/Moscow"
    session.commit.assert_awaited_once_with()


async def test_resend_regenerate_confirmation_cut_and_preview_helpers(monkeypatch) -> None:
    handler, messenger, services, _, _, _ = _make_handler(monkeypatch)
    services.resend_service.handle_resend.return_value = SimpleNamespace(
        replaced_previous=True,
        sent_message_ids=["resent-2"],
    )
    handler._cut_original_html["chat-1:orig-1"] = ("<b>cached</b>", "html")

    await handler._handle_resend_last(
        _make_message(message_type=MessageType.COMMAND, command="resend_last")
    )
    await handler._handle_regenerate(_make_message(chat_id="chat-1"))
    await handler._show_confirmation(
        _make_message(chat_id="chat-1"),
        "Confirm?",
        confirm_data="yes",
        cancel_data="no",
    )
    await handler._handle_cut_above(
        _make_message(chat_id="chat-1"),
        7,
        original_tg_msg_id="orig-1",
    )
    preview = await handler._get_message_preview(7, max_len=20)

    assert handler._response_message_ids["chat-1"] == ["regen-1"]
    assert messenger.build_inline_keyboard.call_args.args[0] == [[("Yes", "yes"), ("Cancel", "no")]]
    services.history_actions.handle_cut_above.assert_awaited_once_with(
        _make_message(chat_id="chat-1"),
        7,
        original_tg_msg_id="orig-1",
        cached_original=("<b>cached</b>", "html"),
    )
    assert preview == "preview"


async def test_get_chat_and_chat_id_for_bot_scoped_messages(monkeypatch) -> None:
    handler, _, _, _, _, _ = _make_handler(monkeypatch)
    result = MagicMock(scalar_one_or_none=MagicMock(return_value=SimpleNamespace(id="chat-1")))
    session = MagicMock(execute=AsyncMock(return_value=result))
    message = _make_message(bot_id="bot-9")

    chat = await handler._get_chat(session, "chat-1")

    assert handler._chat_id_for(message) == "user-1@bot-9"
    assert chat.id == "chat-1"
