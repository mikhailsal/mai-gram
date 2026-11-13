"""Tests for guest_mode bot configuration and behavior."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.bot import setup_workflow as setup_workflow_module
from mai_gram.bot.conversation_executor import ConversationExecutor, StreamOutcome
from mai_gram.bot.handler import BotHandler
from mai_gram.bot.setup_workflow import SetupWorkflow
from mai_gram.config import Settings
from mai_gram.config_loaders import BotConfig, BotsConfigLoader, PromptConfig
from mai_gram.llm.provider import TokenUsage
from mai_gram.messenger.base import IncomingMessage, MessageType


def _make_message(
    *,
    user_id: str = "user-1",
    chat_id: str = "chat-1",
    bot_id: str = "guest-bot",
    text: str = "/start",
    command: str | None = "start",
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        bot_id=bot_id,
        message_id="msg-1",
        message_type=MessageType.COMMAND,
        text=text,
        command=command,
    )


def test_bot_config_guest_mode_defaults_and_overrides(tmp_path) -> None:
    cfg_file = tmp_path / "bots.toml"
    cfg_file.write_text(
        """
[[bots]]
token = "tok-guest"
guest_mode = true
default_model = "google/gemini-3-flash-preview"
default_prompt = "independent-short"
default_template = "empty"
welcome_message = "Hi friend!"

[[bots]]
token = "tok-normal"
""",
        encoding="utf-8",
    )
    configs = BotsConfigLoader(str(cfg_file)).get_bot_configs()
    assert len(configs) == 2

    guest = configs[0]
    assert guest.guest_mode is True
    assert guest.default_model == "google/gemini-3-flash-preview"
    assert guest.default_prompt == "independent-short"
    assert guest.default_template == "empty"
    assert guest.welcome_message == "Hi friend!"
    assert guest.buttons_enabled is False
    assert guest.usage_enabled is False
    assert guest.tool_calls_override is False

    normal = configs[1]
    assert normal.guest_mode is False
    assert normal.buttons_enabled is True
    assert normal.usage_enabled is True
    assert normal.tool_calls_override is None

    overridden = BotConfig(
        token="tok-over",  # noqa: S106
        guest_mode=True,
        show_buttons=True,
        show_usage=True,
        show_tool_calls=True,
    )
    assert overridden.buttons_enabled is True
    assert overridden.usage_enabled is True
    assert overridden.tool_calls_override is True


def test_handler_registers_only_start_and_reset_in_guest_mode(monkeypatch, tmp_path) -> None:
    from mai_gram.bot import handler as handler_module

    messenger = MagicMock()
    registered: list[str] = []
    messenger.register_command_handler.side_effect = lambda name, _h, description="": (
        registered.append(name)
    )
    settings = MagicMock()
    settings.get_allowed_user_ids.return_value = set()
    settings.get_format_repair_config.return_value = {"enabled": False}
    settings.memory_data_dir = str(tmp_path / "mem")
    settings.wiki_context_limit = 5
    settings.short_term_limit = 20
    settings.tool_max_iterations = 5
    settings.database_url = ""
    monkeypatch.setattr(handler_module, "get_settings", lambda: settings)

    BotHandler(
        messenger,
        MagicMock(),
        test_mode=True,
        bot_config=BotConfig(
            token="tok-guest",  # noqa: S106
            guest_mode=True,
            allowed_users=[],
        ),
    )

    assert registered == ["start", "reset"]


@pytest.mark.asyncio
async def test_guest_mode_start_auto_creates_chat_without_keyboards(monkeypatch) -> None:
    messenger = MagicMock()
    messenger.send_message = AsyncMock()
    settings = MagicMock(spec=Settings)
    settings.default_timezone = "UTC"
    settings.get_allowed_models.return_value = ["google/gemini-3-flash-preview"]
    settings.get_default_model.return_value = "google/gemini-3-flash-preview"
    settings.get_available_prompts.return_value = {"independent-short": "Be brief."}
    settings.get_prompt_config.return_value = PromptConfig(
        show_reasoning=False,
        show_tool_calls=True,
        send_datetime=True,
    )

    added_chats: list[object] = []
    existing_chat: object | None = None

    class _FakeSession:
        def add(self, obj: object) -> None:
            added_chats.append(obj)

        async def commit(self) -> None:
            pass

    @asynccontextmanager
    async def _fake_get_session():
        yield _FakeSession()

    monkeypatch.setattr(setup_workflow_module, "get_session", _fake_get_session)

    bot_cfg = BotConfig(
        token="tok-guest",  # noqa: S106
        guest_mode=True,
        default_model="google/gemini-3-flash-preview",
        default_prompt="independent-short",
        default_template="empty",
    )
    workflow = SetupWorkflow(
        messenger,
        settings=settings,
        bot_config=bot_cfg,
        resolve_chat_id=lambda m: f"{m.user_id}:{m.bot_id}",
    )

    async def _fake_get_chat(_session, _chat_id):
        return existing_chat

    monkeypatch.setattr(workflow, "_get_chat", _fake_get_chat)

    await workflow.handle_start(_make_message())

    assert len(added_chats) == 1
    created = added_chats[0]
    assert created.llm_model == "google/gemini-3-flash-preview"
    assert created.prompt_name == "independent-short"
    assert created.system_prompt == "Be brief."
    assert created.response_template is None
    assert created.show_tool_calls is False
    # Default system message in English without buttons
    assert messenger.send_message.await_count == 1
    sent = messenger.send_message.call_args.args[0]
    assert sent.text == "[System] Chat is ready. Send a message to start."
    assert sent.keyboard is None

    # Subsequent /start also sends the system ready message without model/reset details
    existing_chat = created
    await workflow.handle_start(_make_message())
    assert messenger.send_message.await_count == 2
    assert (
        messenger.send_message.call_args.args[0].text
        == "[System] Chat is ready. Send a message to start."
    )

    # When welcome_message IS configured, it is sent
    bot_cfg.welcome_message = "[System] Custom welcome."
    await workflow.handle_start(_make_message())
    assert messenger.send_message.await_count == 3
    assert messenger.send_message.call_args.args[0].text == "[System] Custom welcome."


@pytest.mark.asyncio
async def test_executor_hides_buttons_and_usage_when_disabled() -> None:
    messenger = MagicMock()
    renderer = MagicMock()
    renderer._format_usage_footer.return_value = "100/20 tokens | $0.0010"

    executor = ConversationExecutor(
        messenger,
        MagicMock(),
        tool_max_iterations=3,
        renderer=renderer,
        show_buttons=False,
        show_usage=False,
    )
    executor._finalize_fresh = AsyncMock()
    request = SimpleNamespace(telegram_chat_id="chat-1")
    outcome = StreamOutcome(
        response_text="Hello!",
        response_reasoning="",
        placeholder_msg_id=None,
        committed_content_offset=0,
        reasoning_committed=False,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        cost=0.001,
        is_byok=False,
    )

    await executor._finalize_response(request, [], outcome, saved_msg_id=42)

    executor._finalize_fresh.assert_awaited_once_with(request, [], outcome, "", None)
    renderer._format_usage_footer.assert_not_called()


@pytest.mark.asyncio
async def test_executor_keeps_regenerate_button_on_error_when_buttons_disabled() -> None:
    messenger = MagicMock()
    messenger.build_inline_keyboard.return_value = "error-kb"
    renderer = MagicMock()
    renderer._deliver_error = AsyncMock()

    executor = ConversationExecutor(
        messenger,
        MagicMock(),
        tool_max_iterations=3,
        renderer=renderer,
        show_buttons=False,
        show_usage=False,
    )
    request = SimpleNamespace(telegram_chat_id="chat-1")
    sent_ids: list[str] = []

    await executor._deliver_error(request, sent_ids, "AI provider error.")

    messenger.build_inline_keyboard.assert_called_once_with([[("🔄 Regenerate", "regen")]])
    renderer._deliver_error.assert_awaited_once_with(
        "chat-1",
        "AI provider error.",
        placeholder_msg_id=None,
        keyboard="error-kb",
        sent_msg_ids=sent_ids,
    )
