"""Tests for the setup workflow extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.setup_workflow import SetupSession, SetupState, SetupWorkflow
from mai_gram.config import BotConfig, PromptConfig
from mai_gram.db.models import Chat
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _fake_secret(label: str) -> str:
    return f"test-{label}-value"


def _make_workflow(
    *,
    settings: MagicMock | None = None,
    bot_config: BotConfig | None = None,
) -> SetupWorkflow:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.build_inline_keyboard.return_value = {
        "inline_keyboard": [[{"text": "One", "callback_data": "model:test"}]]
    }

    workflow_settings = settings if settings is not None else MagicMock()
    if settings is None:
        workflow_settings.get_allowed_models.return_value = [
            "openrouter/free",
            "openai/test-model",
        ]
        workflow_settings.get_default_model.return_value = "openrouter/free"
        workflow_settings.get_available_prompts.return_value = {
            "default": "Default prompt",
            "support": "Support prompt",
        }
        workflow_settings.get_prompt_config.return_value = None
        workflow_settings.default_timezone = "UTC"

    return SetupWorkflow(
        messenger,
        workflow_settings,
        bot_config=bot_config,
        resolve_chat_id=lambda message: (
            f"{message.user_id}@{message.bot_id}" if message.bot_id else message.chat_id
        ),
    )


def _make_message(
    *,
    user_id: str = "test-user",
    chat_id: str = "test-chat",
    text: str = "",
    callback_data: str | None = None,
    bot_id: str = "",
    message_type: MessageType = MessageType.COMMAND,
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        message_id="msg-1",
        message_type=message_type,
        text=text,
        callback_data=callback_data,
        bot_id=bot_id,
    )


def _last_sent_message(workflow: SetupWorkflow) -> OutgoingMessage:
    send_message = cast("AsyncMock", workflow._messenger.send_message)
    await_args = send_message.await_args
    assert await_args is not None
    sent_msg = await_args.args[0]
    assert isinstance(sent_msg, OutgoingMessage)
    return sent_msg


class TestHandleStart:
    async def test_existing_chat_reports_reset_hint(self, session: AsyncSession) -> None:
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="openrouter/free",
            system_prompt="Prompt",
        )
        session.add(chat)
        await session.commit()

        workflow = _make_workflow()
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_start(message)

        sent_msg = _last_sent_message(workflow)
        assert "Chat already configured" in sent_msg.text
        assert "Use /reset to start over." in sent_msg.text

    async def test_missing_chat_starts_setup_and_shows_models(self, session: AsyncSession) -> None:
        workflow = _make_workflow()
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_start(message)

        assert workflow.is_in_setup(message.user_id)
        setup_session = workflow.get_setup_session(message.user_id)
        assert setup_session is not None
        assert setup_session.state == SetupState.CHOOSING_MODEL

        sent_msg = _last_sent_message(workflow)
        assert sent_msg.text == "Choose an LLM model:"
        assert sent_msg.keyboard == workflow._messenger.build_inline_keyboard.return_value
        workflow._messenger.build_inline_keyboard.assert_called_once()


class TestSetupCallbacks:
    async def test_rejects_disallowed_model(self) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(user_id="test-user", chat_id="tg-chat")

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                callback_data="model:not-allowed",
                message_type=MessageType.CALLBACK,
            )
        )

        sent_msg = _last_sent_message(workflow)
        assert "not available for this bot" in sent_msg.text
        setup_session = workflow.get_setup_session("test-user")
        assert setup_session is not None
        assert setup_session.selected_model == ""

    async def test_model_selection_advances_to_prompt_selection(self) -> None:
        workflow = _make_workflow()
        workflow._sessions["test-user"] = SetupSession(user_id="test-user", chat_id="tg-chat")

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                callback_data="model:openrouter/free",
                message_type=MessageType.CALLBACK,
            )
        )

        session = workflow.get_setup_session("test-user")
        assert session is not None
        assert session.selected_model == "openrouter/free"
        assert session.state == SetupState.CHOOSING_PROMPT

        sent_msg = _last_sent_message(workflow)
        assert "Now choose a system prompt:" in sent_msg.text

    async def test_custom_prompt_is_blocked_when_bot_restricts_prompts(self) -> None:
        workflow = _make_workflow(
            bot_config=BotConfig(
                token=_fake_secret("bot-token"),
                allowed_prompts=["default"],
            )
        )
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_PROMPT,
            selected_model="openrouter/free",
        )

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                callback_data="prompt:__custom__",
                message_type=MessageType.CALLBACK,
            )
        )

        sent_msg = _last_sent_message(workflow)
        assert sent_msg.text == "Custom prompts are not available for this bot."

    async def test_custom_prompt_requests_text_input(self) -> None:
        workflow = _make_workflow()
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_PROMPT,
            selected_model="openrouter/free",
        )

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                callback_data="prompt:__custom__",
                message_type=MessageType.CALLBACK,
            )
        )

        session = workflow.get_setup_session("test-user")
        assert session is not None
        assert session.state == SetupState.AWAITING_CUSTOM_PROMPT

        sent_msg = _last_sent_message(workflow)
        assert sent_msg.text == "Type your custom system prompt:"

    async def test_missing_prompt_reports_error(self) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_PROMPT,
            selected_model="openrouter/free",
        )

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                callback_data="prompt:missing",
                message_type=MessageType.CALLBACK,
            )
        )

        sent_msg = _last_sent_message(workflow)
        assert "Prompt 'missing' not found" in sent_msg.text


class TestFinishSetup:
    async def test_named_prompt_creates_chat_with_prompt_config(
        self, session: AsyncSession
    ) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = PromptConfig(
            show_reasoning=False,
            show_tool_calls=False,
            send_datetime=False,
        )
        settings.default_timezone = "Europe/Moscow"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_PROMPT,
            selected_model="openrouter/free",
        )

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_setup_callback(
                _make_message(
                    chat_id="tg-chat",
                    bot_id="test-bot",
                    callback_data="prompt:default",
                    message_type=MessageType.CALLBACK,
                )
            )

        stored_chat = (
            await session.execute(select(Chat).where(Chat.id == "test-user@test-bot"))
        ).scalar_one()
        assert stored_chat.prompt_name == "default"
        assert stored_chat.system_prompt == "Default prompt"
        assert stored_chat.timezone == "Europe/Moscow"
        assert stored_chat.show_reasoning is False
        assert stored_chat.show_tool_calls is False
        assert stored_chat.send_datetime is False
        assert not workflow.is_in_setup("test-user")

        sent_msg = _last_sent_message(workflow)
        assert "Chat created!" in sent_msg.text
        assert "Reasoning: OFF | Tool calls: OFF | Datetime: OFF" in sent_msg.text

    async def test_custom_prompt_text_creates_chat(self, session: AsyncSession) -> None:
        workflow = _make_workflow()
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.AWAITING_CUSTOM_PROMPT,
            selected_model="openrouter/free",
        )

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_setup_text(
                _make_message(
                    chat_id="tg-chat",
                    bot_id="test-bot",
                    text="You are a careful assistant.",
                    message_type=MessageType.TEXT,
                )
            )

        stored_chat = (
            await session.execute(select(Chat).where(Chat.id == "test-user@test-bot"))
        ).scalar_one()
        assert stored_chat.prompt_name is None
        assert stored_chat.system_prompt == "You are a careful assistant."
        assert stored_chat.send_datetime is True
        assert not workflow.is_in_setup("test-user")
