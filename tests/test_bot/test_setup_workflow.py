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
    messenger.delete_callback_source_message = AsyncMock(return_value=True)
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
        workflow_settings.get_model_title.return_value = None
        workflow_settings.get_model_id.side_effect = lambda key: key
        workflow_settings.get_available_prompts.return_value = {
            "default": "Default prompt",
            "support": "Support prompt",
        }
        workflow_settings.get_prompt_config.return_value = None
        workflow_settings.get_available_templates.return_value = ["empty"]
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


class TestModelDisplayLabels:
    async def test_model_selection_uses_custom_titles(self, session: AsyncSession) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = [
            "google/gemini-2.5-flash",
            "flash-creative",
        ]
        settings.get_default_model.return_value = "google/gemini-2.5-flash"
        settings.get_model_title.side_effect = lambda key: {
            "google/gemini-2.5-flash": "Gemini 2.5 Flash",
            "flash-creative": "Gemini Flash (creative)",
        }.get(key)
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await workflow.handle_start(message)

        build_kb = cast("MagicMock", workflow._messenger.build_inline_keyboard)
        build_kb.assert_called_once()
        rows = build_kb.call_args.args[0]
        labels = [row[0][0] for row in rows]
        assert labels == ["Gemini 2.5 Flash [default]", "Gemini Flash (creative)"]
        callbacks = [row[0][1] for row in rows]
        assert callbacks == ["model:google/gemini-2.5-flash", "model:flash-creative"]

    async def test_model_selection_falls_back_to_last_path_segment(
        self, session: AsyncSession
    ) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["vendor/some-model"]
        settings.get_default_model.return_value = "vendor/some-model"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await workflow.handle_start(message)

        build_kb = cast("MagicMock", workflow._messenger.build_inline_keyboard)
        rows = build_kb.call_args.args[0]
        assert rows[0][0][0] == "some-model [default]"

    async def test_model_selection_uses_key_as_label_without_slash(
        self, session: AsyncSession
    ) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["my-alias"]
        settings.get_default_model.return_value = ""
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await workflow.handle_start(message)

        build_kb = cast("MagicMock", workflow._messenger.build_inline_keyboard)
        rows = build_kb.call_args.args[0]
        assert rows[0][0][0] == "my-alias"


class TestSetupCallbacks:
    async def test_rejects_disallowed_model(self) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
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
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
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
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = PromptConfig(
            show_reasoning=False,
            show_tool_calls=False,
            send_datetime=False,
        )
        settings.get_available_templates.return_value = ["empty"]
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


class TestModelChange:
    """In-place model switching for an already-started chat (/model command)."""

    async def test_show_model_change_lists_models_with_cancel(self) -> None:
        workflow = _make_workflow()
        message = _make_message(chat_id="tg-chat", message_type=MessageType.COMMAND)

        await workflow.show_model_change(message, "openrouter/free")

        build_kb = cast("MagicMock", workflow._messenger.build_inline_keyboard)
        build_kb.assert_called_once()
        rows = build_kb.call_args.args[0]
        callbacks = [row[0][1] for row in rows]
        assert callbacks == [
            "setmodel:openrouter/free",
            "setmodel:openai/test-model",
            "cancel_action",
        ]
        # The currently active model is marked, and the cancel row is last.
        assert rows[0][0][0].startswith("✅")
        assert rows[-1][0][0] == "Cancel"

        sent_msg = _last_sent_message(workflow)
        assert "Current model: openrouter/free" in sent_msg.text
        assert "Choose a new model:" in sent_msg.text

    async def test_handle_model_change_updates_model_and_keeps_history(
        self, session: AsyncSession
    ) -> None:
        from mai_gram.db.models import Message

        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="openrouter/free",
            system_prompt="Existing prompt",
        )
        session.add(chat)
        session.add(Message(role="user", content="Remember this", chat_id="test-user@test-bot"))
        await session.commit()

        workflow = _make_workflow()
        message = _make_message(
            chat_id="tg-chat",
            bot_id="test-bot",
            callback_data="setmodel:openai/test-model",
            message_type=MessageType.CALLBACK,
        )

        with patch("mai_gram.bot.model_picker.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await workflow.handle_model_change(message, "openai/test-model")

        stored_chat = (
            await session.execute(select(Chat).where(Chat.id == "test-user@test-bot"))
        ).scalar_one()
        assert stored_chat.llm_model == "openai/test-model"
        assert stored_chat.system_prompt == "Existing prompt"

        from mai_gram.db.models import Message as MessageModel

        surviving = (
            (
                await session.execute(
                    select(MessageModel).where(MessageModel.chat_id == "test-user@test-bot")
                )
            )
            .scalars()
            .all()
        )
        assert [m.content for m in surviving] == ["Remember this"]

        cast("AsyncMock", workflow._messenger.delete_callback_source_message).assert_awaited_once()
        sent_msg = _last_sent_message(workflow)
        assert "Model changed: openrouter/free → openai/test-model" in sent_msg.text

    async def test_handle_model_change_rejects_disallowed_model(self) -> None:
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        message = _make_message(
            chat_id="tg-chat",
            bot_id="test-bot",
            callback_data="setmodel:not-allowed",
            message_type=MessageType.CALLBACK,
        )

        await workflow.handle_model_change(message, "not-allowed")

        sent_msg = _last_sent_message(workflow)
        assert "not available for this bot" in sent_msg.text
        cast("AsyncMock", workflow._messenger.delete_callback_source_message).assert_not_awaited()

    async def test_handle_model_change_reports_missing_chat(self, session: AsyncSession) -> None:
        workflow = _make_workflow()
        message = _make_message(
            chat_id="tg-chat",
            bot_id="test-bot",
            callback_data="setmodel:openai/test-model",
            message_type=MessageType.CALLBACK,
        )

        with patch("mai_gram.bot.model_picker.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            await workflow.handle_model_change(message, "openai/test-model")

        sent_msg = _last_sent_message(workflow)
        assert sent_msg.text == "No chat exists yet. Use /start to create one."


class TestTemplateGroupSelection:
    async def test_prompt_selection_shows_group_chooser(self) -> None:
        """After prompt selection, template groups should be shown (not flat list)."""
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = [
            "empty",
            "xml",
            "xml_prefill",
            "json",
            "json_prefill",
        ]
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
                bot_id="test-bot",
                callback_data="prompt:default",
                message_type=MessageType.CALLBACK,
            )
        )

        session = workflow.get_setup_session("test-user")
        assert session is not None
        assert session.state == SetupState.CHOOSING_TEMPLATE_GROUP

        sent_msg = _last_sent_message(workflow)
        assert "category" in sent_msg.text.lower()

    async def test_group_selection_shows_variants(self) -> None:
        """Selecting a group with multiple templates shows the variant list."""
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = [
            "empty",
            "xml",
            "xml_prefill",
            "json",
            "json_prefill",
        ]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_TEMPLATE_GROUP,
            selected_model="openrouter/free",
            selected_prompt_text="Default prompt",
        )

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                bot_id="test-bot",
                callback_data="tpl_group:xml",
                message_type=MessageType.CALLBACK,
            )
        )

        session = workflow.get_setup_session("test-user")
        assert session is not None
        assert session.state == SetupState.CHOOSING_TEMPLATE

        sent_msg = _last_sent_message(workflow)
        assert "variant" in sent_msg.text.lower()

    async def test_single_template_in_group_skips_variant_step(self) -> None:
        """If a group has only one available template, skip variant selection and go to params."""
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty", "json"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_TEMPLATE_GROUP,
            selected_model="openrouter/free",
            selected_prompt_text="Default prompt",
        )

        await workflow.handle_setup_callback(
            _make_message(
                chat_id="tg-chat",
                bot_id="test-bot",
                callback_data="tpl_group:json",
                message_type=MessageType.CALLBACK,
            )
        )

        session = workflow.get_setup_session("test-user")
        assert session is not None
        assert session.state == SetupState.CONFIGURING_TEMPLATE_PARAMS
        assert session.selected_template == "json"

        sent_msg = _last_sent_message(workflow)
        assert "configurable parameters" in sent_msg.text.lower()

    async def test_ungrouped_template_selection_finishes_setup(self, session: AsyncSession) -> None:
        """Selecting an ungrouped template (empty) should finish setup directly."""
        settings = MagicMock()
        settings.get_allowed_models.return_value = ["openrouter/free"]
        settings.get_default_model.return_value = "openrouter/free"
        settings.get_model_title.return_value = None
        settings.get_model_id.side_effect = lambda key: key
        settings.get_available_prompts.return_value = {"default": "Default prompt"}
        settings.get_prompt_config.return_value = None
        settings.get_available_templates.return_value = ["empty", "xml", "json"]
        settings.default_timezone = "UTC"

        workflow = _make_workflow(settings=settings)
        workflow._sessions["test-user"] = SetupSession(
            user_id="test-user",
            chat_id="tg-chat",
            state=SetupState.CHOOSING_TEMPLATE_GROUP,
            selected_model="openrouter/free",
            selected_prompt_text="Default prompt",
        )

        with patch("mai_gram.bot.setup_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_setup_callback(
                _make_message(
                    chat_id="tg-chat",
                    bot_id="test-bot",
                    callback_data="tpl_group:__single__:empty",
                    message_type=MessageType.CALLBACK,
                )
            )

        assert not workflow.is_in_setup("test-user")
        sent_msg = _last_sent_message(workflow)
        assert "Chat created!" in sent_msg.text
