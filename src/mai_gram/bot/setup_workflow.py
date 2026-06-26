"""Setup workflow for chat creation and prompt/model selection."""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from mai_gram.bot import custom_model, model_picker, setup_finalizer
from mai_gram.bot.setup_templates import (
    get_available_templates_for_bot,
    parse_kv_params,
    show_template_group_selection,
    show_template_params_summary,
)
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import BotConfig, Settings
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


class SetupState(str, enum.Enum):
    """States in the setup flow."""

    CHOOSING_MODEL = "choosing_model"
    AWAITING_CUSTOM_MODEL = "awaiting_custom_model"
    CHOOSING_PROMPT = "choosing_prompt"
    AWAITING_CUSTOM_PROMPT = "awaiting_custom_prompt"
    CHOOSING_TEMPLATE_GROUP = "choosing_template_group"
    CHOOSING_TEMPLATE = "choosing_template"
    CONFIGURING_TEMPLATE_PARAMS = "configuring_template_params"


@dataclass
class SetupSession:
    """Tracks the state of an ongoing setup session."""

    user_id: str
    chat_id: str
    state: SetupState = SetupState.CHOOSING_MODEL
    selected_model: str = ""
    selected_prompt_name: str | None = None
    selected_prompt_text: str = ""
    selected_template: str | None = None
    template_params: dict[str, str] | None = None
    custom_model_params: dict[str, Any] | None = None
    bot_id: str = ""


class SetupWorkflow:
    """Own the /start onboarding flow and setup callbacks."""

    def __init__(
        self,
        messenger: Messenger,
        settings: Settings,
        *,
        bot_config: BotConfig | None,
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._settings = settings
        self._bot_config = bot_config
        self._resolve_chat_id = resolve_chat_id
        self._sessions: dict[str, SetupSession] = {}
        # user_id -> chat_id for an in-place /model custom switch awaiting text.
        self._pending_custom_model: dict[str, str] = {}

    def is_in_setup(self, user_id: str) -> bool:
        return user_id in self._sessions

    def _custom_model_allowed(self, user_id: str) -> bool:
        return custom_model.is_user_allowed(self._bot_config, user_id)

    def get_setup_session(self, user_id: str) -> SetupSession | None:
        return self._sessions.get(user_id)

    def clear_setup_session(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    async def handle_start(self, message: IncomingMessage) -> None:
        """Start the chat setup flow unless the chat already exists."""
        chat_id = self._resolve_chat_id(message)
        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
        if chat:
            await self._messenger.send_message(
                OutgoingMessage(
                    text=(
                        f"Chat already configured (model: {chat.llm_model}).\n"
                        "Use /reset to start over."
                    ),
                    chat_id=message.chat_id,
                )
            )
            return
        await self._start_setup(message.user_id, message.chat_id)

    async def handle_setup_callback(self, message: IncomingMessage) -> None:
        """Handle setup callback queries for model, prompt, and template selection."""
        session = self._sessions.get(message.user_id)
        if not session or not message.callback_data:
            return
        parts = message.callback_data.split(":", 1)
        if len(parts) != 2:
            return
        category, value = parts
        await self._route_callback(message, session, category, value)

    async def _route_callback(
        self,
        message: IncomingMessage,
        session: SetupSession,
        category: str,
        value: str,
    ) -> None:
        if category == "model" and session.state == SetupState.CHOOSING_MODEL:
            await self._handle_model_selection(session, value)
        elif category == "prompt" and session.state == SetupState.CHOOSING_PROMPT:
            await self._handle_prompt_selection(message, session, value)
        elif category == "tpl_group" and session.state == SetupState.CHOOSING_TEMPLATE_GROUP:
            await self._handle_template_group_selection(message, session, value)
        elif category == "template" and session.state == SetupState.CHOOSING_TEMPLATE:
            await self._handle_template_selection(message, session, value)
        elif category == "tpl_params" and session.state == SetupState.CONFIGURING_TEMPLATE_PARAMS:
            if value == "__defaults__":
                session.template_params = None
            session.bot_id = message.bot_id or session.bot_id
            await self._finish_setup_from_session(session, message=message)

    async def handle_setup_text(self, message: IncomingMessage) -> None:
        """Handle free-form text entered during the setup flow."""
        session = self._sessions.get(message.user_id)
        if not session:
            return
        if session.state == SetupState.AWAITING_CUSTOM_MODEL:
            await self._handle_custom_model_text(session, message)
            return
        if session.state == SetupState.AWAITING_CUSTOM_PROMPT:
            session.selected_prompt_text = message.text.strip()
            session.selected_prompt_name = None
            session.bot_id = message.bot_id or session.bot_id
            await self._show_template_selection(session)
            return
        if session.state == SetupState.CONFIGURING_TEMPLATE_PARAMS:
            parsed = parse_kv_params(message.text)
            if parsed:
                session.template_params = parsed
            session.bot_id = message.bot_id or session.bot_id
            await self._finish_setup_from_session(session)

    async def _start_setup(self, user_id: str, chat_id: str) -> None:
        session = SetupSession(user_id=user_id, chat_id=chat_id)
        self._sessions[user_id] = session
        await self._show_model_selection(session)

    def _get_allowed_models_for_bot(self) -> list[str]:
        return model_picker.allowed_models_for_bot(self._settings, self._bot_config)

    def _get_available_prompts_for_bot(self) -> dict[str, str]:
        all_prompts = self._settings.get_available_prompts()
        if self._bot_config and self._bot_config.allowed_prompts:
            bot_set = set(self._bot_config.allowed_prompts)
            return {name: value for name, value in all_prompts.items() if name in bot_set}
        return all_prompts

    def _model_display_label(self, model_key: str) -> str:
        return model_picker.model_display_label(self._settings, model_key)

    async def _show_model_selection(self, session: SetupSession) -> None:
        session.state = SetupState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._get_allowed_models_for_bot()
        default_model = self._settings.get_default_model()
        for model in allowed_models:
            label = self._model_display_label(model)
            if model == default_model:
                label = f"{label} [default]"
            keyboard_rows.append([(label, f"model:{model}")])
        if self._custom_model_allowed(session.user_id):
            keyboard_rows.append(
                [(custom_model.CUSTOM_MODEL_LABEL, f"model:{custom_model.CUSTOM_MODEL_VALUE}")]
            )
        await self._messenger.send_message(
            OutgoingMessage(
                text="Choose an LLM model:",
                chat_id=session.chat_id,
                keyboard=self._messenger.build_inline_keyboard(keyboard_rows),
            )
        )

    async def show_model_change(self, message: IncomingMessage, current_model: str) -> None:
        """Show the in-place model switcher for an already-started chat."""
        await model_picker.show_model_picker(
            self._messenger,
            message,
            current_model=current_model,
            allowed_models=self._get_allowed_models_for_bot(),
            label_for=self._model_display_label,
            allow_custom=self._custom_model_allowed(message.user_id),
        )

    async def handle_model_change(self, message: IncomingMessage, model: str) -> None:
        """Switch the model for an existing chat without wiping its history."""
        if model == custom_model.CUSTOM_MODEL_VALUE:
            await model_picker.begin_custom_model_change(
                self._messenger,
                message,
                allowed=self._custom_model_allowed(message.user_id),
                chat_id=self._resolve_chat_id(message),
                pending=self._pending_custom_model,
            )
            return
        await model_picker.apply_model_change(
            self._messenger,
            message,
            model,
            chat_id=self._resolve_chat_id(message),
            allowed_models=self._get_allowed_models_for_bot(),
        )

    def is_awaiting_custom_model_change(self, user_id: str) -> bool:
        return user_id in self._pending_custom_model

    async def handle_custom_model_change_text(self, message: IncomingMessage) -> None:
        """Apply an in-place custom model switch from free-form text input."""
        await model_picker.apply_custom_model_change_text(
            self._messenger,
            message,
            chat_id=self._resolve_chat_id(message),
            pending=self._pending_custom_model,
        )

    async def _show_prompt_selection(self, session: SetupSession) -> None:
        session.state = SetupState.CHOOSING_PROMPT
        keyboard_rows = []
        for name in self._get_available_prompts_for_bot():
            keyboard_rows.append([(name.replace("_", " ").title(), f"prompt:{name}")])
        if not (self._bot_config and self._bot_config.allowed_prompts):
            keyboard_rows.append([("Custom (type your own)", "prompt:__custom__")])
        await self._messenger.send_message(
            OutgoingMessage(
                text=f"Model: {session.selected_model}\n\nNow choose a system prompt:",
                chat_id=session.chat_id,
                keyboard=self._messenger.build_inline_keyboard(keyboard_rows),
            )
        )

    async def _handle_model_selection(self, session: SetupSession, model: str) -> None:
        if model == custom_model.CUSTOM_MODEL_VALUE:
            await self._prompt_custom_model(session)
            return
        allowed = self._get_allowed_models_for_bot()
        if allowed and model not in allowed:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="This model is not available for this bot. Please choose another.",
                    chat_id=session.chat_id,
                )
            )
            return
        session.selected_model = model
        await self._show_prompt_selection(session)

    async def _prompt_custom_model(self, session: SetupSession) -> None:
        if not self._custom_model_allowed(session.user_id):
            await self._messenger.send_message(
                OutgoingMessage(
                    text="Custom models are not available for this bot.",
                    chat_id=session.chat_id,
                )
            )
            return
        session.state = SetupState.AWAITING_CUSTOM_MODEL
        await self._messenger.send_message(
            OutgoingMessage(text=custom_model.CUSTOM_MODEL_PROMPT, chat_id=session.chat_id)
        )

    async def _handle_custom_model_text(
        self, session: SetupSession, message: IncomingMessage
    ) -> None:
        model_name, params = custom_model.parse_custom_model_input(message.text)
        if not custom_model.validate_model_name(model_name):
            await self._messenger.send_message(
                OutgoingMessage(
                    text=custom_model.INVALID_MODEL_MESSAGE,
                    chat_id=session.chat_id,
                )
            )
            return
        session.selected_model = model_name
        session.custom_model_params = params or None
        session.bot_id = message.bot_id or session.bot_id
        await self._show_prompt_selection(session)

    async def _handle_prompt_selection(
        self,
        message: IncomingMessage,
        session: SetupSession,
        prompt_name: str,
    ) -> None:
        if prompt_name == "__custom__":
            if self._bot_config and self._bot_config.allowed_prompts:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="Custom prompts are not available for this bot.",
                        chat_id=session.chat_id,
                    )
                )
                return
            session.state = SetupState.AWAITING_CUSTOM_PROMPT
            await self._messenger.send_message(
                OutgoingMessage(text="Type your custom system prompt:", chat_id=session.chat_id)
            )
            return

        prompt_text = self._get_available_prompts_for_bot().get(prompt_name, "")
        if prompt_text:
            session.selected_prompt_name = prompt_name
            session.selected_prompt_text = prompt_text
            session.bot_id = message.bot_id or ""
            await self._show_template_selection(session)
            return

        await self._messenger.send_message(
            OutgoingMessage(
                text=f"Prompt '{prompt_name}' not found. Try again.",
                chat_id=session.chat_id,
            )
        )

    async def _show_template_selection(self, session: SetupSession) -> None:
        showed = await show_template_group_selection(
            session,
            self._messenger,
            self._settings,
            self._bot_config,
        )
        if not showed:
            await self._finish_setup_from_session(session)

    async def _handle_template_group_selection(
        self,
        message: IncomingMessage,
        session: SetupSession,
        value: str,
    ) -> None:
        if value.startswith("__single__:"):
            template_name = value[len("__single__:") :]
            session.selected_template = template_name if template_name != "empty" else None
            session.bot_id = message.bot_id or session.bot_id
            await self._maybe_show_params_or_finish(session, message)
            return

        from mai_gram.response_templates.registry import get_templates_in_group

        available_set = set(get_available_templates_for_bot(self._settings, self._bot_config))
        grp_templates = [t for t in get_templates_in_group(value) if t.name in available_set]

        if len(grp_templates) == 1:
            tpl = grp_templates[0]
            session.selected_template = tpl.name
            session.bot_id = message.bot_id or session.bot_id
            await self._maybe_show_params_or_finish(session, message)
            return

        session.state = SetupState.CHOOSING_TEMPLATE
        keyboard_rows = []
        for tpl in grp_templates:
            keyboard_rows.append([(tpl.description, f"template:{tpl.name}")])
        await self._messenger.send_message(
            OutgoingMessage(
                text="Choose a specific template variant:",
                chat_id=session.chat_id,
                keyboard=self._messenger.build_inline_keyboard(keyboard_rows),
            )
        )

    async def _handle_template_selection(
        self,
        message: IncomingMessage,
        session: SetupSession,
        template_name: str,
    ) -> None:
        available = get_available_templates_for_bot(self._settings, self._bot_config)
        if template_name not in available:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="This template is not available for this bot. Please choose another.",
                    chat_id=session.chat_id,
                )
            )
            return
        session.selected_template = template_name if template_name != "empty" else None
        session.bot_id = message.bot_id or session.bot_id
        await self._maybe_show_params_or_finish(session, message)

    async def _maybe_show_params_or_finish(
        self,
        session: SetupSession,
        message: IncomingMessage | None = None,
    ) -> None:
        from mai_gram.response_templates.registry import get_template as _get_tpl

        tpl = _get_tpl(session.selected_template)
        if tpl.get_params():
            await show_template_params_summary(session, tpl, self._messenger)
        else:
            await self._finish_setup_from_session(session, message=message)

    async def _finish_setup_from_session(
        self,
        session: SetupSession,
        *,
        message: IncomingMessage | None = None,
    ) -> None:
        """Complete setup using all values stored in the session."""
        if message is None:
            from mai_gram.messenger.base import MessageType

            message = IncomingMessage(
                platform="internal",
                text="",
                chat_id=session.chat_id,
                user_id=session.user_id,
                message_id="setup-finish",
                message_type=MessageType.TEXT,
                bot_id=session.bot_id,
            )
        await self._finish_setup(
            message,
            session,
            session.selected_prompt_text,
            prompt_name=session.selected_prompt_name,
            template_name=session.selected_template,
            template_params=session.template_params,
        )

    async def _finish_setup(
        self,
        message: IncomingMessage,
        session: SetupSession,
        system_prompt: str,
        *,
        prompt_name: str | None = None,
        template_name: str | None = None,
        template_params: dict[str, str] | None = None,
    ) -> None:
        chat_id = self._resolve_chat_id(message)
        prompt_cfg = self._settings.get_prompt_config(prompt_name) if prompt_name else None
        params_json = json.dumps(template_params, ensure_ascii=False) if template_params else None
        custom_params = session.custom_model_params
        custom_params_json = (
            json.dumps(custom_params, ensure_ascii=False) if custom_params else None
        )

        chat = setup_finalizer.build_chat_record(
            chat_id,
            message,
            session,
            system_prompt,
            settings=self._settings,
            prompt_name=prompt_name,
            template_name=template_name,
            params_json=params_json,
            prompt_cfg=prompt_cfg,
            custom_model_params_json=custom_params_json,
        )
        async with get_session() as db:
            db.add(chat)
            await db.commit()

        self.clear_setup_session(message.user_id)
        await setup_finalizer.send_setup_confirmation(
            self._messenger, message, session, chat, system_prompt, template_name
        )

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
