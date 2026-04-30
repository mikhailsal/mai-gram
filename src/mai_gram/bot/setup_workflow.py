"""Setup workflow for chat creation and prompt/model selection."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

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
    CHOOSING_PROMPT = "choosing_prompt"
    AWAITING_CUSTOM_PROMPT = "awaiting_custom_prompt"


@dataclass
class SetupSession:
    """Tracks the state of an ongoing setup session."""

    user_id: str
    chat_id: str
    state: SetupState = SetupState.CHOOSING_MODEL
    selected_model: str = ""


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

    def is_in_setup(self, user_id: str) -> bool:
        return user_id in self._sessions

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
        """Handle setup callback queries for model and prompt selection."""
        session = self._sessions.get(message.user_id)
        if not session or not message.callback_data:
            return

        parts = message.callback_data.split(":", 1)
        if len(parts) != 2:
            return

        category, value = parts
        if category == "model" and session.state == SetupState.CHOOSING_MODEL:
            await self._handle_model_selection(session, value)
            return
        if category == "prompt" and session.state == SetupState.CHOOSING_PROMPT:
            await self._handle_prompt_selection(message, session, value)

    async def handle_setup_text(self, message: IncomingMessage) -> None:
        """Handle free-form text entered during the setup flow."""
        session = self._sessions.get(message.user_id)
        if not session:
            return

        if session.state == SetupState.AWAITING_CUSTOM_PROMPT:
            await self._finish_setup(message, session, message.text.strip())

    async def _start_setup(self, user_id: str, chat_id: str) -> None:
        session = SetupSession(user_id=user_id, chat_id=chat_id)
        self._sessions[user_id] = session
        await self._show_model_selection(session)

    def _get_allowed_models_for_bot(self) -> list[str]:
        global_models = self._settings.get_allowed_models()
        if self._bot_config and self._bot_config.allowed_models:
            bot_set = set(self._bot_config.allowed_models)
            return [model for model in global_models if model in bot_set]
        return global_models

    def _get_available_prompts_for_bot(self) -> dict[str, str]:
        all_prompts = self._settings.get_available_prompts()
        if self._bot_config and self._bot_config.allowed_prompts:
            bot_set = set(self._bot_config.allowed_prompts)
            return {name: value for name, value in all_prompts.items() if name in bot_set}
        return all_prompts

    async def _show_model_selection(self, session: SetupSession) -> None:
        session.state = SetupState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._get_allowed_models_for_bot()
        default_model = self._settings.get_default_model()
        for model in allowed_models:
            short_name = model.split("/")[-1] if "/" in model else model
            label = f"{short_name} [default]" if model == default_model else short_name
            keyboard_rows.append([(label, f"model:{model}")])

        await self._messenger.send_message(
            OutgoingMessage(
                text="Choose an LLM model:",
                chat_id=session.chat_id,
                keyboard=self._messenger.build_inline_keyboard(keyboard_rows),
            )
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
                OutgoingMessage(
                    text="Type your custom system prompt:",
                    chat_id=session.chat_id,
                )
            )
            return

        prompt_text = self._get_available_prompts_for_bot().get(prompt_name, "")
        if prompt_text:
            await self._finish_setup(message, session, prompt_text, prompt_name=prompt_name)
            return

        await self._messenger.send_message(
            OutgoingMessage(
                text=f"Prompt '{prompt_name}' not found. Try again.",
                chat_id=session.chat_id,
            )
        )

    async def _finish_setup(
        self,
        message: IncomingMessage,
        session: SetupSession,
        system_prompt: str,
        *,
        prompt_name: str | None = None,
    ) -> None:
        chat_id = self._resolve_chat_id(message)
        prompt_cfg = self._settings.get_prompt_config(prompt_name) if prompt_name else None

        async with get_session() as db:
            send_dt = True
            if prompt_cfg is not None and prompt_cfg.send_datetime is not None:
                send_dt = prompt_cfg.send_datetime

            chat = Chat(
                id=chat_id,
                user_id=message.user_id,
                bot_id=message.bot_id or "",
                llm_model=session.selected_model,
                system_prompt=system_prompt,
                prompt_name=prompt_name,
                timezone=self._settings.default_timezone,
                show_reasoning=prompt_cfg.show_reasoning if prompt_cfg else True,
                show_tool_calls=prompt_cfg.show_tool_calls if prompt_cfg else True,
                send_datetime=send_dt,
            )
            db.add(chat)
            await db.commit()

        self.clear_setup_session(message.user_id)

        reasoning_status = "ON" if chat.show_reasoning else "OFF"
        toolcalls_status = "ON" if chat.show_tool_calls else "OFF"
        datetime_status = "ON" if chat.send_datetime else "OFF"
        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    "Chat created!\n"
                    f"Model: {session.selected_model}\n"
                    f"Prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}\n"
                    f"Reasoning: {reasoning_status} | Tool calls: {toolcalls_status} "
                    f"| Datetime: {datetime_status}\n\n"
                    "Send a message to start chatting.\n"
                    "Toggle display with /reasoning, /toolcalls, and /datetime."
                ),
                chat_id=message.chat_id,
            )
        )
        logger.info(
            "Created chat: id=%s model=%s prompt_len=%d reasoning=%s toolcalls=%s",
            chat_id,
            session.selected_model,
            len(system_prompt),
            chat.show_reasoning,
            chat.show_tool_calls,
        )

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
