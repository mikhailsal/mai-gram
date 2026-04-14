"""Telegram bot message handlers.

Connects incoming Telegram messages to the LLM conversation engine.
Handles:
- /start command (setup: model selection + prompt selection)
- /reset command (deletes chat configuration)
- /model command (shows/changes current model)
- /help command
- Regular messages (conversation)
- Callback queries (button presses during setup)
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.middleware import MessageLogger, RateLimiter, RateLimitConfig
from mai_gram.config import get_settings
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_gram.mcp_servers.bridge import run_with_tools
from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.mcp_servers.wiki_server import WikiMCPServer
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


def make_chat_id(user_id: str, bot_id: str) -> str:
    """Create a composite chat ID from user_id and bot_id.

    Format: ``{user_id}@{bot_id}``
    """
    return f"{user_id}@{bot_id}"


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


class BotHandler:
    """Main handler for bot messages and commands.

    Coordinates between the messenger, database, and LLM provider.
    """

    def __init__(
        self,
        messenger: Messenger,
        llm_provider: LLMProvider,
        *,
        rate_limit_config: RateLimitConfig | None = None,
        memory_data_dir: str | None = None,
        wiki_context_limit: int | None = None,
        short_term_limit: int | None = None,
        tool_max_iterations: int | None = None,
        test_mode: bool = False,
    ) -> None:
        self._messenger = messenger
        self._llm = llm_provider
        self._rate_limiter = RateLimiter(
            rate_limit_config,
            on_rate_limited=self._handle_rate_limited,
        )
        self._message_logger = MessageLogger(log_content=False)
        self._test_mode = test_mode
        self._setup_sessions: dict[str, SetupSession] = {}

        settings = get_settings()
        self._memory_data_dir = memory_data_dir or settings.memory_data_dir
        self._wiki_context_limit = wiki_context_limit or settings.wiki_context_limit
        self._short_term_limit = short_term_limit or settings.short_term_limit
        self._tool_max_iterations = tool_max_iterations or settings.tool_max_iterations
        self._allowed_users = settings.get_allowed_user_ids()
        self._allowed_models = settings.get_allowed_models()
        self._default_model = settings.get_default_model()
        self._available_prompts = settings.get_available_prompts()

        if self._allowed_users:
            logger.info(
                "Access control enabled: %d user(s) allowed",
                len(self._allowed_users),
            )

        messenger.register_command_handler("start", self._handle_start)
        messenger.register_command_handler("reset", self._handle_reset)
        messenger.register_command_handler("model", self._handle_model)
        messenger.register_command_handler("help", self._handle_help)
        messenger.register_message_handler(self._handle_message)
        messenger.register_callback_handler(self._handle_callback)

    # -- Setup session helpers --

    def is_in_setup(self, user_id: str) -> bool:
        return user_id in self._setup_sessions

    def get_setup_session(self, user_id: str) -> SetupSession | None:
        return self._setup_sessions.get(user_id)

    def clear_setup_session(self, user_id: str) -> None:
        self._setup_sessions.pop(user_id, None)

    # -- Access control --

    async def _handle_rate_limited(self, user_id: str, chat_id: str) -> None:
        await self._messenger.send_message(
            OutgoingMessage(
                text="Slow down! Too many messages. Wait a moment and try again.",
                chat_id=chat_id,
            )
        )

    async def _check_access(self, message: IncomingMessage) -> bool:
        if not self._allowed_users:
            return True
        if message.user_id in self._allowed_users:
            return True
        logger.warning("Access denied for user_id=%s", message.user_id)
        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    "Access denied. This is a private bot. "
                    f"Your user ID: {message.user_id}"
                ),
                chat_id=message.chat_id,
            )
        )
        return False

    def _chat_id_for(self, message: IncomingMessage) -> str:
        if message.bot_id:
            return make_chat_id(message.user_id, message.bot_id)
        return message.chat_id

    # -- Command handlers --

    async def _handle_start(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        chat_id = self._chat_id_for(message)

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

    async def _handle_reset(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if chat:
                await session.delete(chat)
                await session.commit()
                msg = "Chat reset. All history deleted. Use /start to create a new one."
            else:
                msg = "No chat to reset. Use /start to create one."

        self.clear_setup_session(message.user_id)

        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

    async def _handle_model(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No chat exists yet. Use /start to create one.",
                        chat_id=message.chat_id,
                    )
                )
                return

            await self._messenger.send_message(
                OutgoingMessage(
                    text=f"Current model: {chat.llm_model}\n\nUse /reset + /start to change.",
                    chat_id=message.chat_id,
                )
            )

    async def _handle_help(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        msg = (
            "Available commands:\n\n"
            "/start - Set up a new chat (choose model + prompt)\n"
            "/reset - Delete current chat and history\n"
            "/model - Show current model\n"
            "/help - Show this help message\n\n"
            "Just send a message to chat!"
        )
        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

    # -- Message handler --

    async def _handle_message(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return
        if not await self._rate_limiter.check_rate_limit(message.user_id, message.chat_id):
            return

        if self.is_in_setup(message.user_id):
            await self._handle_setup_text(message)
            return

        await self._handle_conversation(message)

    async def _handle_callback(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        if self.is_in_setup(message.user_id):
            await self._handle_setup_callback(message)
            return

        logger.debug("Unhandled callback: %s", message.callback_data)

    # -- Setup flow --

    async def _start_setup(self, user_id: str, chat_id: str) -> None:
        session = SetupSession(user_id=user_id, chat_id=chat_id)
        self._setup_sessions[user_id] = session
        await self._show_model_selection(session)

    async def _show_model_selection(self, session: SetupSession) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = SetupState.CHOOSING_MODEL
        keyboard_rows = []
        for model in self._allowed_models:
            short_name = model.split("/")[-1] if "/" in model else model
            label = f"{short_name} [default]" if model == self._default_model else short_name
            keyboard_rows.append([(label, f"model:{model}")])

        kb = build_inline_keyboard(keyboard_rows)

        await self._messenger.send_message(
            OutgoingMessage(
                text="Choose an LLM model:",
                chat_id=session.chat_id,
                keyboard=kb,
            )
        )

    async def _show_prompt_selection(self, session: SetupSession) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = SetupState.CHOOSING_PROMPT
        keyboard_rows = []
        for name in self._available_prompts:
            keyboard_rows.append([(name.replace("_", " ").title(), f"prompt:{name}")])
        keyboard_rows.append([("Custom (type your own)", "prompt:__custom__")])

        kb = build_inline_keyboard(keyboard_rows)

        await self._messenger.send_message(
            OutgoingMessage(
                text=f"Model: {session.selected_model}\n\nNow choose a system prompt:",
                chat_id=session.chat_id,
                keyboard=kb,
            )
        )

    async def _handle_setup_callback(self, message: IncomingMessage) -> None:
        session = self._setup_sessions.get(message.user_id)
        if not session or not message.callback_data:
            return

        parts = message.callback_data.split(":", 1)
        if len(parts) != 2:
            return

        category, value = parts

        if category == "model" and session.state == SetupState.CHOOSING_MODEL:
            session.selected_model = value
            await self._show_prompt_selection(session)

        elif category == "prompt" and session.state == SetupState.CHOOSING_PROMPT:
            if value == "__custom__":
                session.state = SetupState.AWAITING_CUSTOM_PROMPT
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="Type your custom system prompt:",
                        chat_id=session.chat_id,
                    )
                )
            else:
                prompt_text = self._available_prompts.get(value, "")
                if prompt_text:
                    await self._finish_setup(message, session, prompt_text)
                else:
                    await self._messenger.send_message(
                        OutgoingMessage(
                            text=f"Prompt '{value}' not found. Try again.",
                            chat_id=session.chat_id,
                        )
                    )

    async def _handle_setup_text(self, message: IncomingMessage) -> None:
        session = self._setup_sessions.get(message.user_id)
        if not session:
            return

        if session.state == SetupState.AWAITING_CUSTOM_PROMPT:
            await self._finish_setup(message, session, message.text.strip())

    async def _finish_setup(
        self, message: IncomingMessage, session: SetupSession, system_prompt: str
    ) -> None:
        chat_id = self._chat_id_for(message)
        user_id = message.user_id
        bot_id = message.bot_id or ""

        async with get_session() as db:
            chat = Chat(
                id=chat_id,
                user_id=user_id,
                bot_id=bot_id,
                llm_model=session.selected_model,
                system_prompt=system_prompt,
            )
            db.add(chat)
            await db.commit()

        self.clear_setup_session(message.user_id)

        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    f"Chat created!\n"
                    f"Model: {session.selected_model}\n"
                    f"Prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}\n\n"
                    "Send a message to start chatting."
                ),
                chat_id=message.chat_id,
            )
        )
        logger.info(
            "Created chat: id=%s model=%s prompt_len=%d",
            chat_id,
            session.selected_model,
            len(system_prompt),
        )

    # -- Conversation --

    async def _handle_conversation(self, message: IncomingMessage) -> None:
        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No chat configured. Use /start to set one up.",
                        chat_id=message.chat_id,
                    )
                )
                return

            await self._messenger.send_typing_indicator(message.chat_id)

            message_store = MessageStore(session)
            wiki_store = WikiStore(session, data_dir=self._memory_data_dir)

            prompt_builder = PromptBuilder(
                self._llm,
                message_store,
                wiki_store,
                wiki_context_limit=self._wiki_context_limit,
                short_term_limit=self._short_term_limit,
                test_mode=self._test_mode,
            )

            mcp_manager = MCPManager()
            mcp_manager.register_server(
                "messages",
                MessagesMCPServer(message_store, chat.id),
            )
            mcp_manager.register_server(
                "wiki",
                WikiMCPServer(wiki_store, chat.id),
            )

            now = datetime.now(timezone.utc)
            await message_store.save_message(
                chat.id, "user", message.text, timestamp=now,
            )

            llm_messages = await prompt_builder.build_context(chat, current_time=now)

            response_text: str | None = None
            try:
                response = await run_with_tools(
                    self._llm,
                    mcp_manager,
                    llm_messages,
                    model=chat.llm_model,
                    max_iterations=self._tool_max_iterations,
                )
                response_text = response.content

                if response_text and response_text.strip():
                    await message_store.save_message(
                        chat.id, "assistant", response_text, timestamp=datetime.now(timezone.utc),
                    )

            except Exception:
                logger.exception("Failed to generate response")
                response_text = "Error generating response. Please try again."

        if response_text and response_text.strip():
            result = await self._messenger.send_message(
                OutgoingMessage(text=response_text, chat_id=message.chat_id)
            )
            self._message_logger.log_outgoing(
                message.chat_id,
                response_text,
                success=result.success,
                message_id=result.message_id,
            )

    # -- DB helpers --

    async def _get_chat(self, session: object, chat_id: str) -> Chat | None:
        result = await session.execute(  # type: ignore[union-attr]
            select(Chat).where(Chat.id == chat_id)
        )
        return result.scalar_one_or_none()  # type: ignore[union-attr]
