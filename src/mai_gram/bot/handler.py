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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.middleware import MessageLogger, RateLimitConfig, RateLimiter
from mai_gram.config import get_settings
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.llm.provider import LLMProvider
from mai_gram.mcp_servers.bridge import run_with_tools_stream
from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.mcp_servers.wiki_server import WikiMCPServer
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

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
        external_mcp_pool: object | None = None,
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
        self._response_message_ids: dict[str, list[str]] = {}

        settings = get_settings()
        self._memory_data_dir = memory_data_dir or settings.memory_data_dir
        self._wiki_context_limit = wiki_context_limit or settings.wiki_context_limit
        self._short_term_limit = short_term_limit or settings.short_term_limit
        self._tool_max_iterations = tool_max_iterations or settings.tool_max_iterations
        self._settings = settings
        self._allowed_users = settings.get_allowed_user_ids()
        self._external_mcp_pool = external_mcp_pool

        if self._allowed_users:
            logger.info(
                "Access control enabled: %d user(s) allowed",
                len(self._allowed_users),
            )

        messenger.register_command_handler(
            "start", self._handle_start, description="Set up a new chat",
        )
        messenger.register_command_handler(
            "reset", self._handle_reset, description="Delete chat and history",
        )
        messenger.register_command_handler(
            "model", self._handle_model, description="Show current model",
        )
        messenger.register_command_handler(
            "help", self._handle_help, description="Show available commands",
        )
        messenger.register_command_handler(
            "datetime", self._handle_datetime_toggle,
            description="Toggle date/time in messages",
        )
        messenger.register_command_handler(
            "timezone", self._handle_timezone,
            description="Set timezone (e.g. /timezone Europe/Moscow)",
        )
        messenger.register_command_handler(
            "reasoning", self._handle_reasoning_toggle,
            description="Toggle reasoning display",
        )
        messenger.register_command_handler(
            "toolcalls", self._handle_toolcalls_toggle,
            description="Toggle tool call display",
        )
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
            "/timezone - Set timezone (e.g. /timezone Europe/Moscow)\n"
            "/datetime - Toggle date/time in messages sent to LLM\n"
            "/reasoning - Toggle display of LLM reasoning\n"
            "/toolcalls - Toggle display of tool call details\n"
            "/help - Show this help message\n\n"
            "Just send a message to chat!"
        )
        await self._messenger.send_message(
            OutgoingMessage(text=msg, chat_id=message.chat_id)
        )

    async def _toggle_chat_flag(
        self, message: IncomingMessage, field_name: str, label: str
    ) -> None:
        """Generic toggle for boolean chat settings."""
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
            current_value = getattr(chat, field_name)
            new_value = not current_value
            setattr(chat, field_name, new_value)
            await session.commit()

        status = "ON" if new_value else "OFF"
        await self._messenger.send_message(
            OutgoingMessage(
                text=f"{label}: {status}",
                chat_id=message.chat_id,
            )
        )

    async def _handle_datetime_toggle(self, message: IncomingMessage) -> None:
        await self._toggle_chat_flag(message, "send_datetime", "Date/time in messages")

    async def _handle_reasoning_toggle(self, message: IncomingMessage) -> None:
        await self._toggle_chat_flag(message, "show_reasoning", "Reasoning display")

    async def _handle_toolcalls_toggle(self, message: IncomingMessage) -> None:
        await self._toggle_chat_flag(message, "show_tool_calls", "Tool call display")

    async def _handle_timezone(self, message: IncomingMessage) -> None:
        """Handle /timezone command -- show or set the chat's timezone."""
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        chat_id = self._chat_id_for(message)
        tz_arg = (message.command_args or "").strip()

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

            if not tz_arg:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text=f"Current timezone: {chat.timezone}\n\nUsage: /timezone Europe/Moscow",
                        chat_id=message.chat_id,
                    )
                )
                return

            from zoneinfo import available_timezones

            if tz_arg not in available_timezones():
                await self._messenger.send_message(
                    OutgoingMessage(
                        text=(
                            f"Unknown timezone: {tz_arg}\n\n"
                            "Examples: Europe/Moscow, US/Eastern, Asia/Tokyo, UTC"
                        ),
                        chat_id=message.chat_id,
                    )
                )
                return

            chat.timezone = tz_arg
            await session.commit()

        await self._messenger.send_message(
            OutgoingMessage(
                text=f"Timezone set to: {tz_arg}",
                chat_id=message.chat_id,
            )
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

        data = message.callback_data or ""

        if data == "regen":
            await self._show_confirmation(
                message, "Regenerate this response?",
                confirm_data="confirm_regen", cancel_data="cancel_action",
            )
            return

        if data.startswith("cut:"):
            cut_msg_id = data.split(":", 1)[1]
            preview = await self._get_message_preview(int(cut_msg_id))
            confirm_text = (
                "Cut history above this message?\n"
                "Older messages won't be sent to AI but remain searchable."
            )
            if preview:
                confirm_text += f'\n\nMessage: "{preview}"'
            await self._show_confirmation(
                message,
                confirm_text,
                confirm_data=f"confirm_cut:{cut_msg_id}",
                cancel_data="cancel_action",
            )
            return

        if data == "confirm_regen":
            await self._delete_callback_message(message)
            await self._handle_regenerate(message)
            return

        if data.startswith("confirm_cut:"):
            cut_msg_id_str = data.split(":", 1)[1]
            await self._delete_callback_message(message)
            await self._handle_cut_above(message, int(cut_msg_id_str))
            return

        if data == "cancel_action":
            await self._delete_callback_message(message)
            return

        logger.debug("Unhandled callback: %s", data)

    # -- Setup flow --

    async def _start_setup(self, user_id: str, chat_id: str) -> None:
        session = SetupSession(user_id=user_id, chat_id=chat_id)
        self._setup_sessions[user_id] = session
        await self._show_model_selection(session)

    async def _show_model_selection(self, session: SetupSession) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = SetupState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._settings.get_allowed_models()
        default_model = self._settings.get_default_model()
        for model in allowed_models:
            short_name = model.split("/")[-1] if "/" in model else model
            label = f"{short_name} [default]" if model == default_model else short_name
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
        available_prompts = self._settings.get_available_prompts()
        for name in available_prompts:
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
                prompt_text = self._settings.get_available_prompts().get(value, "")
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
                timezone=self._settings.default_timezone,
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
        import html as _html

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

            enabled_tools, disabled_tools = self._settings.get_tool_filter()
            mcp_manager = MCPManager(
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
            )
            mcp_manager.register_server(
                "messages",
                MessagesMCPServer(message_store, chat.id),
            )
            mcp_manager.register_server(
                "wiki",
                WikiMCPServer(wiki_store, chat.id),
            )
            if self._external_mcp_pool is not None:
                for srv_name, srv in self._external_mcp_pool.get_all_servers().items():
                    mcp_manager.register_server(f"ext:{srv_name}", srv)

            now = datetime.now(timezone.utc)
            chat_tz = chat.timezone
            await message_store.save_message(
                chat.id, "user", message.text, timestamp=now,
                timezone_name=chat_tz,
            )

            llm_messages = await prompt_builder.build_context(
                chat,
                current_time=now,
                send_datetime=chat.send_datetime,
                chat_timezone=chat.timezone,
                cut_above_message_id=chat.cut_above_message_id,
            )

            extra_params = self._settings.get_model_params(chat.llm_model)
            show_tool_calls = chat.show_tool_calls
            show_reasoning = chat.show_reasoning
            tg_chat_id = message.chat_id
            sent_msg_ids: list[str] = []

            async def _on_tool_call_display(
                *, content: str, tool_calls_json: str
            ) -> None:
                await message_store.save_message(
                    chat.id, "assistant", content or "",
                    tool_calls=tool_calls_json,
                    timezone_name=chat_tz,
                )

                if not show_tool_calls:
                    return
                try:
                    calls = json.loads(tool_calls_json)
                except (json.JSONDecodeError, TypeError):
                    return
                lines = []
                for tc in calls:
                    name = tc.get("name", "?")
                    args = tc.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args) if isinstance(args, str) else args
                        args_str = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        args_str = str(args)
                    lines.append(f"🔧 {name}({args_str})")
                if lines:
                    r = await self._messenger.send_message(
                        OutgoingMessage(text="\n".join(lines), chat_id=tg_chat_id)
                    )
                    if r.success and r.message_id:
                        sent_msg_ids.append(r.message_id)

            async def _on_tool_result_display(
                *,
                tool_call_id: str,
                tool_name: str,
                arguments: str,
                result: object,
                content: str,
                error: str | None,
                server_name: str | None,
            ) -> None:
                await message_store.save_message(
                    chat.id, "tool", content,
                    tool_call_id=tool_call_id,
                    timezone_name=chat_tz,
                )

                if not show_tool_calls:
                    return
                if error:
                    text = f"❌ {tool_name}: {error}"
                else:
                    result_str = str(result) if result is not None else ""
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "…"
                    text = f"✅ {tool_name}: {result_str}" if result_str else f"✅ {tool_name}"
                r = await self._messenger.send_message(
                    OutgoingMessage(text=text, chat_id=tg_chat_id)
                )
                if r.success and r.message_id:
                    sent_msg_ids.append(r.message_id)

            response_text: str | None = None
            response_reasoning: str | None = None
            stream_usage = None
            stream_cost: float | None = None
            stream_is_byok = False
            try:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                placeholder_msg_id: str | None = None
                last_edit_time = 0.0
                last_display_len = 0
                edit_interval = 1.0
                edit_min_chars = 60

                async for chunk in run_with_tools_stream(
                    self._llm,
                    mcp_manager,
                    llm_messages,
                    model=chat.llm_model,
                    max_iterations=self._tool_max_iterations,
                    extra_params=extra_params or None,
                    on_assistant_tool_call=_on_tool_call_display,
                    on_tool_result=_on_tool_result_display,
                ):
                    if chunk.usage is not None:
                        stream_usage = chunk.usage
                        stream_cost = chunk.cost
                        stream_is_byok = chunk.is_byok

                    if chunk.turn_complete:
                        if placeholder_msg_id:
                            turn_text = self._build_intermediate_display(
                                "".join(content_parts),
                                "".join(reasoning_parts),
                                show_reasoning,
                            )
                            if turn_text.strip():
                                await self._messenger.edit_message(
                                    tg_chat_id, placeholder_msg_id, turn_text,
                                )
                            sent_msg_ids.append(placeholder_msg_id)
                        content_parts.clear()
                        reasoning_parts.clear()
                        placeholder_msg_id = None
                        last_edit_time = 0.0
                        last_display_len = 0
                        continue

                    if chunk.reasoning:
                        reasoning_parts.append(chunk.reasoning)
                    if chunk.content:
                        content_parts.append(chunk.content)

                    current_reasoning = "".join(reasoning_parts)
                    current_content = "".join(content_parts)

                    display_text = ""
                    live_parse_mode: str | None = None
                    if show_reasoning and current_reasoning.strip():
                        from mai_gram.core.md_to_telegram import markdown_to_html

                        r_esc = _html.escape(current_reasoning.strip())
                        c_html = (
                            markdown_to_html(current_content)
                            if current_content.strip() else ""
                        )
                        display_text = (
                            f"<blockquote>\U0001f4ad Reasoning\n"
                            f"{r_esc}</blockquote>"
                        )
                        if c_html:
                            display_text += f"\n\n{c_html}"
                        live_parse_mode = "html"
                    elif current_content.strip():
                        from mai_gram.core.md_to_telegram import markdown_to_html

                        display_text = markdown_to_html(current_content)
                        live_parse_mode = "html"
                    else:
                        continue

                    display_len = len(current_reasoning) + len(current_content)
                    now_mono = time.monotonic()
                    chars_since_edit = display_len - last_display_len
                    time_since_edit = now_mono - last_edit_time

                    should_edit = (
                        display_text.strip()
                        and (time_since_edit >= edit_interval or chars_since_edit >= edit_min_chars)
                        and chars_since_edit > 0
                    )

                    if should_edit:
                        live_text = display_text + " \u258d"
                        if placeholder_msg_id is None:
                            result = await self._messenger.send_message(
                                OutgoingMessage(
                                    text=live_text,
                                    chat_id=tg_chat_id,
                                    parse_mode=live_parse_mode,
                                )
                            )
                            if not result.success and live_parse_mode:
                                fallback = current_content or current_reasoning
                                result = await self._messenger.send_message(
                                    OutgoingMessage(
                                        text=fallback + " \u258d",
                                        chat_id=tg_chat_id,
                                    )
                                )
                            if result.success:
                                placeholder_msg_id = result.message_id
                        else:
                            edit_result = await self._messenger.edit_message(
                                tg_chat_id,
                                placeholder_msg_id,
                                live_text,
                                parse_mode=live_parse_mode,
                            )
                            if not edit_result.success and live_parse_mode:
                                fallback = current_content or current_reasoning
                                await self._messenger.edit_message(
                                    tg_chat_id,
                                    placeholder_msg_id,
                                    fallback + " \u258d",
                                )
                        last_edit_time = now_mono
                        last_display_len = display_len

                response_text = "".join(content_parts)
                response_reasoning = "".join(reasoning_parts) or None

                saved_msg_id: int | None = None
                if response_text and response_text.strip():
                    saved_msg = await message_store.save_message(
                        chat.id, "assistant", response_text,
                        timestamp=datetime.now(timezone.utc),
                        reasoning=response_reasoning,
                        timezone_name=chat_tz,
                    )
                    saved_msg_id = saved_msg.id

            except Exception:
                logger.exception("Failed to generate response")
                response_text = "Error generating response. Please try again."
                placeholder_msg_id = None
                saved_msg_id = None

        self._response_message_ids[tg_chat_id] = sent_msg_ids

        from mai_gram.messenger.telegram import build_inline_keyboard

        kb_buttons = [[("\U0001f504 Regenerate", "regen")]]
        if saved_msg_id is not None:
            kb_buttons[0].append(("\u2702 Cut above", f"cut:{saved_msg_id}"))
        action_kb = build_inline_keyboard(kb_buttons)

        usage_footer = self._format_usage_footer(
            stream_usage, stream_cost, stream_is_byok
        )

        if placeholder_msg_id and response_text and response_text.strip():
            if show_reasoning and response_reasoning and response_reasoning.strip():
                from mai_gram.core.md_to_telegram import markdown_to_html

                escaped_r = _html.escape(response_reasoning.strip())
                html_body = markdown_to_html(response_text)
                footer_html = (
                    f"\n\n<i>{_html.escape(usage_footer)}</i>" if usage_footer else ""
                )
                display_text = (
                    f"<blockquote expandable>\U0001f4ad Reasoning\n"
                    f"{escaped_r}</blockquote>\n\n{html_body}{footer_html}"
                )
                await self._messenger.edit_message(
                    tg_chat_id, placeholder_msg_id, display_text,
                    parse_mode="html", keyboard=action_kb,
                )
            else:
                text_with_footer = response_text
                if usage_footer:
                    text_with_footer += f"\n\n{usage_footer}"
                await self._edit_with_mdv2_fallback(
                    tg_chat_id, placeholder_msg_id, text_with_footer,
                    keyboard=action_kb,
                )
            sent_msg_ids.append(placeholder_msg_id)
        elif response_text and response_text.strip():
            text_with_footer = response_text
            if usage_footer:
                text_with_footer += f"\n\n{usage_footer}"
            final_msg_id = await self._send_response(
                tg_chat_id,
                response_text=text_with_footer,
                response_reasoning=response_reasoning,
                show_reasoning=show_reasoning,
                keyboard=action_kb,
            )
            if final_msg_id:
                sent_msg_ids.append(final_msg_id)

    @staticmethod
    def _build_intermediate_display(
        content: str, reasoning: str, show_reasoning: bool
    ) -> str:
        """Build display text for an intermediate turn (before tool calls)."""
        display = ""
        if show_reasoning and reasoning.strip():
            display = f"\U0001f4ad Reasoning:\n{reasoning.strip()}"
            if content.strip():
                display += "\n\n\u2500\u2500\u2500\n\n" + content
        elif content.strip():
            display = content
        return display

    @staticmethod
    def _format_usage_footer(
        usage: object, cost: float | None, is_byok: bool
    ) -> str:
        """Build a compact token/cost footer string."""
        if usage is None:
            return ""
        prompt_t = getattr(usage, "prompt_tokens", 0)
        comp_t = getattr(usage, "completion_tokens", 0)
        parts = [f"{prompt_t}/{comp_t} tokens"]
        if cost is not None and cost > 0:
            parts.append(f"${cost:.4f}")
        return " | ".join(parts)

    async def _send_with_mdv2_fallback(
        self,
        chat_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> str | None:
        """Try sending as MarkdownV2; fall back to plain text on failure."""
        from mai_gram.core.md_to_telegram import markdown_to_mdv2

        mdv2_text = markdown_to_mdv2(text)
        result = await self._messenger.send_message(
            OutgoingMessage(
                text=mdv2_text,
                chat_id=chat_id,
                parse_mode="markdown",
                keyboard=keyboard,
            )
        )
        if result.success:
            return result.message_id

        logger.warning("MarkdownV2 send failed, falling back to plain text")
        result = await self._messenger.send_message(
            OutgoingMessage(text=text, chat_id=chat_id, keyboard=keyboard)
        )
        return result.message_id if result.success else None

    async def _edit_with_mdv2_fallback(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> None:
        """Try editing as MarkdownV2; fall back to plain text on failure."""
        from mai_gram.core.md_to_telegram import markdown_to_mdv2

        mdv2_text = markdown_to_mdv2(text)
        logger.info(
            "MarkdownV2 conversion: input=%d chars, output=%d chars",
            len(text), len(mdv2_text),
        )
        result = await self._messenger.edit_message(
            chat_id, message_id, mdv2_text,
            parse_mode="markdown", keyboard=keyboard,
        )
        if not result.success:
            logger.warning(
                "MarkdownV2 edit failed (error=%s), falling back to plain text",
                result.error,
            )
            await self._messenger.edit_message(
                chat_id, message_id, text, keyboard=keyboard,
            )

    async def _send_response(
        self,
        chat_id: str,
        *,
        response_text: str | None,
        response_reasoning: str | None = None,
        show_reasoning: bool = False,
        keyboard: object = None,
    ) -> str | None:
        """Send the final assistant response, optionally with reasoning.

        Returns the Telegram message ID of the sent message, or None.
        """
        if not response_text or not response_text.strip():
            return None

        import html as _html

        if show_reasoning and response_reasoning and response_reasoning.strip():
            from mai_gram.core.md_to_telegram import markdown_to_html

            escaped_r = _html.escape(response_reasoning.strip())
            html_body = markdown_to_html(response_text)
            display_text = (
                f"<blockquote expandable>\U0001f4ad Reasoning\n"
                f"{escaped_r}</blockquote>\n\n{html_body}"
            )
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=display_text,
                    chat_id=chat_id,
                    parse_mode="html",
                    keyboard=keyboard,
                )
            )
        else:
            msg_id = await self._send_with_mdv2_fallback(
                chat_id, response_text, keyboard=keyboard,
            )
            self._message_logger.log_outgoing(
                chat_id, response_text, success=msg_id is not None, message_id=msg_id,
            )
            return msg_id

        self._message_logger.log_outgoing(
            chat_id,
            response_text,
            success=result.success,
            message_id=result.message_id,
        )
        return result.message_id if result.success else None

    # -- Confirmation & Cut-above --

    async def _show_confirmation(
        self,
        message: IncomingMessage,
        text: str,
        *,
        confirm_data: str,
        cancel_data: str,
    ) -> None:
        """Send a confirmation dialog with Yes/Cancel buttons."""
        from mai_gram.messenger.telegram import build_inline_keyboard

        kb = build_inline_keyboard([
            [("Yes", confirm_data), ("Cancel", cancel_data)],
        ])
        await self._messenger.send_message(
            OutgoingMessage(text=text, chat_id=message.chat_id, keyboard=kb)
        )

    async def _get_message_preview(
        self, db_message_id: int, max_len: int = 80
    ) -> str:
        """Fetch a truncated preview of a stored message by its DB id."""
        async with get_session() as session:
            result = await session.execute(
                select(Message).where(Message.id == db_message_id)
            )
            msg = result.scalar_one_or_none()
            if not msg or not msg.content:
                return ""
            text = msg.content.replace("\n", " ").strip()
            if len(text) > max_len:
                text = text[:max_len] + "..."
            return text

    async def _delete_callback_message(self, message: IncomingMessage) -> None:
        """Delete the message that contained the callback button."""
        if message.raw and hasattr(message.raw, "callback_query"):
            cb_msg = message.raw.callback_query.message
            if cb_msg:
                await self._messenger.delete_message(
                    message.chat_id, str(cb_msg.message_id)
                )

    async def _handle_cut_above(
        self, message: IncomingMessage, db_message_id: int
    ) -> None:
        """Set the cut-above point so messages before db_message_id are excluded from context."""
        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return
            chat.cut_above_message_id = db_message_id
            await session.commit()

        await self._messenger.send_message(
            OutgoingMessage(
                text="History cut applied. Older messages are hidden from AI but still searchable.",
                chat_id=message.chat_id,
            )
        )

    # -- Regenerate --

    async def _handle_regenerate(self, message: IncomingMessage) -> None:
        """Handle the regen callback: delete last assistant message, re-generate."""
        import html as _html

        if not message.callback_data:
            return

        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)

            recent = await message_store.get_recent(chat.id, limit=5)
            recent_sorted = sorted(recent, key=lambda m: m.id)

            to_delete: list[Message] = []
            for msg in reversed(recent_sorted):
                if msg.role in ("assistant", "tool"):
                    to_delete.append(msg)
                else:
                    break

            for msg in to_delete:
                await session.delete(msg)
            await session.flush()

        # Delete intermediate messages (tool calls, results, intermediate content)
        prev_msg_ids = self._response_message_ids.pop(message.chat_id, [])
        for mid in prev_msg_ids:
            await self._messenger.delete_message(message.chat_id, mid)

        # Delete the Telegram message with the regen button
        if message.raw and hasattr(message.raw, "callback_query"):
            cb_msg = message.raw.callback_query.message
            if cb_msg:
                await self._messenger.delete_message(
                    message.chat_id, str(cb_msg.message_id)
                )

        # Build a synthetic message to re-trigger the conversation pipeline
        # We need the last user message to reconstruct context
        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)
            wiki_store = WikiStore(session, data_dir=self._memory_data_dir)

            recent = await message_store.get_recent(chat.id, limit=1)
            if not recent or recent[0].role != "user":
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="Cannot regenerate: no user message found.",
                        chat_id=message.chat_id,
                    )
                )
                return

            await self._messenger.send_typing_indicator(message.chat_id)

            prompt_builder = PromptBuilder(
                self._llm,
                message_store,
                wiki_store,
                wiki_context_limit=self._wiki_context_limit,
                short_term_limit=self._short_term_limit,
                test_mode=self._test_mode,
            )

            enabled_tools, disabled_tools = self._settings.get_tool_filter()
            mcp_manager = MCPManager(
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
            )
            mcp_manager.register_server(
                "messages", MessagesMCPServer(message_store, chat.id),
            )
            mcp_manager.register_server(
                "wiki", WikiMCPServer(wiki_store, chat.id),
            )
            if self._external_mcp_pool is not None:
                for srv_name, srv in self._external_mcp_pool.get_all_servers().items():
                    mcp_manager.register_server(f"ext:{srv_name}", srv)

            regen_tz = chat.timezone
            llm_messages = await prompt_builder.build_context(
                chat,
                current_time=datetime.now(timezone.utc),
                send_datetime=chat.send_datetime,
                chat_timezone=chat.timezone,
                cut_above_message_id=chat.cut_above_message_id,
            )
            extra_params = self._settings.get_model_params(chat.llm_model)
            show_reasoning = chat.show_reasoning
            show_tool_calls = chat.show_tool_calls
            tg_chat_id = message.chat_id
            sent_msg_ids: list[str] = []

            async def _on_tool_call_display(
                *, content: str, tool_calls_json: str
            ) -> None:
                await message_store.save_message(
                    chat.id, "assistant", content or "",
                    tool_calls=tool_calls_json,
                    timezone_name=regen_tz,
                )

                if not show_tool_calls:
                    return
                try:
                    calls = json.loads(tool_calls_json)
                except (json.JSONDecodeError, TypeError):
                    return
                lines = []
                for tc in calls:
                    name = tc.get("name", "?")
                    args = tc.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args) if isinstance(args, str) else args
                        args_str = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        args_str = str(args)
                    lines.append(f"🔧 {name}({args_str})")
                if lines:
                    r = await self._messenger.send_message(
                        OutgoingMessage(text="\n".join(lines), chat_id=tg_chat_id)
                    )
                    if r.success and r.message_id:
                        sent_msg_ids.append(r.message_id)

            async def _on_tool_result_display(
                *,
                tool_call_id: str,
                tool_name: str,
                arguments: str,
                result: object,
                content: str,
                error: str | None,
                server_name: str | None,
            ) -> None:
                await message_store.save_message(
                    chat.id, "tool", content,
                    tool_call_id=tool_call_id,
                    timezone_name=regen_tz,
                )

                if not show_tool_calls:
                    return
                if error:
                    text = f"❌ {tool_name}: {error}"
                else:
                    result_str = str(result) if result is not None else ""
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "…"
                    text = f"✅ {tool_name}: {result_str}" if result_str else f"✅ {tool_name}"
                r = await self._messenger.send_message(
                    OutgoingMessage(text=text, chat_id=tg_chat_id)
                )
                if r.success and r.message_id:
                    sent_msg_ids.append(r.message_id)

            response_text: str | None = None
            response_reasoning: str | None = None
            placeholder_msg_id: str | None = None
            stream_usage = None
            stream_cost: float | None = None
            stream_is_byok = False
            try:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                last_edit_time = 0.0
                last_display_len = 0

                async for chunk in run_with_tools_stream(
                    self._llm,
                    mcp_manager,
                    llm_messages,
                    model=chat.llm_model,
                    max_iterations=self._tool_max_iterations,
                    extra_params=extra_params or None,
                    on_assistant_tool_call=_on_tool_call_display,
                    on_tool_result=_on_tool_result_display,
                ):
                    if chunk.usage is not None:
                        stream_usage = chunk.usage
                        stream_cost = chunk.cost
                        stream_is_byok = chunk.is_byok

                    if chunk.turn_complete:
                        if placeholder_msg_id:
                            turn_text = self._build_intermediate_display(
                                "".join(content_parts),
                                "".join(reasoning_parts),
                                show_reasoning,
                            )
                            if turn_text.strip():
                                await self._messenger.edit_message(
                                    tg_chat_id, placeholder_msg_id, turn_text,
                                )
                            sent_msg_ids.append(placeholder_msg_id)
                        content_parts.clear()
                        reasoning_parts.clear()
                        placeholder_msg_id = None
                        last_edit_time = 0.0
                        last_display_len = 0
                        continue

                    if chunk.reasoning:
                        reasoning_parts.append(chunk.reasoning)
                    if chunk.content:
                        content_parts.append(chunk.content)

                    current_reasoning = "".join(reasoning_parts)
                    current_content = "".join(content_parts)

                    display_text = ""
                    live_parse_mode: str | None = None
                    if show_reasoning and current_reasoning.strip():
                        from mai_gram.core.md_to_telegram import markdown_to_html

                        r_esc = _html.escape(current_reasoning.strip())
                        c_html = (
                            markdown_to_html(current_content)
                            if current_content.strip() else ""
                        )
                        display_text = (
                            f"<blockquote>\U0001f4ad Reasoning\n"
                            f"{r_esc}</blockquote>"
                        )
                        if c_html:
                            display_text += f"\n\n{c_html}"
                        live_parse_mode = "html"
                    elif current_content.strip():
                        from mai_gram.core.md_to_telegram import markdown_to_html

                        display_text = markdown_to_html(current_content)
                        live_parse_mode = "html"
                    else:
                        continue

                    display_len = len(current_reasoning) + len(current_content)
                    now_mono = time.monotonic()
                    chars_since_edit = display_len - last_display_len
                    time_since_edit = now_mono - last_edit_time

                    should_edit = (
                        display_text.strip()
                        and (time_since_edit >= 1.0 or chars_since_edit >= 60)
                        and chars_since_edit > 0
                    )

                    if should_edit:
                        live_text = display_text + " \u258d"
                        if placeholder_msg_id is None:
                            result = await self._messenger.send_message(
                                OutgoingMessage(
                                    text=live_text,
                                    chat_id=tg_chat_id,
                                    parse_mode=live_parse_mode,
                                )
                            )
                            if not result.success and live_parse_mode:
                                fallback = current_content or current_reasoning
                                result = await self._messenger.send_message(
                                    OutgoingMessage(
                                        text=fallback + " \u258d",
                                        chat_id=tg_chat_id,
                                    )
                                )
                            if result.success:
                                placeholder_msg_id = result.message_id
                        else:
                            edit_result = await self._messenger.edit_message(
                                tg_chat_id,
                                placeholder_msg_id,
                                live_text,
                                parse_mode=live_parse_mode,
                            )
                            if not edit_result.success and live_parse_mode:
                                fallback = current_content or current_reasoning
                                await self._messenger.edit_message(
                                    tg_chat_id,
                                    placeholder_msg_id,
                                    fallback + " \u258d",
                            )
                        last_edit_time = now_mono
                        last_display_len = display_len

                response_text = "".join(content_parts)
                response_reasoning = "".join(reasoning_parts) or None

                saved_msg_id: int | None = None
                if response_text and response_text.strip():
                    saved_msg = await message_store.save_message(
                        chat.id, "assistant", response_text,
                        timestamp=datetime.now(timezone.utc),
                        reasoning=response_reasoning,
                        timezone_name=chat.timezone,
                    )
                    saved_msg_id = saved_msg.id

            except Exception:
                logger.exception("Failed to regenerate response")
                response_text = "Error generating response. Please try again."
                placeholder_msg_id = None
                saved_msg_id = None

        self._response_message_ids[tg_chat_id] = sent_msg_ids

        from mai_gram.messenger.telegram import build_inline_keyboard

        kb_buttons = [[("\U0001f504 Regenerate", "regen")]]
        if saved_msg_id is not None:
            kb_buttons[0].append(("\u2702 Cut above", f"cut:{saved_msg_id}"))
        action_kb = build_inline_keyboard(kb_buttons)

        usage_footer = self._format_usage_footer(
            stream_usage, stream_cost, stream_is_byok
        )

        if placeholder_msg_id and response_text and response_text.strip():
            if show_reasoning and response_reasoning and response_reasoning.strip():
                from mai_gram.core.md_to_telegram import markdown_to_html

                escaped_r = _html.escape(response_reasoning.strip())
                html_body = markdown_to_html(response_text)
                footer_html = (
                    f"\n\n<i>{_html.escape(usage_footer)}</i>" if usage_footer else ""
                )
                display_text = (
                    f"<blockquote expandable>\U0001f4ad Reasoning\n"
                    f"{escaped_r}</blockquote>\n\n{html_body}{footer_html}"
                )
                await self._messenger.edit_message(
                    tg_chat_id, placeholder_msg_id, display_text,
                    parse_mode="html", keyboard=action_kb,
                )
            else:
                text_with_footer = response_text
                if usage_footer:
                    text_with_footer += f"\n\n{usage_footer}"
                await self._edit_with_mdv2_fallback(
                    tg_chat_id, placeholder_msg_id, text_with_footer,
                    keyboard=action_kb,
                )
            sent_msg_ids.append(placeholder_msg_id)
        elif response_text and response_text.strip():
            text_with_footer = response_text
            if usage_footer:
                text_with_footer += f"\n\n{usage_footer}"
            final_msg_id = await self._send_response(
                tg_chat_id,
                response_text=text_with_footer,
                response_reasoning=response_reasoning,
                show_reasoning=show_reasoning,
                keyboard=action_kb,
            )
            if final_msg_id:
                sent_msg_ids.append(final_msg_id)

    # -- DB helpers --

    async def _get_chat(self, session: object, chat_id: str) -> Chat | None:
        result = await session.execute(  # type: ignore[union-attr]
            select(Chat).where(Chat.id == chat_id)
        )
        return result.scalar_one_or_none()  # type: ignore[union-attr]
