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

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.access_control import AccessControl
from mai_gram.bot.handler_services import build_handler_services
from mai_gram.bot.middleware import MessageLogger, RateLimitConfig, RateLimiter
from mai_gram.config import get_settings
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.bot.setup_workflow import SetupSession
    from mai_gram.config import BotConfig, Settings
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


def make_chat_id(user_id: str, bot_id: str) -> str:
    """Create a composite chat ID from user_id and bot_id.

    Format: ``{user_id}@{bot_id}``
    """
    return f"{user_id}@{bot_id}"


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
        external_mcp_pool: ExternalMCPPool | None = None,
        bot_config: BotConfig | None = None,
    ) -> None:
        self._messenger = messenger
        self._message_logger = MessageLogger(log_content=False)
        self._test_mode = test_mode
        self._response_message_ids: dict[str, list[str]] = {}
        self._cut_original_html: dict[str, tuple[str, str | None]] = {}

        settings = get_settings()
        self._configure_runtime_settings(
            settings,
            memory_data_dir=memory_data_dir,
            wiki_context_limit=wiki_context_limit,
            short_term_limit=short_term_limit,
            tool_max_iterations=tool_max_iterations,
        )
        self._build_services(
            llm_provider,
            settings=settings,
            external_mcp_pool=external_mcp_pool,
            bot_config=bot_config,
        )
        self._bot_config = bot_config
        allowed_users = self._allowed_users(settings, bot_config=bot_config)
        self._access_control = AccessControl(self._messenger, allowed_users=allowed_users)
        self._rate_limiter = RateLimiter(
            rate_limit_config,
            on_rate_limited=self._access_control.handle_rate_limited,
        )
        self._register_handlers()

    def _configure_runtime_settings(
        self,
        settings: Settings,
        *,
        memory_data_dir: str | None,
        wiki_context_limit: int | None,
        short_term_limit: int | None,
        tool_max_iterations: int | None,
    ) -> None:
        self._memory_data_dir = memory_data_dir or settings.memory_data_dir
        self._wiki_context_limit = wiki_context_limit or settings.wiki_context_limit
        self._short_term_limit = short_term_limit or settings.short_term_limit
        self._tool_max_iterations = tool_max_iterations or settings.tool_max_iterations
        self._settings = settings

    def _build_services(
        self,
        llm_provider: LLMProvider,
        *,
        settings: Settings,
        external_mcp_pool: ExternalMCPPool | None,
        bot_config: BotConfig | None,
    ) -> None:
        services = build_handler_services(
            self._messenger,
            llm_provider,
            settings=settings,
            message_logger=self._message_logger,
            presenter=self,
            resolve_chat_id=self._chat_id_for,
            clear_setup_session=self.clear_setup_session,
            show_confirmation=self._show_confirmation,
            cut_original_html=self._cut_original_html,
            response_message_ids=self._response_message_ids,
            memory_data_dir=self._memory_data_dir,
            wiki_context_limit=self._wiki_context_limit,
            short_term_limit=self._short_term_limit,
            tool_max_iterations=self._tool_max_iterations,
            test_mode=self._test_mode,
            bot_config=bot_config,
            external_mcp_pool=external_mcp_pool,
        )
        self._response_renderer = services.response_renderer
        self._mcp_manager_factory = services.mcp_manager_factory
        self._assistant_turn_builder = services.assistant_turn_builder
        self._conversation_executor = services.conversation_executor
        self._conversation_service = services.conversation_service
        self._import_workflow = services.import_workflow
        self._history_actions = services.history_actions
        self._regenerate_service = services.regenerate_service
        self._resend_service = services.resend_service
        self._reset_workflow = services.reset_workflow
        self._setup_workflow = services.setup_workflow
        self._callback_router = services.callback_router

    def _allowed_users(
        self,
        settings: Settings,
        *,
        bot_config: BotConfig | None,
    ) -> set[str]:
        # Per-bot user whitelist takes precedence over the global ALLOWED_USERS
        if bot_config and bot_config.allowed_users is not None:
            return {str(uid) for uid in bot_config.allowed_users}
        return settings.get_allowed_user_ids()

    def _register_handlers(self) -> None:
        for name, handler, description in (
            ("start", self._handle_start, "Set up a new chat"),
            ("reset", self._handle_reset, "Delete chat and history"),
            ("model", self._handle_model, "Show current model"),
            ("help", self._handle_help, "Show available commands"),
            ("datetime", self._handle_datetime_toggle, "Toggle date/time in messages"),
            ("timezone", self._handle_timezone, "Set timezone (e.g. /timezone Europe/Moscow)"),
            ("reasoning", self._handle_reasoning_toggle, "Toggle reasoning display"),
            ("toolcalls", self._handle_toolcalls_toggle, "Toggle tool call display"),
            ("import", self._handle_import, "Import conversation from JSON file"),
            (
                "resend_last",
                self._handle_resend_last,
                "Re-send last AI message (if truncated)",
            ),
        ):
            self._messenger.register_command_handler(
                name,
                handler,
                description=description,
            )
        self._messenger.register_message_handler(self._handle_message)
        self._messenger.register_callback_handler(self._handle_callback)
        self._messenger.register_document_handler(self._handle_document)

    # -- Setup session helpers --

    def is_in_setup(self, user_id: str) -> bool:
        return self._setup_workflow.is_in_setup(user_id)

    def get_setup_session(self, user_id: str) -> SetupSession | None:
        return self._setup_workflow.get_setup_session(user_id)

    def clear_setup_session(self, user_id: str) -> None:
        self._setup_workflow.clear_setup_session(user_id)

    def _chat_id_for(self, message: IncomingMessage) -> str:
        if message.bot_id:
            return make_chat_id(message.user_id, message.bot_id)
        return message.chat_id

    # -- Command handlers --

    async def _handle_start(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        await self._setup_workflow.handle_start(message)

    async def _handle_reset(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        await self._reset_workflow.handle_reset(message)

    async def _handle_model(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
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
        if not await self._access_control.check_access(message):
            return

        msg = (
            "Available commands:\n\n"
            "/start - Set up a new chat (choose model + prompt)\n"
            "/import - Import conversation from JSON file\n"
            "/reset - Delete current chat and history\n"
            "/model - Show current model\n"
            "/timezone - Set timezone (e.g. /timezone Europe/Moscow)\n"
            "/datetime - Toggle date/time in messages sent to LLM\n"
            "/reasoning - Toggle display of LLM reasoning\n"
            "/toolcalls - Toggle display of tool call details\n"
            "/resend_last - Re-send last AI message (if truncated)\n"
            "/help - Show this help message\n\n"
            "Just send a message to chat!"
        )
        await self._messenger.send_message(OutgoingMessage(text=msg, chat_id=message.chat_id))

    async def _toggle_chat_flag(
        self, message: IncomingMessage, field_name: str, label: str
    ) -> None:
        """Generic toggle for boolean chat settings."""
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
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
        if not await self._access_control.check_access(message):
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

    # -- Resend last --

    async def _handle_resend_last(self, message: IncomingMessage) -> None:
        """Re-send the last assistant message from DB (handles truncation)."""
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return

        result = await self._resend_service.handle_resend(
            message,
            previous_response_ids=self._response_message_ids.get(message.chat_id, []),
        )
        if result.replaced_previous:
            self._response_message_ids[message.chat_id] = result.sent_message_ids

    # -- Import command --

    async def _handle_import(self, message: IncomingMessage) -> None:
        """Handle /import command -- start the import flow."""
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        await self._import_workflow.handle_import(
            message,
            in_setup=self.is_in_setup(message.user_id),
        )

    async def _handle_document(self, message: IncomingMessage) -> None:
        """Handle uploaded documents (JSON files for import)."""
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        await self._import_workflow.handle_document(message)

    # -- Message handler --

    async def _handle_message(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        if not await self._rate_limiter.check_rate_limit(message.user_id, message.chat_id):
            return

        if self.is_in_setup(message.user_id):
            await self._setup_workflow.handle_setup_text(message)
            return

        await self._handle_conversation(message)

    async def _handle_callback(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._access_control.check_access(message):
            return
        await self._callback_router.handle_callback(message)

    # -- Conversation --

    async def _handle_conversation(self, message: IncomingMessage) -> None:
        sent_message_ids = await self._conversation_service.handle_message(message)
        self._response_message_ids[message.chat_id] = sent_message_ids

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
        kb = self._messenger.build_inline_keyboard(
            [
                [("Yes", confirm_data), ("Cancel", cancel_data)],
            ]
        )
        await self._messenger.send_message(
            OutgoingMessage(text=text, chat_id=message.chat_id, keyboard=kb)
        )

    async def _get_message_preview(self, db_message_id: int, max_len: int = 80) -> str:
        """Fetch a truncated preview of a stored message by its DB id."""
        return await self._history_actions.get_message_preview(db_message_id, max_len=max_len)

    async def _handle_cut_above(
        self,
        message: IncomingMessage,
        db_message_id: int,
        *,
        original_tg_msg_id: str = "",
    ) -> None:
        """Set the cut-above point so the target message and all before it are excluded."""
        cached_original = None
        if original_tg_msg_id:
            cache_key = f"{message.chat_id}:{original_tg_msg_id}"
            cached_original = self._cut_original_html.pop(cache_key, None)

        await self._history_actions.handle_cut_above(
            message,
            db_message_id,
            original_tg_msg_id=original_tg_msg_id,
            cached_original=cached_original,
        )

    # -- Regenerate --

    async def _handle_regenerate(self, message: IncomingMessage) -> None:
        """Handle the regen callback: delete last assistant message, re-generate."""
        sent_message_ids = await self._regenerate_service.handle_regenerate(
            message,
            previous_response_ids=self._response_message_ids.get(message.chat_id, []),
        )
        self._response_message_ids[message.chat_id] = sent_message_ids

    # -- DB helpers --

    async def _get_chat(self, session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
