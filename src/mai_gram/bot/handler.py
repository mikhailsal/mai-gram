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

from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
from mai_gram.bot.callback_router import CallbackRouter
from mai_gram.bot.conversation_executor import ConversationExecutor
from mai_gram.bot.conversation_service import ConversationService
from mai_gram.bot.history_actions import HistoryActions
from mai_gram.bot.import_workflow import ImportWorkflow
from mai_gram.bot.middleware import MessageLogger, RateLimitConfig, RateLimiter
from mai_gram.bot.regenerate_service import RegenerateService
from mai_gram.bot.resend_service import ResendService
from mai_gram.bot.reset_workflow import ResetWorkflow
from mai_gram.bot.response_renderer import ResponseRenderer
from mai_gram.bot.setup_workflow import SetupSession, SetupWorkflow
from mai_gram.config import get_settings
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.mcp_servers.wiki_server import WikiMCPServer
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import BotConfig
    from mai_gram.llm.provider import LLMProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore
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
        self._llm = llm_provider
        self._rate_limiter = RateLimiter(
            rate_limit_config,
            on_rate_limited=self._handle_rate_limited,
        )
        self._message_logger = MessageLogger(log_content=False)
        self._test_mode = test_mode
        self._response_message_ids: dict[str, list[str]] = {}
        self._cut_original_html: dict[str, tuple[str, str | None]] = {}

        settings = get_settings()
        self._memory_data_dir = memory_data_dir or settings.memory_data_dir
        self._wiki_context_limit = wiki_context_limit or settings.wiki_context_limit
        self._short_term_limit = short_term_limit or settings.short_term_limit
        self._tool_max_iterations = tool_max_iterations or settings.tool_max_iterations
        self._settings = settings
        self._response_renderer = ResponseRenderer(
            messenger,
            message_logger=self._message_logger,
        )
        self._assistant_turn_builder = AssistantTurnBuilder(
            llm_provider,
            settings,
            build_mcp_manager=self._build_mcp_manager,
            memory_data_dir=self._memory_data_dir,
            wiki_context_limit=self._wiki_context_limit,
            short_term_limit=self._short_term_limit,
            test_mode=self._test_mode,
        )
        self._conversation_executor = ConversationExecutor(
            messenger,
            llm_provider,
            tool_max_iterations=self._tool_max_iterations,
            renderer=self._response_renderer,
        )
        self._conversation_service = ConversationService(
            messenger,
            self._conversation_executor,
            turn_builder=self._assistant_turn_builder,
            resolve_chat_id=self._chat_id_for,
        )
        self._import_workflow = ImportWorkflow(
            messenger,
            settings,
            get_allowed_models=self._get_allowed_models_for_bot,
            resolve_chat_id=self._chat_id_for,
        )
        self._history_actions = HistoryActions(
            messenger,
            resolve_chat_id=self._chat_id_for,
        )
        self._regenerate_service = RegenerateService(
            messenger,
            self._conversation_executor,
            turn_builder=self._assistant_turn_builder,
            resolve_chat_id=self._chat_id_for,
        )
        self._resend_service = ResendService(
            messenger,
            renderer=self._response_renderer,
            resolve_chat_id=self._chat_id_for,
        )
        self._reset_workflow = ResetWorkflow(
            messenger,
            presenter=self,
            resolve_chat_id=self._chat_id_for,
            clear_setup_session=self.clear_setup_session,
            memory_data_dir=self._memory_data_dir,
        )
        self._setup_workflow = SetupWorkflow(
            messenger,
            settings,
            bot_config=bot_config,
            resolve_chat_id=self._chat_id_for,
        )
        self._callback_router = CallbackRouter(
            messenger,
            import_workflow=self._import_workflow,
            setup_workflow=self._setup_workflow,
            reset_workflow=self._reset_workflow,
            history_actions=self._history_actions,
            regenerate_service=self._regenerate_service,
            show_confirmation=self._show_confirmation,
            delete_callback_message=self._delete_callback_message,
            cut_original_html=self._cut_original_html,
            response_message_ids=self._response_message_ids,
        )
        self._bot_config = bot_config
        self._external_mcp_pool = external_mcp_pool

        # Per-bot user whitelist takes precedence over the global ALLOWED_USERS
        if bot_config and bot_config.allowed_users is not None:
            self._allowed_users = {str(uid) for uid in bot_config.allowed_users}
        else:
            self._allowed_users = settings.get_allowed_user_ids()

        if self._allowed_users:
            logger.info(
                "Access control enabled: %d user(s) allowed",
                len(self._allowed_users),
            )

        messenger.register_command_handler(
            "start",
            self._handle_start,
            description="Set up a new chat",
        )
        messenger.register_command_handler(
            "reset",
            self._handle_reset,
            description="Delete chat and history",
        )
        messenger.register_command_handler(
            "model",
            self._handle_model,
            description="Show current model",
        )
        messenger.register_command_handler(
            "help",
            self._handle_help,
            description="Show available commands",
        )
        messenger.register_command_handler(
            "datetime",
            self._handle_datetime_toggle,
            description="Toggle date/time in messages",
        )
        messenger.register_command_handler(
            "timezone",
            self._handle_timezone,
            description="Set timezone (e.g. /timezone Europe/Moscow)",
        )
        messenger.register_command_handler(
            "reasoning",
            self._handle_reasoning_toggle,
            description="Toggle reasoning display",
        )
        messenger.register_command_handler(
            "toolcalls",
            self._handle_toolcalls_toggle,
            description="Toggle tool call display",
        )
        messenger.register_command_handler(
            "import",
            self._handle_import,
            description="Import conversation from JSON file",
        )
        messenger.register_command_handler(
            "resend_last",
            self._handle_resend_last,
            description="Re-send last AI message (if truncated)",
        )
        messenger.register_message_handler(self._handle_message)
        messenger.register_callback_handler(self._handle_callback)
        messenger.register_document_handler(self._handle_document)

    # -- Setup session helpers --

    def is_in_setup(self, user_id: str) -> bool:
        return self._setup_workflow.is_in_setup(user_id)

    def get_setup_session(self, user_id: str) -> SetupSession | None:
        return self._setup_workflow.get_setup_session(user_id)

    def clear_setup_session(self, user_id: str) -> None:
        self._setup_workflow.clear_setup_session(user_id)

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
                text=(f"Access denied. This is a private bot. Your user ID: {message.user_id}"),
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
        await self._setup_workflow.handle_start(message)

    async def _handle_reset(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return
        await self._reset_workflow.handle_reset(message)

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

    # -- Resend last --

    async def _handle_resend_last(self, message: IncomingMessage) -> None:
        """Re-send the last assistant message from DB (handles truncation)."""
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
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
        if not await self._check_access(message):
            return
        await self._import_workflow.handle_import(
            message,
            in_setup=self.is_in_setup(message.user_id),
        )

    async def _handle_document(self, message: IncomingMessage) -> None:
        """Handle uploaded documents (JSON files for import)."""
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return
        await self._import_workflow.handle_document(message)

    # -- Message handler --

    async def _handle_message(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return
        if not await self._rate_limiter.check_rate_limit(message.user_id, message.chat_id):
            return

        if self.is_in_setup(message.user_id):
            await self._setup_workflow.handle_setup_text(message)
            return

        await self._handle_conversation(message)

    async def _handle_callback(self, message: IncomingMessage) -> None:
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return
        await self._callback_router.handle_callback(message)

    def _get_allowed_models_for_bot(self) -> list[str]:
        """Return the model list for this bot, respecting per-bot restrictions."""
        global_models = self._settings.get_allowed_models()
        if self._bot_config and self._bot_config.allowed_models:
            bot_set = set(self._bot_config.allowed_models)
            return [m for m in global_models if m in bot_set]
        return global_models

    # -- MCP manager builder --

    def _build_mcp_manager(
        self,
        chat: Chat,
        message_store: MessageStore,
        wiki_store: WikiStore,
    ) -> MCPManager:
        """Build an MCPManager with tool/server filters from global and per-prompt config."""
        from mai_gram.config import PromptConfig

        global_enabled, global_disabled = self._settings.get_tool_filter()
        prompt_cfg: PromptConfig | None = None
        if chat.prompt_name:
            prompt_cfg = self._settings.get_prompt_config(chat.prompt_name)

        enabled_tools = global_enabled
        disabled_tools = global_disabled
        if prompt_cfg is not None and (
            prompt_cfg.tools_enabled is not None or prompt_cfg.tools_disabled is not None
        ):
            enabled_tools = prompt_cfg.tools_enabled
            disabled_tools = prompt_cfg.tools_disabled

        mcp_manager = MCPManager(
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
        )

        mcp_servers_enabled = prompt_cfg.mcp_servers_enabled if prompt_cfg else None
        mcp_servers_disabled = prompt_cfg.mcp_servers_disabled if prompt_cfg else None

        def _is_server_allowed(name: str) -> bool:
            if mcp_servers_enabled is not None:
                return name in mcp_servers_enabled
            if mcp_servers_disabled is not None:
                return name not in mcp_servers_disabled
            return True

        if _is_server_allowed("messages"):
            mcp_manager.register_server(
                "messages",
                MessagesMCPServer(message_store, chat.id),
            )
        if _is_server_allowed("wiki"):
            mcp_manager.register_server(
                "wiki",
                WikiMCPServer(wiki_store, chat.id),
            )
        if self._external_mcp_pool is not None:
            for srv_name, srv in self._external_mcp_pool.get_all_servers().items():
                if _is_server_allowed(srv_name):
                    mcp_manager.register_server(f"ext:{srv_name}", srv)

        return mcp_manager

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
        from mai_gram.messenger.telegram import build_inline_keyboard

        kb = build_inline_keyboard(
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

    async def _delete_callback_message(self, message: IncomingMessage) -> None:
        """Delete the message that contained the callback button."""
        if message.raw and hasattr(message.raw, "callback_query"):
            cb_msg = message.raw.callback_query.message
            if cb_msg:
                await self._messenger.delete_message(message.chat_id, str(cb_msg.message_id))

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
