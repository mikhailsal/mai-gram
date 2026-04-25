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

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.bot.callback_router import CallbackRouter
from mai_gram.bot.conversation_executor import AssistantTurnRequest, ConversationExecutor
from mai_gram.bot.history_actions import HistoryActions
from mai_gram.bot.import_workflow import ImportWorkflow
from mai_gram.bot.middleware import MessageLogger, RateLimitConfig, RateLimiter
from mai_gram.bot.regenerate_service import RegenerateService
from mai_gram.bot.resend_service import ResendService
from mai_gram.bot.reset_workflow import ResetWorkflow
from mai_gram.bot.setup_workflow import SetupSession, SetupWorkflow
from mai_gram.config import get_settings
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat
from mai_gram.llm.provider import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
)
from mai_gram.mcp_servers.manager import MCPManager
from mai_gram.mcp_servers.messages_server import MessagesMCPServer
from mai_gram.mcp_servers.wiki_server import WikiMCPServer
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import BotConfig
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
        self._conversation_executor = ConversationExecutor(
            messenger,
            llm_provider,
            tool_max_iterations=self._tool_max_iterations,
            renderer=self,
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
            llm_provider,
            self._conversation_executor,
            settings,
            resolve_chat_id=self._chat_id_for,
            build_mcp_manager=self._build_mcp_manager,
            memory_data_dir=self._memory_data_dir,
            wiki_context_limit=self._wiki_context_limit,
            short_term_limit=self._short_term_limit,
            test_mode=self._test_mode,
        )
        self._resend_service = ResendService(
            messenger,
            renderer=self,
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

            mcp_manager = self._build_mcp_manager(chat, message_store, wiki_store)

            now = datetime.now(timezone.utc)
            chat_tz = chat.timezone
            chat_send_dt = chat.send_datetime
            await message_store.save_message(
                chat.id,
                "user",
                message.text,
                timestamp=now,
                timezone_name=chat_tz,
                show_datetime=chat_send_dt,
            )

            llm_messages = await prompt_builder.build_context(
                chat,
                current_time=now,
                send_datetime=chat.send_datetime,
                chat_timezone=chat.timezone,
                cut_above_message_id=chat.cut_above_message_id,
            )

            result = await self._conversation_executor.execute(
                AssistantTurnRequest(
                    chat=chat,
                    message_store=message_store,
                    mcp_manager=mcp_manager,
                    llm_messages=llm_messages,
                    telegram_chat_id=message.chat_id,
                    timezone_name=chat_tz,
                    show_datetime=chat_send_dt,
                    show_reasoning=chat.show_reasoning,
                    show_tool_calls=chat.show_tool_calls,
                    extra_params=self._settings.get_model_params(chat.llm_model),
                    failure_log_message="Failed to generate response",
                )
            )
            self._response_message_ids[message.chat_id] = result.sent_message_ids

    @staticmethod
    def _build_intermediate_display(content: str, reasoning: str, show_reasoning: bool) -> str:
        """Build display text for an intermediate turn (before tool calls)."""
        display = ""
        if show_reasoning and reasoning.strip():
            display = f"💭 Reasoning:\n{reasoning.strip()}"
            if content.strip():
                display += "\n\n───\n\n" + content
        elif content.strip():
            display = content
        return display

    @staticmethod
    def _format_usage_footer(usage: object, cost: float | None, is_byok: bool) -> str:
        """Build a compact token/cost footer string."""
        del is_byok
        if usage is None:
            return ""
        prompt_t = getattr(usage, "prompt_tokens", 0)
        comp_t = getattr(usage, "completion_tokens", 0)
        parts = [f"{prompt_t}/{comp_t} tokens"]
        if cost is not None and cost > 0:
            parts.append(f"${cost:.4f}")
        return " | ".join(parts)

    async def _deliver_error(
        self,
        chat_id: str,
        error_text: str,
        *,
        placeholder_msg_id: str | None,
        keyboard: object = None,
        sent_msg_ids: list[str],
        max_attempts: int = 5,
    ) -> None:
        """Deliver an error message to the user with retry on failure."""
        for attempt in range(1, max_attempts + 1):
            if placeholder_msg_id:
                result = await self._messenger.edit_message(
                    chat_id,
                    placeholder_msg_id,
                    error_text,
                    keyboard=keyboard,
                )
                if result.success:
                    sent_msg_ids.append(placeholder_msg_id)
                    return
            else:
                result = await self._messenger.send_message(
                    OutgoingMessage(text=error_text, chat_id=chat_id, keyboard=keyboard)
                )
                if result.success and result.message_id:
                    sent_msg_ids.append(result.message_id)
                    return
            if attempt < max_attempts:
                delay = 2.0 * attempt
                logger.warning(
                    "Failed to deliver error (attempt %d/%d), retrying in %.0fs",
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
        logger.error("Could not deliver error message after %d attempts", max_attempts)

    @staticmethod
    def _user_friendly_error(exc: BaseException) -> str:
        """Map an exception to a user-facing error message with actionable advice."""
        if isinstance(exc, LLMAuthenticationError):
            return (
                "⚠️ API authentication failed.\n\n"
                "The bot's API key is invalid or expired. "
                "Please contact the bot administrator."
            )
        if isinstance(exc, LLMRateLimitError):
            msg = "⏳ Rate limit reached — the AI provider is temporarily overloaded."
            if exc.retry_after is not None:
                msg += f"\n\nPlease wait ~{int(exc.retry_after)}s and try again."
            else:
                msg += "\n\nPlease wait a moment and tap Regenerate."
            return msg
        if isinstance(exc, LLMModelNotFoundError):
            return (
                "❌ The selected model is no longer available.\n\n"
                "Use /reset and /start to choose a different model."
            )
        if isinstance(exc, LLMContextLengthError):
            return (
                "❌ The conversation is too long for this model's context window.\n\n"
                'Use "✂ Cut this & above" on a message to trim history, '
                "or /reset to start fresh."
            )
        if isinstance(exc, LLMProviderError):
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            return (
                f"⚠️ AI provider error{status}.\n\n"
                "This is usually temporary. Tap Regenerate to retry."
            )
        if isinstance(exc, LLMError):
            return "⚠️ Something went wrong with the AI provider.\n\nTap Regenerate to retry."
        return (
            "❌ Unexpected error while generating a response.\n\n"
            "Tap Regenerate to retry, or use /reset if the problem persists."
        )

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
            len(text),
            len(mdv2_text),
        )
        result = await self._messenger.edit_message(
            chat_id,
            message_id,
            mdv2_text,
            parse_mode="markdown",
            keyboard=keyboard,
        )
        if not result.success:
            logger.warning(
                "MarkdownV2 edit failed (error=%s), falling back to plain text",
                result.error,
            )
            await self._messenger.edit_message(
                chat_id,
                message_id,
                text,
                keyboard=keyboard,
            )

    async def _commit_overflow(
        self,
        *,
        tg_chat_id: str,
        header_html: str,
        reasoning_committed: bool,
        placeholder_msg_id: str | None,
        sent_msg_ids: list[str],
        remaining_content: str,
        current_content: str,
        committed_content_offset: int,
    ) -> tuple[int, str | None]:
        """Commit overflowing streamed content into finalized messages."""
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        if header_html and not reasoning_committed:
            if placeholder_msg_id:
                await self._edit_part(tg_chat_id, placeholder_msg_id, header_html)
                sent_msg_ids.append(placeholder_msg_id)
            else:
                mid = await self._send_part(tg_chat_id, header_html)
                if mid:
                    sent_msg_ids.append(mid)
            placeholder_msg_id = None

        while len(remaining_content) > 0:
            chunk_text = remaining_content[: SAFE_MAX_LENGTH - 200]
            para_break = chunk_text.rfind("\n\n")
            if para_break > len(chunk_text) // 3:
                chunk_text = chunk_text[:para_break]
            elif (nl := chunk_text.rfind("\n")) > len(chunk_text) // 3:
                chunk_text = chunk_text[:nl]

            chunk_html = markdown_to_html(chunk_text)
            if len(chunk_html) > SAFE_MAX_LENGTH:
                chunk_html = chunk_html[: SAFE_MAX_LENGTH - 10]

            if placeholder_msg_id:
                await self._edit_part(tg_chat_id, placeholder_msg_id, chunk_html)
                sent_msg_ids.append(placeholder_msg_id)
                placeholder_msg_id = None
            else:
                mid = await self._send_part(tg_chat_id, chunk_html)
                if mid:
                    sent_msg_ids.append(mid)

            committed_content_offset += len(chunk_text)
            remaining_content = current_content[committed_content_offset:]

            if len(remaining_content) <= SAFE_MAX_LENGTH - 200:
                break

        new_placeholder: str | None = None
        if remaining_content.strip():
            c_html = markdown_to_html(remaining_content) + " ▍"
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=c_html,
                    chat_id=tg_chat_id,
                    parse_mode="html",
                )
            )
            if result.success:
                new_placeholder = result.message_id
        return committed_content_offset, new_placeholder

    async def _send_part(self, chat_id: str, text: str, *, keyboard: object = None) -> str | None:
        """Send a single message part, falling back to plain text if HTML fails."""
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        result = await self._messenger.send_message(
            OutgoingMessage(text=text, chat_id=chat_id, parse_mode="html", keyboard=keyboard)
        )
        if result.success:
            return result.message_id
        error = (result.error or "").lower()
        if "too long" in error or "message is too long" in error:
            return await self._send_part_split(chat_id, text, keyboard=keyboard)
        if "parse entities" in error or "can't find end tag" in error:
            import re

            plain = re.sub(r"<[^>]+>", "", text)
            if len(plain) > SAFE_MAX_LENGTH:
                return await self._send_part_split(chat_id, text, keyboard=keyboard)
            result = await self._messenger.send_message(
                OutgoingMessage(text=plain, chat_id=chat_id, keyboard=keyboard)
            )
            if result.success:
                return result.message_id
        return None

    async def _send_part_split(
        self, chat_id: str, text: str, *, keyboard: object = None
    ) -> str | None:
        """Emergency split: the text exceeded the limit even after our split."""
        import re

        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        plain = re.sub(r"<[^>]+>", "", text)
        chunks = split_html_safe(plain, max_len=SAFE_MAX_LENGTH)
        last_id: str | None = None
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=chunk,
                    chat_id=chat_id,
                    keyboard=keyboard if is_last else None,
                )
            )
            if result.success and result.message_id:
                last_id = result.message_id
        return last_id

    async def _send_long_message(
        self,
        chat_id: str,
        raw_text: str,
        *,
        header_html: str = "",
        keyboard: object = None,
    ) -> list[str]:
        """Send a message, splitting the raw markdown BEFORE HTML conversion.

        Each chunk is independently converted to HTML so split boundaries
        never break HTML tags. If ``header_html`` is small enough it is
        prepended to the first part; otherwise it is sent as a separate
        message. The keyboard is attached only to the last part.
        """
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        sent_ids: list[str] = []
        header_sent = False

        max_first = SAFE_MAX_LENGTH // 2 if header_html else SAFE_MAX_LENGTH
        raw_parts = split_html_safe(raw_text, max_len=max_first)
        if len(raw_parts) > 1:
            rest = split_html_safe("".join(raw_parts[1:]), max_len=SAFE_MAX_LENGTH)
            raw_parts = [raw_parts[0]] + rest

        for index, raw_part in enumerate(raw_parts):
            is_last = index == len(raw_parts) - 1
            html_part = markdown_to_html(raw_part)

            if index == 0 and header_html and not header_sent:
                combined = header_html + "\n\n" + html_part
                if len(combined) <= SAFE_MAX_LENGTH:
                    html_part = combined
                    header_sent = True
                else:
                    header_id = await self._send_part(chat_id, header_html)
                    if header_id:
                        sent_ids.append(header_id)
                    header_sent = True

            msg_id = await self._send_part(
                chat_id,
                html_part,
                keyboard=keyboard if is_last else None,
            )
            if msg_id:
                sent_ids.append(msg_id)
        return sent_ids

    async def _edit_part(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        keyboard: object = None,
    ) -> bool:
        """Edit a message, falling back to plain text if HTML fails."""
        result = await self._messenger.edit_message(
            chat_id, message_id, text, parse_mode="html", keyboard=keyboard
        )
        if result.success:
            return True
        error = (result.error or "").lower()
        if "parse entities" in error or "can't find end tag" in error:
            import re

            plain = re.sub(r"<[^>]+>", "", text)
            result = await self._messenger.edit_message(
                chat_id, message_id, plain, keyboard=keyboard
            )
            return result.success
        return False

    async def _finalize_placeholder(
        self,
        chat_id: str,
        placeholder_msg_id: str,
        raw_text: str,
        *,
        header_html: str = "",
        keyboard: object = None,
    ) -> list[str]:
        """Edit the placeholder with first chunk, send rest as new messages.

        Splits raw markdown BEFORE HTML conversion to keep tags intact.
        Returns extra message IDs (placeholder ID is NOT included).
        """
        from mai_gram.core.md_to_telegram import markdown_to_html
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH, split_html_safe

        max_first = SAFE_MAX_LENGTH // 2 if header_html else SAFE_MAX_LENGTH
        raw_parts = split_html_safe(raw_text, max_len=max_first)
        if len(raw_parts) > 1:
            rest = split_html_safe("".join(raw_parts[1:]), max_len=SAFE_MAX_LENGTH)
            raw_parts = [raw_parts[0]] + rest

        first_html = markdown_to_html(raw_parts[0])
        if header_html:
            combined = header_html + "\n\n" + first_html
            if len(combined) <= SAFE_MAX_LENGTH:
                first_html = combined
            else:
                await self._edit_part(chat_id, placeholder_msg_id, header_html)
                initial_extra_ids: list[str] = []
                for index, raw_part in enumerate(raw_parts):
                    is_last = index == len(raw_parts) - 1
                    html_part = markdown_to_html(raw_part)
                    msg_id = await self._send_part(
                        chat_id,
                        html_part,
                        keyboard=keyboard if is_last else None,
                    )
                    if msg_id:
                        initial_extra_ids.append(msg_id)
                return initial_extra_ids

        if len(raw_parts) == 1:
            await self._edit_part(chat_id, placeholder_msg_id, first_html, keyboard=keyboard)
            return []

        await self._edit_part(chat_id, placeholder_msg_id, first_html)

        extra_ids: list[str] = []
        for index, raw_part in enumerate(raw_parts[1:], start=1):
            is_last = index == len(raw_parts) - 1
            html_part = markdown_to_html(raw_part)
            msg_id = await self._send_part(
                chat_id,
                html_part,
                keyboard=keyboard if is_last else None,
            )
            if msg_id:
                extra_ids.append(msg_id)
        return extra_ids

    async def _send_response(
        self,
        chat_id: str,
        *,
        response_text: str | None,
        response_reasoning: str | None = None,
        show_reasoning: bool = False,
        keyboard: object = None,
    ) -> list[str]:
        """Send the final assistant response, splitting if needed.

        Returns a list of Telegram message IDs for all sent parts.
        """
        if not response_text or not response_text.strip():
            return []

        from mai_gram.core.md_to_telegram import format_reasoning_html

        header_html = ""
        if show_reasoning and response_reasoning and response_reasoning.strip():
            header_html = format_reasoning_html(response_reasoning, expandable=True)

        sent_ids = await self._send_long_message(
            chat_id,
            response_text,
            header_html=header_html,
            keyboard=keyboard,
        )
        self._message_logger.log_outgoing(
            chat_id,
            response_text,
            success=bool(sent_ids),
            message_id=sent_ids[-1] if sent_ids else None,
        )
        return sent_ids

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
