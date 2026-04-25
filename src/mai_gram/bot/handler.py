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
import enum
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from mai_gram.bot.conversation_executor import AssistantTurnRequest, ConversationExecutor
from mai_gram.bot.middleware import MessageLogger, RateLimitConfig, RateLimiter
from mai_gram.config import get_settings
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
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


class ImportState(str, enum.Enum):
    """States in the import flow."""

    CHOOSING_MODEL = "choosing_model"
    AWAITING_FILE = "awaiting_file"


@dataclass
class ImportSession:
    """Tracks the state of an ongoing import session."""

    user_id: str
    chat_id: str
    state: ImportState = ImportState.CHOOSING_MODEL
    selected_model: str = ""
    created_at: float = 0.0


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
        self._setup_sessions: dict[str, SetupSession] = {}
        self._import_sessions: dict[str, ImportSession] = {}
        self._import_replay_tasks: dict[str, asyncio.Task[None]] = {}
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
        return user_id in self._setup_sessions

    def get_setup_session(self, user_id: str) -> SetupSession | None:
        return self._setup_sessions.get(user_id)

    def clear_setup_session(self, user_id: str) -> None:
        self._setup_sessions.pop(user_id, None)

    # -- Import session helpers --

    IMPORT_SESSION_TIMEOUT_SECONDS = 300  # 5 minutes

    def is_in_import(self, user_id: str) -> bool:
        session = self._import_sessions.get(user_id)
        if session is None:
            return False
        if time.monotonic() - session.created_at > self.IMPORT_SESSION_TIMEOUT_SECONDS:
            self._import_sessions.pop(user_id, None)
            return False
        return True

    def get_import_session(self, user_id: str) -> ImportSession | None:
        if not self.is_in_import(user_id):
            return None
        return self._import_sessions.get(user_id)

    def clear_import_session(self, user_id: str) -> None:
        self._import_sessions.pop(user_id, None)

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

        if not chat:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="No chat to reset. Use /start to create one.",
                    chat_id=message.chat_id,
                )
            )
            return

        msg_count = 0
        async with get_session() as session:
            result = await session.execute(
                select(func.count(Message.id)).where(Message.chat_id == chat_id)
            )
            msg_count = result.scalar() or 0

        confirm_text = (
            "⚠️ Reset this chat?\n\n"
            f"Model: {chat.llm_model}\n"
            f"Messages: {msg_count}\n\n"
            "All history and wiki entries will be deleted.\n"
            "A backup archive will be created before deletion."
        )
        await self._show_confirmation(
            message,
            confirm_text,
            confirm_data=f"confirm_reset:{chat_id}",
            cancel_data="cancel_action",
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

        chat_id = self._chat_id_for(message)
        tg_chat_id = message.chat_id

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No chat configured. Use /start first.",
                        chat_id=tg_chat_id,
                    )
                )
                return

            result = await session.execute(
                select(Message)
                .where(
                    Message.chat_id == chat_id,
                    Message.role == "assistant",
                    Message.content.isnot(None),
                    Message.content != "",
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            last_msg = result.scalar_one_or_none()

            if not last_msg:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="No assistant message found to resend.",
                        chat_id=tg_chat_id,
                    )
                )
                return

        old_tg_ids = self._response_message_ids.pop(tg_chat_id, [])
        for old_id in old_tg_ids:
            await self._messenger.delete_message(tg_chat_id, old_id)

        from mai_gram.messenger.telegram import build_inline_keyboard

        kb_buttons = [[("🔄 Regenerate", "regen")]]
        kb_buttons[0].append(("✂ Cut this & above", f"cut:{last_msg.id}"))
        action_kb = build_inline_keyboard(kb_buttons)

        show_reasoning = chat.show_reasoning
        reasoning = last_msg.reasoning if last_msg.reasoning else None

        sent_ids = await self._send_response(
            tg_chat_id,
            response_text=last_msg.content,
            response_reasoning=reasoning,
            show_reasoning=show_reasoning,
            keyboard=action_kb,
        )
        self._response_message_ids[tg_chat_id] = sent_ids

        if sent_ids:
            await self._messenger.send_message(
                OutgoingMessage(
                    text=f"✅ Resent last AI message ({len(sent_ids)} part(s)).",
                    chat_id=tg_chat_id,
                )
            )

    # -- Import command --

    async def _handle_import(self, message: IncomingMessage) -> None:
        """Handle /import command -- start the import flow."""
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        user_id = message.user_id

        if self.is_in_setup(user_id):
            await self._messenger.send_message(
                OutgoingMessage(
                    text="You are in the middle of a setup. Finish it first or use /reset.",
                    chat_id=message.chat_id,
                )
            )
            return

        if self.is_in_import(user_id):
            self.clear_import_session(user_id)
            logger.info("Cleared stale import session for user %s", user_id)

        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)

        if chat:
            await self._messenger.send_message(
                OutgoingMessage(
                    text=(
                        f"A chat already exists for this bot (model: {chat.llm_model}).\n"
                        "Importing would conflict with existing history.\n\n"
                        "Use /reset first to clear the existing chat, then try /import again."
                    ),
                    chat_id=message.chat_id,
                )
            )
            return

        import_session = ImportSession(
            user_id=user_id,
            chat_id=message.chat_id,
            created_at=time.monotonic(),
        )
        self._import_sessions[user_id] = import_session
        await self._show_import_model_selection(import_session)

    async def _show_import_model_selection(self, session: ImportSession) -> None:
        """Show model selection keyboard for the import flow."""
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = ImportState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._get_allowed_models_for_bot()
        default_model = self._settings.get_default_model()
        for model in allowed_models:
            short_name = model.split("/")[-1] if "/" in model else model
            label = f"{short_name} [default]" if model == default_model else short_name
            keyboard_rows.append([(label, f"import_model:{model}")])

        kb = build_inline_keyboard(keyboard_rows)

        await self._messenger.send_message(
            OutgoingMessage(
                text=("📥 Import Mode\n\nChoose the LLM model for the imported conversation:"),
                chat_id=session.chat_id,
                keyboard=kb,
            )
        )

    async def _handle_import_callback(self, message: IncomingMessage) -> None:
        """Handle callback queries during the import flow."""
        session = self._import_sessions.get(message.user_id)
        if not session or not message.callback_data:
            return

        data = message.callback_data
        if not data.startswith("import_model:"):
            return

        model = data.split(":", 1)[1]

        allowed = self._get_allowed_models_for_bot()
        if allowed and model not in allowed:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="This model is not available. Please choose another.",
                    chat_id=session.chat_id,
                )
            )
            return

        session.selected_model = model
        session.state = ImportState.AWAITING_FILE

        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    f"Model: {model}\n\n"
                    "Now upload your JSON file.\n\n"
                    "Supported formats:\n"
                    '1. Array of messages: [{"role": "user", "content": "..."}, ...]\n'
                    "2. AI Proxy v2 request JSON\n\n"
                    "The system prompt from the file will be preserved."
                ),
                chat_id=session.chat_id,
            )
        )

    async def _handle_document(self, message: IncomingMessage) -> None:
        """Handle uploaded documents (JSON files for import)."""
        self._message_logger.log_incoming(message)
        if not await self._check_access(message):
            return

        import_session = self.get_import_session(message.user_id)
        if not import_session or import_session.state != ImportState.AWAITING_FILE:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="No import in progress. Use /import to start importing a conversation.",
                    chat_id=message.chat_id,
                )
            )
            return

        file_name = message.document_file_name or ""
        file_size = message.document_file_size or 0

        if not file_name.lower().endswith(".json"):
            await self._messenger.send_message(
                OutgoingMessage(
                    text="Please upload a .json file.",
                    chat_id=message.chat_id,
                )
            )
            return

        if file_size > 20 * 1024 * 1024:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="File is too large (max 20 MB). Please upload a smaller file.",
                    chat_id=message.chat_id,
                )
            )
            return

        file_id = message.document_file_id
        if not file_id:
            await self._messenger.send_message(
                OutgoingMessage(
                    text="Could not read the file. Please try again.",
                    chat_id=message.chat_id,
                )
            )
            return

        await self._messenger.send_message(
            OutgoingMessage(
                text=f"📄 Received: {file_name} ({file_size:,} bytes)\nParsing...",
                chat_id=message.chat_id,
            )
        )

        try:
            file_data = await self._messenger.download_file(file_id)
        except Exception:
            logger.exception("Failed to download file %s", file_id)
            await self._messenger.send_message(
                OutgoingMessage(
                    text="Failed to download the file from Telegram. Please try again.",
                    chat_id=message.chat_id,
                )
            )
            return

        from mai_gram.core.importer import ImportError as ImportParseError
        from mai_gram.core.importer import (
            extract_system_prompt,
            parse_import_json,
            save_imported_messages,
        )

        try:
            messages_data = parse_import_json(file_data)
        except ImportParseError as exc:
            self.clear_import_session(message.user_id)
            await self._messenger.send_message(
                OutgoingMessage(
                    text=f"❌ Import failed: {exc}\n\nUse /import to try again.",
                    chat_id=message.chat_id,
                )
            )
            return

        if not messages_data:
            self.clear_import_session(message.user_id)
            await self._messenger.send_message(
                OutgoingMessage(
                    text="❌ The file contains no messages.\n\nUse /import to try again.",
                    chat_id=message.chat_id,
                )
            )
            return

        system_prompt = extract_system_prompt(messages_data) or "You are a helpful assistant."
        chat_id = self._chat_id_for(message)
        user_id = message.user_id
        bot_id = message.bot_id or ""

        async with get_session() as db:
            existing = await self._get_chat(db, chat_id)
            if existing:
                self.clear_import_session(message.user_id)
                await self._messenger.send_message(
                    OutgoingMessage(
                        text=(
                            "A chat was created while you were importing. "
                            "Use /reset first, then /import again."
                        ),
                        chat_id=message.chat_id,
                    )
                )
                return

            chat = Chat(
                id=chat_id,
                user_id=user_id,
                bot_id=bot_id,
                llm_model=import_session.selected_model,
                system_prompt=system_prompt,
                prompt_name=None,
                timezone=self._settings.default_timezone,
                show_reasoning=True,
                show_tool_calls=True,
                send_datetime=False,
            )
            db.add(chat)
            await db.flush()

            message_store = MessageStore(db)
            imported_count = await save_imported_messages(chat_id, messages_data, message_store)
            await db.commit()

        self.clear_import_session(message.user_id)

        if imported_count == 0:
            await self._messenger.send_message(
                OutgoingMessage(
                    text=(
                        "❌ No messages could be imported (all entries were skipped).\n\n"
                        "The chat was created but has no history. "
                        "Send a message to start chatting."
                    ),
                    chat_id=message.chat_id,
                )
            )
            return

        await self._messenger.send_message(
            OutgoingMessage(
                text=(
                    f"✅ Saved {imported_count} messages to database.\n"
                    f"Model: {import_session.selected_model}\n"
                    f"System prompt: {system_prompt[:100]}"
                    f"{'...' if len(system_prompt) > 100 else ''}\n\n"
                    "Starting message replay..."
                ),
                chat_id=message.chat_id,
            )
        )

        async with get_session() as db:
            from sqlalchemy import asc

            result = await db.execute(
                select(Message).where(Message.chat_id == chat_id).order_by(asc(Message.id))
            )
            all_messages = list(result.scalars().all())

        from mai_gram.core.replay import replay_imported_messages

        tg_chat_id = message.chat_id

        async def _replay_task() -> None:
            try:
                await replay_imported_messages(
                    self._messenger,
                    tg_chat_id,
                    all_messages,
                )
            except Exception:
                logger.exception("Replay task failed for chat %s", chat_id)
                await self._messenger.send_message(
                    OutgoingMessage(
                        text=(
                            "⚠️ Replay was interrupted. "
                            "Your messages are saved in the database. "
                            "Send a message to continue chatting."
                        ),
                        chat_id=tg_chat_id,
                    )
                )
            finally:
                self._import_replay_tasks.pop(tg_chat_id, None)

        task = asyncio.create_task(_replay_task())
        self._import_replay_tasks[tg_chat_id] = task

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

        if self.is_in_import(message.user_id):
            await self._handle_import_callback(message)
            return

        if self.is_in_setup(message.user_id):
            await self._handle_setup_callback(message)
            return

        data = message.callback_data or ""

        if data.startswith("model:") or data.startswith("prompt:"):
            await self._messenger.send_message(
                OutgoingMessage(
                    text=(
                        f"Setup callback '{data}' ignored — no setup session active.\n"
                        "Hint: use --start --model MODEL --prompt PROMPT in a single command."
                    ),
                    chat_id=message.chat_id,
                )
            )
            return

        if data == "regen":
            await self._show_confirmation(
                message,
                "Regenerate this response?",
                confirm_data="confirm_regen",
                cancel_data="cancel_action",
            )
            return

        if data.startswith("cut:"):
            cut_msg_id = data.split(":", 1)[1]
            preview = await self._get_message_preview(int(cut_msg_id))
            confirm_text = (
                "Cut this message and all above?\nThey won't be sent to AI but remain searchable."
            )
            if preview:
                confirm_text += f'\n\nMessage: "{preview}"'
            tg_msg_id = ""
            if message.raw and hasattr(message.raw, "callback_query"):
                cb_msg = message.raw.callback_query.message
                if cb_msg:
                    tg_msg_id = str(cb_msg.message_id)
                    original_html = getattr(cb_msg, "text_html", None)
                    original_parse = None
                    if original_html:
                        original_parse = "html"
                    else:
                        original_html = cb_msg.text or ""
                    cache_key = f"{message.chat_id}:{tg_msg_id}"
                    self._cut_original_html[cache_key] = (original_html, original_parse)
            await self._show_confirmation(
                message,
                confirm_text,
                confirm_data=f"confirm_cut:{cut_msg_id}:{tg_msg_id}",
                cancel_data="cancel_action",
            )
            return

        if data == "confirm_regen":
            await self._delete_callback_message(message)
            await self._handle_regenerate(message)
            return

        if data.startswith("confirm_cut:"):
            parts = data.split(":", 2)
            cut_msg_id_str = parts[1]
            original_tg_msg_id = parts[2] if len(parts) > 2 else ""
            await self._delete_callback_message(message)
            await self._handle_cut_above(
                message, int(cut_msg_id_str), original_tg_msg_id=original_tg_msg_id
            )
            return

        if data.startswith("confirm_reset:"):
            chat_id = data.split(":", 1)[1]
            await self._delete_callback_message(message)
            await self._execute_reset(message, chat_id)
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

    def _get_allowed_models_for_bot(self) -> list[str]:
        """Return the model list for this bot, respecting per-bot restrictions."""
        global_models = self._settings.get_allowed_models()
        if self._bot_config and self._bot_config.allowed_models:
            bot_set = set(self._bot_config.allowed_models)
            return [m for m in global_models if m in bot_set]
        return global_models

    def _get_available_prompts_for_bot(self) -> dict[str, str]:
        """Return prompt templates available for this bot, respecting per-bot restrictions."""
        all_prompts = self._settings.get_available_prompts()
        if self._bot_config and self._bot_config.allowed_prompts:
            bot_set = set(self._bot_config.allowed_prompts)
            return {k: v for k, v in all_prompts.items() if k in bot_set}
        return all_prompts

    async def _show_model_selection(self, session: SetupSession) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = SetupState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._get_allowed_models_for_bot()
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
        available_prompts = self._get_available_prompts_for_bot()
        for name in available_prompts:
            keyboard_rows.append([(name.replace("_", " ").title(), f"prompt:{name}")])

        # Only show "Custom" if no per-bot prompt restriction is active
        if not (self._bot_config and self._bot_config.allowed_prompts):
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
            allowed = self._get_allowed_models_for_bot()
            if allowed and value not in allowed:
                await self._messenger.send_message(
                    OutgoingMessage(
                        text="This model is not available for this bot. Please choose another.",
                        chat_id=session.chat_id,
                    )
                )
                return
            session.selected_model = value
            await self._show_prompt_selection(session)

        elif category == "prompt" and session.state == SetupState.CHOOSING_PROMPT:
            if value == "__custom__":
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
            else:
                available = self._get_available_prompts_for_bot()
                prompt_text = available.get(value, "")
                if prompt_text:
                    await self._finish_setup(message, session, prompt_text, prompt_name=value)
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
        self,
        message: IncomingMessage,
        session: SetupSession,
        system_prompt: str,
        *,
        prompt_name: str | None = None,
    ) -> None:
        chat_id = self._chat_id_for(message)
        user_id = message.user_id
        bot_id = message.bot_id or ""

        prompt_cfg = self._settings.get_prompt_config(prompt_name) if prompt_name else None

        async with get_session() as db:
            send_dt = True
            if prompt_cfg is not None and prompt_cfg.send_datetime is not None:
                send_dt = prompt_cfg.send_datetime

            chat = Chat(
                id=chat_id,
                user_id=user_id,
                bot_id=bot_id,
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
                    f"Chat created!\n"
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
        async with get_session() as session:
            result = await session.execute(select(Message).where(Message.id == db_message_id))
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
                await self._messenger.delete_message(message.chat_id, str(cb_msg.message_id))

    async def _handle_cut_above(
        self,
        message: IncomingMessage,
        db_message_id: int,
        *,
        original_tg_msg_id: str = "",
    ) -> None:
        """Set the cut-above point so the target message and all before it are excluded."""
        chat_id = self._chat_id_for(message)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)

            cut_count = 0
            all_msgs = await message_store.get_all(chat.id)
            for m in all_msgs:
                if m.id <= db_message_id:
                    cut_count += 1

            chat.cut_above_message_id = db_message_id
            await session.commit()

        if original_tg_msg_id:
            cache_key = f"{message.chat_id}:{original_tg_msg_id}"
            cached = self._cut_original_html.pop(cache_key, None)
            if cached:
                original_html, original_parse = cached
                badge = "\u2702\ufe0f <i>[this and above are hidden from the AI]</i>"
                if original_parse == "html":
                    marked_text = f"{badge}\n\n{original_html}"
                else:
                    import html as _html

                    marked_text = f"{badge}\n\n{_html.escape(original_html)}"
                if len(marked_text) > 4000:
                    marked_text = marked_text[:4000] + "..."
                await self._messenger.edit_message(
                    message.chat_id,
                    original_tg_msg_id,
                    marked_text,
                    parse_mode="html",
                )

        footer_lines = ["\u2500" * 20, "\u2702\ufe0f History cut applied"]
        if cut_count > 0:
            footer_lines.append(f"\U0001f4e6 {cut_count} message(s) hidden from AI")
        footer_lines.append("\u2139\ufe0f Hidden messages are still searchable via tools")
        footer_lines.append("\u2500" * 20)
        await self._messenger.send_message(
            OutgoingMessage(
                text="\n".join(footer_lines),
                chat_id=message.chat_id,
            )
        )

    # -- Reset with backup --

    async def _create_reset_backup(self, chat_id: str) -> Path | None:
        """Create a zip archive of the data directory before reset.

        Archives the SQLite database and the chat's wiki directory (if any).
        Returns the path to the created archive, or None on failure.
        """
        import shutil
        import tempfile

        settings = get_settings()
        data_dir = Path(settings.memory_data_dir)
        backups_dir = data_dir / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_chat_id = chat_id.replace("@", "_at_")
        archive_name = f"reset_backup_{safe_chat_id}_{timestamp}"

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                staging = Path(tmp_dir) / archive_name
                staging.mkdir()

                db_url = settings.database_url
                if "sqlite" in db_url:
                    db_path = Path(db_url.split("///")[-1])
                    if db_path.exists():
                        shutil.copy2(db_path, staging / db_path.name)

                wiki_dir = data_dir / chat_id / "wiki"
                if wiki_dir.exists():
                    shutil.copytree(wiki_dir, staging / "wiki")

                archive_path = shutil.make_archive(
                    str(backups_dir / archive_name), "zip", tmp_dir, archive_name
                )
                result = Path(archive_path)
                logger.info(
                    "Reset backup created: %s (%.1f KB)",
                    result,
                    result.stat().st_size / 1024,
                )
                return result
        except Exception:
            logger.exception("Failed to create reset backup for chat %s", chat_id)
            return None

    async def _execute_reset(self, message: IncomingMessage, chat_id: str) -> None:
        """Create a backup and then delete the chat and all its history."""
        import shutil

        await self._messenger.send_message(
            OutgoingMessage(
                text="\U0001f4be Creating backup...",
                chat_id=message.chat_id,
            )
        )

        backup_path = await self._create_reset_backup(chat_id)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if chat:
                await session.delete(chat)
                await session.commit()
                chat_data_dir = Path(self._memory_data_dir) / chat_id
                if chat_data_dir.exists():
                    shutil.rmtree(chat_data_dir, ignore_errors=True)
                if backup_path:
                    msg = (
                        "\u2705 Chat reset. All history deleted.\n"
                        f"\U0001f4be Backup saved: {backup_path.name}\n\n"
                        "Use /start to create a new chat."
                    )
                else:
                    msg = (
                        "\u2705 Chat reset. All history deleted.\n"
                        "\u26a0\ufe0f Backup could not be created (check logs).\n\n"
                        "Use /start to create a new chat."
                    )
            else:
                msg = "Chat was already deleted. Use /start to create a new one."

        self.clear_setup_session(message.user_id)

        await self._messenger.send_message(OutgoingMessage(text=msg, chat_id=message.chat_id))

    # -- Regenerate --

    async def _handle_regenerate(self, message: IncomingMessage) -> None:
        """Handle the regen callback: delete last assistant message, re-generate."""
        if not message.callback_data:
            return

        chat_id = self._chat_id_for(message)

        # Determine whether trailing messages form a tool chain that should
        # be preserved (assistant with tool_calls + tool results).  When
        # tools already ran and only the final LLM response failed, we must
        # keep the tool chain so the LLM can simply retry the text response
        # without re-executing side-effecting tools (e.g. wiki_create).
        has_tool_chain = False
        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)

            recent = await message_store.get_recent(chat.id, limit=20)
            recent_sorted = sorted(recent, key=lambda m: m.id)

            trailing: list[Message] = []
            for msg in reversed(recent_sorted):
                if msg.role in ("assistant", "tool"):
                    trailing.append(msg)
                else:
                    break

            has_tool_results = any(m.role == "tool" for m in trailing)
            has_tool_call_assistant = any(m.role == "assistant" and m.tool_calls for m in trailing)
            has_tool_chain = has_tool_results and has_tool_call_assistant

            if not has_tool_chain:
                for msg in trailing:
                    await session.delete(msg)
                await session.flush()

        if has_tool_chain:
            # Keep tool chain in DB; only delete the error/button message
            # from Telegram, not the tool call display messages.
            self._response_message_ids.pop(message.chat_id, None)
        else:
            # Normal regeneration: delete intermediate Telegram messages
            prev_msg_ids = self._response_message_ids.pop(message.chat_id, [])
            for mid in prev_msg_ids:
                await self._messenger.delete_message(message.chat_id, mid)

        # Delete the Telegram message with the regen button
        if message.raw and hasattr(message.raw, "callback_query"):
            cb_msg = message.raw.callback_query.message
            if cb_msg:
                await self._messenger.delete_message(message.chat_id, str(cb_msg.message_id))

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if not chat:
                return

            message_store = MessageStore(session)
            wiki_store = WikiStore(session, data_dir=self._memory_data_dir)

            recent = await message_store.get_recent(chat.id, limit=1)
            last_role = recent[0].role if recent else None
            if last_role not in ("user", "tool"):
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

            mcp_manager = self._build_mcp_manager(chat, message_store, wiki_store)

            regen_tz = chat.timezone
            regen_send_dt = chat.send_datetime
            llm_messages = await prompt_builder.build_context(
                chat,
                current_time=datetime.now(timezone.utc),
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
                    timezone_name=regen_tz,
                    show_datetime=regen_send_dt,
                    show_reasoning=chat.show_reasoning,
                    show_tool_calls=chat.show_tool_calls,
                    extra_params=self._settings.get_model_params(chat.llm_model),
                    failure_log_message="Failed to regenerate response",
                )
            )
            self._response_message_ids[message.chat_id] = result.sent_message_ids

    # -- DB helpers --

    async def _get_chat(self, session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
