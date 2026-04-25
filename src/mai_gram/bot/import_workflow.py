"""Import workflow for JSON conversation uploads and replay."""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import Settings
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


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


class ImportWorkflow:
    """Own the /import state machine and document upload flow."""

    IMPORT_SESSION_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        messenger: Messenger,
        settings: Settings,
        *,
        get_allowed_models: Callable[[], list[str]],
        resolve_chat_id: Callable[[IncomingMessage], str],
    ) -> None:
        self._messenger = messenger
        self._settings = settings
        self._get_allowed_models = get_allowed_models
        self._resolve_chat_id = resolve_chat_id
        self._sessions: dict[str, ImportSession] = {}
        self._replay_tasks: dict[str, asyncio.Task[None]] = {}

    def is_in_import(self, user_id: str) -> bool:
        session = self._sessions.get(user_id)
        if session is None:
            return False
        if time.monotonic() - session.created_at > self.IMPORT_SESSION_TIMEOUT_SECONDS:
            self._sessions.pop(user_id, None)
            return False
        return True

    def get_import_session(self, user_id: str) -> ImportSession | None:
        if not self.is_in_import(user_id):
            return None
        return self._sessions.get(user_id)

    def clear_import_session(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    async def handle_import(self, message: IncomingMessage, *, in_setup: bool) -> None:
        """Start the import flow by validating preconditions and prompting for a model."""
        user_id = message.user_id
        if in_setup:
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

        chat_id = self._resolve_chat_id(message)
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
        self._sessions[user_id] = import_session
        await self._show_import_model_selection(import_session)

    async def _show_import_model_selection(self, session: ImportSession) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        session.state = ImportState.CHOOSING_MODEL
        keyboard_rows = []
        allowed_models = self._get_allowed_models()
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

    async def handle_import_callback(self, message: IncomingMessage) -> None:
        """Handle callback queries during the import flow."""
        session = self._sessions.get(message.user_id)
        if not session or not message.callback_data:
            return

        data = message.callback_data
        if not data.startswith("import_model:"):
            return

        model = data.split(":", 1)[1]
        allowed = self._get_allowed_models()
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

    async def handle_document(self, message: IncomingMessage) -> None:
        """Handle an uploaded JSON file for the active import session."""
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
        chat_id = self._resolve_chat_id(message)
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
                self._replay_tasks.pop(tg_chat_id, None)

        task = asyncio.create_task(_replay_task())
        self._replay_tasks[tg_chat_id] = task

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
