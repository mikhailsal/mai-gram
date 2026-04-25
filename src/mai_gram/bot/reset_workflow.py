"""Reset workflow for chat confirmation, backup, and deletion."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import func, select

from mai_gram.config import get_settings
from mai_gram.db.database import get_session
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


class ResetPresenter(Protocol):
    async def _show_confirmation(
        self,
        message: IncomingMessage,
        text: str,
        *,
        confirm_data: str,
        cancel_data: str,
    ) -> None: ...


class ResetWorkflow:
    """Own the reset confirmation, backup, and deletion flow."""

    def __init__(
        self,
        messenger: Messenger,
        *,
        presenter: ResetPresenter,
        resolve_chat_id: Callable[[IncomingMessage], str],
        clear_setup_session: Callable[[str], None],
        memory_data_dir: str,
    ) -> None:
        self._messenger = messenger
        self._presenter = presenter
        self._resolve_chat_id = resolve_chat_id
        self._clear_setup_session = clear_setup_session
        self._memory_data_dir = memory_data_dir

    async def handle_reset(self, message: IncomingMessage) -> None:
        """Prepare the reset confirmation or report when no chat exists."""
        chat_id = self._resolve_chat_id(message)

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
        await self._presenter._show_confirmation(
            message,
            confirm_text,
            confirm_data=f"confirm_reset:{chat_id}",
            cancel_data="cancel_action",
        )

    async def create_reset_backup(self, chat_id: str) -> Path | None:
        """Create a backup archive for the chat before deletion."""
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

    async def execute_reset(self, message: IncomingMessage, chat_id: str) -> None:
        """Create a backup and then delete the chat and its artifacts."""
        import shutil

        await self._messenger.send_message(
            OutgoingMessage(
                text="💾 Creating backup...",
                chat_id=message.chat_id,
            )
        )

        backup_path = await self.create_reset_backup(chat_id)

        async with get_session() as session:
            chat = await self._get_chat(session, chat_id)
            if chat:
                await session.delete(chat)
                await session.commit()
                chat_data_dir = Path(self._memory_data_dir) / chat_id
                if chat_data_dir.exists():
                    shutil.rmtree(chat_data_dir, ignore_errors=True)
                if backup_path:
                    result_text = (
                        "✅ Chat reset. All history deleted.\n"
                        f"💾 Backup saved: {backup_path.name}\n\n"
                        "Use /start to create a new chat."
                    )
                else:
                    result_text = (
                        "✅ Chat reset. All history deleted.\n"
                        "⚠️ Backup could not be created (check logs).\n\n"
                        "Use /start to create a new chat."
                    )
            else:
                result_text = "Chat was already deleted. Use /start to create a new one."

        self._clear_setup_session(message.user_id)
        await self._messenger.send_message(
            OutgoingMessage(text=result_text, chat_id=message.chat_id)
        )

    @staticmethod
    async def _get_chat(session: AsyncSession, chat_id: str) -> Chat | None:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        return result.scalar_one_or_none()
