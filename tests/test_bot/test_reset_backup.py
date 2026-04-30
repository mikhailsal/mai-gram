"""Tests for reset confirmation and backup functionality."""

from __future__ import annotations

import zipfile
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from mai_gram.bot.conversation_executor import AssistantTurnResult
from mai_gram.bot.handler import BotHandler
from mai_gram.bot.reset_workflow import ResetWorkflow
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, MessageType, SendResult

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


def _make_handler(*, memory_data_dir: str = "./data") -> BotHandler:
    """Create a BotHandler with mocked dependencies."""
    messenger = MagicMock()
    messenger.register_command_handler = MagicMock()
    messenger.register_message_handler = MagicMock()
    messenger.register_callback_handler = MagicMock()
    messenger.register_document_handler = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.edit_message = AsyncMock(return_value=SendResult(success=True))
    messenger.delete_message = AsyncMock()
    messenger.send_typing_indicator = AsyncMock()

    llm = MagicMock()

    with patch("mai_gram.bot.handler.get_settings") as mock_settings:
        settings = MagicMock()
        settings.memory_data_dir = memory_data_dir
        settings.wiki_context_limit = 20
        settings.short_term_limit = 500
        settings.tool_max_iterations = 5
        settings.get_allowed_user_ids.return_value = set()
        mock_settings.return_value = settings

        handler = BotHandler(
            messenger,
            llm,
            memory_data_dir=memory_data_dir,
            test_mode=True,
        )
    return handler


def _make_reset_workflow(
    *,
    memory_data_dir: str = "./data",
    database_url: str = "sqlite+aiosqlite:///./data/test.db",
) -> tuple[ResetWorkflow, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    presenter = MagicMock()
    presenter._show_confirmation = AsyncMock()
    clear_setup_session = MagicMock()

    workflow = ResetWorkflow(
        messenger,
        presenter=presenter,
        resolve_chat_id=lambda message: (
            f"{message.user_id}@{message.bot_id}" if message.bot_id else message.chat_id
        ),
        clear_setup_session=clear_setup_session,
        memory_data_dir=memory_data_dir,
        database_url=database_url,
    )
    return workflow, presenter, clear_setup_session


def _make_message(
    *,
    user_id: str = "test-user",
    chat_id: str = "test-chat",
    text: str = "",
    callback_data: str | None = None,
    bot_id: str = "",
    message_type: MessageType = MessageType.COMMAND,
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        message_id="msg-1",
        message_type=message_type,
        text=text,
        callback_data=callback_data,
        bot_id=bot_id,
    )


class TestCreateResetBackup:
    async def test_backup_creates_zip(self, tmp_path: Path) -> None:
        """Backup should create a zip archive in data/backups/."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        db_path = data_dir / "test.db"
        db_path.write_text("fake database content")

        chat_id = "test-user@test-bot"
        wiki_dir = data_dir / chat_id / "wiki"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "note.md").write_text("# Test wiki entry")

        workflow, _, _ = _make_reset_workflow(
            memory_data_dir=str(data_dir),
            database_url=f"sqlite+aiosqlite:///{db_path}",
        )

        result = await workflow.create_reset_backup(chat_id)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".zip"
        assert "reset_backup_" in result.name

        backups_dir = data_dir / "backups"
        assert backups_dir.exists()
        assert result.parent == backups_dir

        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
            assert any("test.db" in n for n in names)
            assert any("wiki/note.md" in n for n in names)

    async def test_backup_without_wiki(self, tmp_path: Path) -> None:
        """Backup should work even when the chat has no wiki directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        db_path = data_dir / "test.db"
        db_path.write_text("fake database content")

        workflow, _, _ = _make_reset_workflow(
            memory_data_dir=str(data_dir),
            database_url=f"sqlite+aiosqlite:///{db_path}",
        )

        result = await workflow.create_reset_backup("plain-chat@bot")

        assert result is not None
        assert result.exists()

        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
            assert any("test.db" in n for n in names)
            assert not any(n.endswith("/wiki/") or "/wiki/" in n for n in names)

    async def test_backup_handles_failure_gracefully(self, tmp_path: Path) -> None:
        """Backup should return None on failure instead of raising."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        workflow, _, _ = _make_reset_workflow(
            memory_data_dir=str(data_dir),
            database_url=f"sqlite+aiosqlite:///{data_dir / 'test.db'}",
        )

        with patch("shutil.make_archive", side_effect=OSError("disk full")):
            result = await workflow.create_reset_backup("test@bot")

        assert result is None

    async def test_backup_propagates_unexpected_errors(self, tmp_path: Path) -> None:
        """Unexpected errors should not be swallowed by backup creation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        workflow, _, _ = _make_reset_workflow(
            memory_data_dir=str(data_dir),
            database_url=f"sqlite+aiosqlite:///{data_dir / 'test.db'}",
        )

        with (
            patch("shutil.make_archive", side_effect=TypeError("bug")),
            pytest.raises(TypeError, match="bug"),
        ):
            await workflow.create_reset_backup("test@bot")


class TestResetConfirmation:
    async def test_reset_shows_confirmation(self, session: AsyncSession) -> None:
        """The /reset command should show a Yes/Cancel confirmation."""
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="test prompt",
        )
        session.add(chat)
        msg = Message(
            chat_id="test-user@test-bot",
            role="user",
            content="hello",
        )
        session.add(msg)
        await session.commit()

        workflow, presenter, _ = _make_reset_workflow()
        message = _make_message(
            user_id="test-user",
            chat_id="test-user@test-bot",
            bot_id="test-bot",
        )

        with patch("mai_gram.bot.reset_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_reset(message)

        show_confirmation = cast("AsyncMock", presenter._show_confirmation)
        await_args = show_confirmation.await_args
        assert await_args is not None
        assert "Reset this chat?" in await_args.args[1]
        assert await_args.kwargs["confirm_data"] == "confirm_reset:test-user@test-bot"

    async def test_reset_no_chat_responds_immediately(self) -> None:
        """If there is no chat, /reset should respond immediately without confirmation."""
        workflow, presenter, _ = _make_reset_workflow()
        message = _make_message(
            user_id="test-user",
            chat_id="test-chat",
        )

        with patch("mai_gram.bot.reset_workflow.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_reset(message)

        send_message = cast("AsyncMock", workflow._messenger.send_message)
        calls = send_message.call_args_list
        assert len(calls) == 1
        assert calls[0].args
        sent_msg = calls[0].args[0]
        assert "No chat to reset" in sent_msg.text
        assert cast("AsyncMock", presenter._show_confirmation).await_count == 0


class TestExecuteReset:
    async def test_execute_reset_deletes_chat_and_artifacts(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "data"
        chat_id = "test-user@test-bot"
        chat_dir = data_dir / chat_id / "wiki"
        chat_dir.mkdir(parents=True)
        (chat_dir / "fact.md").write_text("fact")

        chat = Chat(
            id=chat_id,
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="test prompt",
        )
        session.add(chat)
        await session.commit()

        workflow, _, clear_setup_session = _make_reset_workflow(memory_data_dir=str(data_dir))
        message = _make_message(user_id="test-user", chat_id="tg-chat", bot_id="test-bot")
        backup_path = data_dir / "backups" / "reset_backup.zip"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text("backup")

        with (
            patch("mai_gram.bot.reset_workflow.get_session") as mock_get_session,
            patch.object(workflow, "create_reset_backup", new_callable=AsyncMock) as mock_backup,
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_backup.return_value = backup_path

            await workflow.execute_reset(message, chat_id)

        stored_chat = (
            await session.execute(select(Chat).where(Chat.id == chat_id))
        ).scalar_one_or_none()
        assert stored_chat is None
        assert not (data_dir / chat_id).exists()
        clear_setup_session.assert_called_once_with("test-user")

        send_message = cast("AsyncMock", workflow._messenger.send_message)
        sent_texts = [call.args[0].text for call in send_message.await_args_list]
        assert sent_texts[0] == "💾 Creating backup..."
        assert any("Backup saved: reset_backup.zip" in text for text in sent_texts)


class TestRegenerate:
    async def test_regenerate_preserves_trailing_tool_chain(self, session: AsyncSession) -> None:
        """Regenerate should keep a trailing assistant+tool chain and reuse the shared executor."""
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="test prompt",
        )
        session.add(chat)
        session.add_all(
            [
                Message(
                    chat_id=chat.id,
                    role="user",
                    content="Remember this.",
                ),
                Message(
                    chat_id=chat.id,
                    role="assistant",
                    content="",
                    tool_calls='[{"id":"call_1","name":"wiki_create","arguments":"{}"}]',
                ),
                Message(
                    chat_id=chat.id,
                    role="tool",
                    content="Stored successfully",
                    tool_call_id="call_1",
                ),
            ]
        )
        await session.commit()

        handler = _make_handler()
        handler._response_message_ids[chat.id] = ["tool-display-1", "tool-display-2"]

        message = _make_message(
            user_id="test-user",
            chat_id=chat.id,
            bot_id="test-bot",
            callback_data="confirm_regen",
            message_type=MessageType.CALLBACK,
        )

        with (
            patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session,
            patch.object(
                handler._assistant_turn_builder,
                "build_request",
                new_callable=AsyncMock,
            ) as mock_build_request,
            patch.object(
                handler._conversation_executor,
                "execute",
                new_callable=AsyncMock,
            ) as mock_execute,
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_build_request.return_value = MagicMock(llm_messages=[])
            mock_execute.return_value = AssistantTurnResult(sent_message_ids=[])

            await handler._handle_regenerate(message)

        persisted_messages = list(
            (
                await session.execute(
                    select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
                )
            ).scalars()
        )

        assert [item.role for item in persisted_messages] == ["user", "assistant", "tool"]
        assert handler._response_message_ids[chat.id] == []
        delete_message = cast("AsyncMock", handler._messenger.delete_message)
        assert delete_message.await_count == 0
        mock_execute.assert_awaited_once()

        await_args = mock_execute.await_args
        assert await_args is not None
        request = await_args.args[0]
        assert request.llm_messages == []
        mock_build_request.assert_awaited_once()
