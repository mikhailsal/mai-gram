"""Unit tests for the import workflow service."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.import_workflow import ImportSession, ImportState, ImportWorkflow
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_workflow() -> tuple[ImportWorkflow, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.download_file = AsyncMock(return_value=b"[]")

    settings = MagicMock()
    settings.get_default_model.return_value = "openai/test-model"
    settings.default_timezone = "UTC"

    workflow = ImportWorkflow(
        messenger,
        settings,
        get_allowed_models=lambda: ["openai/test-model", "anthropic/other-model"],
        resolve_chat_id=lambda message: message.chat_id,
    )
    return workflow, messenger, settings


def _make_message(
    *,
    user_id: str = "test-user",
    chat_id: str = "test-import-chat",
    callback_data: str | None = None,
    document_file_id: str | None = None,
    document_file_name: str | None = None,
    document_file_size: int | None = None,
    bot_id: str = "",
    message_type: MessageType = MessageType.COMMAND,
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        message_id="msg-1",
        message_type=message_type,
        callback_data=callback_data,
        bot_id=bot_id,
        document_file_id=document_file_id,
        document_file_name=document_file_name,
        document_file_size=document_file_size,
    )


class TestImportWorkflow:
    def test_is_in_import_clears_stale_sessions(self) -> None:
        workflow, _, _ = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(
            user_id="test-user",
            chat_id="test-chat",
            created_at=0.0,
        )

        with patch("mai_gram.bot.import_workflow.time.monotonic", return_value=400.0):
            assert workflow.is_in_import("test-user") is False

        assert "test-user" not in workflow._sessions

    async def test_handle_import_blocks_active_setup(self) -> None:
        workflow, messenger, _ = _make_workflow()
        message = _make_message()

        await workflow.handle_import(message, in_setup=True)

        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "middle of a setup" in sent.text

    async def test_handle_import_rejects_existing_chat(self, session: AsyncSession) -> None:
        workflow, messenger, _ = _make_workflow()
        session.add(
            Chat(
                id="test-import-chat",
                user_id="test-user",
                bot_id="",
                llm_model="existing-model",
                system_prompt="prompt",
            )
        )
        await session.commit()

        with patch("mai_gram.bot.import_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_import(_make_message(), in_setup=False)

        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "Importing would conflict" in sent.text
        assert workflow.is_in_import("test-user") is False

    async def test_handle_import_starts_session_and_shows_models(
        self, session: AsyncSession
    ) -> None:
        workflow, messenger, _ = _make_workflow()

        with patch("mai_gram.bot.import_workflow.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_import(_make_message(), in_setup=False)

        assert workflow.is_in_import("test-user") is True
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "Import Mode" in sent.text
        assert sent.keyboard is not None

    async def test_handle_import_callback_rejects_disallowed_model(self) -> None:
        workflow, messenger, _ = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(user_id="test-user", chat_id="test-chat")
        message = _make_message(
            chat_id="test-chat",
            callback_data="import_model:not-allowed",
            message_type=MessageType.CALLBACK,
        )

        await workflow.handle_import_callback(message)

        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "not available" in sent.text
        assert workflow._sessions["test-user"].selected_model == ""

    async def test_handle_import_callback_moves_to_file_upload(self) -> None:
        workflow, messenger, _ = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(user_id="test-user", chat_id="test-chat")
        message = _make_message(
            chat_id="test-chat",
            callback_data="import_model:openai/test-model",
            message_type=MessageType.CALLBACK,
        )

        await workflow.handle_import_callback(message)

        assert workflow._sessions["test-user"].selected_model == "openai/test-model"
        assert workflow._sessions["test-user"].state == ImportState.AWAITING_FILE
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "Now upload your JSON file" in sent.text

    async def test_handle_document_requires_active_session(self) -> None:
        workflow, messenger, _ = _make_workflow()

        await workflow.handle_document(
            _make_message(
                message_type=MessageType.DOCUMENT,
                document_file_id="file-1",
                document_file_name="import.json",
                document_file_size=10,
            )
        )

        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "No import in progress" in sent.text

    async def test_handle_document_rejects_invalid_file_metadata(self) -> None:
        workflow, messenger, _ = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(
            user_id="test-user",
            chat_id="test-chat",
            state=ImportState.AWAITING_FILE,
            selected_model="openai/test-model",
            created_at=time.monotonic(),
        )

        await workflow.handle_document(
            _make_message(
                chat_id="test-chat",
                message_type=MessageType.DOCUMENT,
                document_file_id="file-1",
                document_file_name="import.txt",
                document_file_size=10,
            )
        )
        await workflow.handle_document(
            _make_message(
                chat_id="test-chat",
                message_type=MessageType.DOCUMENT,
                document_file_id=None,
                document_file_name="import.json",
                document_file_size=10,
            )
        )

        send_message = cast("AsyncMock", messenger.send_message)
        texts = [
            cast("OutgoingMessage", call.args[0]).text for call in send_message.await_args_list
        ]
        assert "Please upload a .json file." in texts[0]
        assert "Could not read the file" in texts[1]

    async def test_handle_document_clears_session_on_parse_failure(self) -> None:
        workflow, messenger, _ = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(
            user_id="test-user",
            chat_id="test-chat",
            state=ImportState.AWAITING_FILE,
            selected_model="openai/test-model",
            created_at=time.monotonic(),
        )
        messenger.download_file = AsyncMock(return_value=b"not json")

        await workflow.handle_document(
            _make_message(
                chat_id="test-chat",
                message_type=MessageType.DOCUMENT,
                document_file_id="file-1",
                document_file_name="import.json",
                document_file_size=10,
            )
        )

        assert workflow.is_in_import("test-user") is False
        send_message = cast("AsyncMock", messenger.send_message)
        assert any(
            "Import failed" in cast("OutgoingMessage", call.args[0]).text
            for call in send_message.await_args_list
        )

    async def test_handle_document_creates_chat_and_messages(self, session: AsyncSession) -> None:
        workflow, messenger, settings = _make_workflow()
        workflow._sessions["test-user"] = ImportSession(
            user_id="test-user",
            chat_id="test-chat",
            state=ImportState.AWAITING_FILE,
            selected_model="openai/test-model",
            created_at=time.monotonic(),
        )
        messenger.download_file = AsyncMock(
            return_value=json.dumps(
                [
                    {"role": "system", "content": "Imported system prompt"},
                    {"role": "user", "content": "Imported hello"},
                    {"role": "assistant", "content": "Imported reply"},
                ]
            ).encode()
        )
        replay_task = MagicMock()

        def _capture_task(coro: object) -> MagicMock:
            cast("Any", coro).close()
            return replay_task

        with (
            patch("mai_gram.bot.import_workflow.get_session") as mock_get_session,
            patch("mai_gram.bot.import_workflow.asyncio.create_task", side_effect=_capture_task),
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await workflow.handle_document(
                _make_message(
                    chat_id="test-import-chat",
                    message_type=MessageType.DOCUMENT,
                    document_file_id="file-1",
                    document_file_name="import.json",
                    document_file_size=10,
                )
            )

        saved_chat = (
            await session.execute(select(Chat).where(Chat.id == "test-import-chat"))
        ).scalar_one()
        saved_messages = list(
            (
                await session.execute(
                    select(Message)
                    .where(Message.chat_id == "test-import-chat")
                    .order_by(Message.id)
                )
            ).scalars()
        )

        assert saved_chat.llm_model == "openai/test-model"
        assert saved_chat.system_prompt == "Imported system prompt"
        assert settings.default_timezone == saved_chat.timezone
        assert [item.role for item in saved_messages] == ["user", "assistant"]
        assert workflow.is_in_import("test-user") is False
        assert workflow._replay_tasks["test-import-chat"] is replay_task
        send_message = cast("AsyncMock", messenger.send_message)
        texts = [
            cast("OutgoingMessage", call.args[0]).text for call in send_message.await_args_list
        ]
        assert any("Saved 2 messages" in text for text in texts)
