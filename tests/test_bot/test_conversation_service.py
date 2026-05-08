"""Unit tests for the ordinary conversation service."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from mai_gram.bot.conversation_executor import AssistantTurnResult
from mai_gram.bot.conversation_service import ConversationService
from mai_gram.db.models import Chat
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_message(chat_id: str = "test-chat") -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id="test-user",
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.TEXT,
        text="Hello there",
        bot_id="test-bot",
    )


def _make_service() -> tuple[ConversationService, MagicMock, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.send_typing_indicator = AsyncMock()
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=AssistantTurnResult(sent_message_ids=["resp-1"]))
    turn_builder = MagicMock()
    turn_builder.save_user_message_and_build_request = AsyncMock(return_value=SimpleNamespace())
    service = ConversationService(
        messenger,
        executor,
        turn_builder=turn_builder,
        resolve_chat_id=lambda message: f"{message.user_id}@{message.bot_id}",
    )
    return service, messenger, executor, turn_builder


class TestConversationService:
    async def test_requires_existing_chat(self, session: AsyncSession) -> None:
        service, messenger, executor, _ = _make_service()

        with patch("mai_gram.bot.conversation_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            sent_ids = await service.handle_message(_make_message())

        assert sent_ids == []
        executor.execute.assert_not_awaited()
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        outgoing = cast("OutgoingMessage", await_args.args[0])
        assert "No chat configured" in outgoing.text

    async def test_executes_built_request_for_existing_chat(self, session: AsyncSession) -> None:
        service, messenger, executor, turn_builder = _make_service()
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="prompt",
        )
        session.add(chat)
        await session.commit()
        request = SimpleNamespace(name="request")
        turn_builder.save_user_message_and_build_request.return_value = request

        with patch("mai_gram.bot.conversation_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            sent_ids = await service.handle_message(_make_message(chat_id="telegram-chat"))

        assert sent_ids == ["resp-1"]
        messenger.send_typing_indicator.assert_awaited_once_with("telegram-chat")
        turn_builder.save_user_message_and_build_request.assert_awaited_once()
        executor.execute.assert_awaited_once_with(request)

    async def test_download_photo_returns_data_uri(self) -> None:
        service, messenger, _, _ = _make_service()
        photo_bytes = b"\x89PNG_fake_data"
        messenger.download_file = AsyncMock(return_value=photo_bytes)

        msg = IncomingMessage(
            platform="telegram",
            user_id="test-user",
            chat_id="test-chat",
            message_id="msg-1",
            message_type=MessageType.TEXT,
            text="",
            photo_file_id="file-id-123",
        )

        result = await service._download_photo(msg)

        expected_b64 = base64.b64encode(photo_bytes).decode("ascii")
        assert result == [f"data:image/jpeg;base64,{expected_b64}"]
        messenger.download_file.assert_awaited_once_with("file-id-123")

    async def test_download_photo_returns_none_without_photo(self) -> None:
        service, _, _, _ = _make_service()

        msg = _make_message()

        result = await service._download_photo(msg)

        assert result is None

    async def test_download_photo_returns_none_on_error(self) -> None:
        service, messenger, _, _ = _make_service()
        messenger.download_file = AsyncMock(side_effect=RuntimeError("download failed"))

        msg = IncomingMessage(
            platform="telegram",
            user_id="test-user",
            chat_id="test-chat",
            message_id="msg-1",
            message_type=MessageType.TEXT,
            text="",
            photo_file_id="file-id-broken",
        )

        result = await service._download_photo(msg)

        assert result is None
