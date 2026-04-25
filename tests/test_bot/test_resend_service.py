"""Unit tests for the resend-last service."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.resend_service import ResendService
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, MessageType, OutgoingMessage, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_service() -> tuple[ResendService, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.delete_message = AsyncMock()

    renderer = MagicMock()
    renderer._send_response = AsyncMock(return_value=["resent-1", "resent-2"])

    service = ResendService(
        messenger,
        renderer=renderer,
        resolve_chat_id=lambda message: message.chat_id,
    )
    return service, messenger, renderer


def _make_message(chat_id: str = "test-resend-chat") -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id="test-user",
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.COMMAND,
    )


class TestResendService:
    async def test_handle_resend_requires_existing_chat(self, session: AsyncSession) -> None:
        service, messenger, _ = _make_service()

        with patch("mai_gram.bot.resend_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await service.handle_resend(_make_message(), previous_response_ids=[])

        assert result.replaced_previous is False
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "No chat configured" in sent.text

    async def test_handle_resend_requires_assistant_message(self, session: AsyncSession) -> None:
        service, messenger, _ = _make_service()
        session.add(
            Chat(
                id="test-resend-chat",
                user_id="test-user",
                bot_id="",
                llm_model="test-model",
                system_prompt="prompt",
            )
        )
        await session.commit()

        with patch("mai_gram.bot.resend_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await service.handle_resend(_make_message(), previous_response_ids=[])

        assert result.replaced_previous is False
        send_message = cast("AsyncMock", messenger.send_message)
        await_args = send_message.await_args
        assert await_args is not None
        sent = cast("OutgoingMessage", await_args.args[0])
        assert "No assistant message found" in sent.text

    async def test_handle_resend_rehydrates_last_assistant_message(
        self, session: AsyncSession
    ) -> None:
        service, messenger, renderer = _make_service()
        session.add(
            Chat(
                id="test-resend-chat",
                user_id="test-user",
                bot_id="",
                llm_model="test-model",
                system_prompt="prompt",
                show_reasoning=True,
            )
        )
        session.add(
            Message(
                chat_id="test-resend-chat",
                role="assistant",
                content="Persisted answer",
                reasoning="Persisted reasoning",
            )
        )
        await session.commit()

        with patch("mai_gram.bot.resend_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await service.handle_resend(
                _make_message(),
                previous_response_ids=["old-1", "old-2"],
            )

        assert result.replaced_previous is True
        assert result.sent_message_ids == ["resent-1", "resent-2"]
        assert messenger.delete_message.await_count == 2
        renderer._send_response.assert_awaited_once()

        saved_message = (
            await session.execute(
                select(Message)
                .where(Message.chat_id == "test-resend-chat")
                .order_by(Message.id.desc())
            )
        ).scalar_one()
        assert saved_message.content == "Persisted answer"

        send_message = cast("AsyncMock", messenger.send_message)
        texts = [
            cast("OutgoingMessage", call.args[0]).text for call in send_message.await_args_list
        ]
        assert any("Resent last AI message" in text for text in texts)
