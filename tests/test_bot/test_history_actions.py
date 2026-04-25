"""Tests for history preview and cut-above actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.history_actions import HistoryActions
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, MessageType, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_message(
    *,
    user_id: str = "test-user",
    chat_id: str = "test-chat",
    bot_id: str = "",
) -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id=user_id,
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.CALLBACK,
        bot_id=bot_id,
    )


def _make_actions() -> HistoryActions:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.edit_message = AsyncMock(return_value=SendResult(success=True))
    return HistoryActions(
        messenger,
        resolve_chat_id=lambda message: (
            f"{message.user_id}@{message.bot_id}" if message.bot_id else message.chat_id
        ),
    )


def _last_sent_text(actions: HistoryActions) -> str:
    send_message = cast("AsyncMock", actions._messenger.send_message)
    await_args = send_message.await_args
    assert await_args is not None
    return cast("str", await_args.args[0].text)


class TestMessagePreview:
    async def test_preview_truncates_long_content(self, session: AsyncSession) -> None:
        message = Message(chat_id="test-user@test-bot", role="assistant", content="x" * 120)
        session.add(message)
        await session.commit()

        actions = _make_actions()
        with patch("mai_gram.bot.history_actions.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            preview = await actions.get_message_preview(message.id, max_len=20)

        assert preview == ("x" * 20) + "..."


class TestCutAbove:
    async def test_cut_above_updates_chat_and_marks_original_message(
        self, session: AsyncSession
    ) -> None:
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="prompt",
        )
        session.add(chat)
        session.add_all(
            [
                Message(chat_id=chat.id, role="user", content="First"),
                Message(chat_id=chat.id, role="assistant", content="Second"),
            ]
        )
        await session.commit()

        cut_target = (
            (
                await session.execute(
                    select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
                )
            )
            .scalars()
            .first()
        )
        assert cut_target is not None

        actions = _make_actions()
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")
        with patch("mai_gram.bot.history_actions.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            await actions.handle_cut_above(
                message,
                cut_target.id,
                original_tg_msg_id="tg-msg-1",
                cached_original=("Original text", None),
            )

        refreshed_chat = (
            await session.execute(select(Chat).where(Chat.id == chat.id))
        ).scalar_one()
        assert refreshed_chat.cut_above_message_id == cut_target.id

        edit_message = cast("AsyncMock", actions._messenger.edit_message)
        assert edit_message.await_count == 1
        sent_text = _last_sent_text(actions)
        assert "History cut applied" in sent_text
        assert "1 message(s) hidden from AI" in sent_text
