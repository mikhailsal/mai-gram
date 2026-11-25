"""Tests for history preview and cut-above actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mai_gram.bot.history_actions import HistoryActions
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore
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
        message = Message(chat_id="test-user@test-bot", role="assistant", content="x" * 250)
        session.add(message)
        await session.commit()

        actions = _make_actions()
        with patch("mai_gram.bot.history_actions.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            preview = await actions.get_message_preview(
                _make_message(bot_id="test-bot"), message.id
            )

        assert preview is not None
        assert preview.token == message.history_ref
        assert preview.question is None
        assert "Response: " + "x" * 220 + "…" in preview.describe()


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
        question = Message(chat_id=chat.id, role="user", content="First")
        cut_target = Message(chat_id=chat.id, role="assistant", content="Second")
        later_question = Message(chat_id=chat.id, role="user", content="Third")
        later_answer = Message(chat_id=chat.id, role="assistant", content="Fourth")
        session.add_all([question, cut_target, later_question, later_answer])
        await session.commit()

        actions = _make_actions()
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")
        with patch("mai_gram.bot.history_actions.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            # A new, valid confirmation can move the cut to an earlier reply.
            await actions.handle_cut_above(message, later_answer.id)
            await session.refresh(chat)
            assert chat.cut_above_message_id == later_answer.id
            await actions.handle_cut_above(
                message,
                cut_target.id,
                original_tg_msg_id="tg-msg-1",
                cached_original=("Original text", None),
            )

        await session.refresh(chat)
        assert chat.cut_above_message_id == cut_target.id
        store = MessageStore(session)
        assert await store.get_all(chat.id) == [question, cut_target, later_question, later_answer]
        assert await store.get_recent(chat.id, after_message_id=chat.cut_above_message_id) == [
            later_answer,
            later_question,
        ]

        edit_message = cast("AsyncMock", actions._messenger.edit_message)
        assert edit_message.await_count == 1
        sent_text = _last_sent_text(actions)
        assert "History cut applied" in sent_text
        assert "2 message(s) hidden from AI" in sent_text

    @pytest.mark.parametrize(
        ("target_chat_id", "role", "delete_target"),
        [
            ("test-user@other-bot", "assistant", False),
            ("another-user@test-bot", "assistant", False),
            ("test-user@test-bot", "user", False),
            ("test-user@test-bot", "assistant", True),
        ],
    )
    async def test_unavailable_targets_leave_cut_and_preview_unchanged(
        self, session: AsyncSession, target_chat_id: str, role: str, delete_target: bool
    ) -> None:
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="prompt",
        )
        session.add(chat)
        boundary = Message(chat_id=chat.id, role="assistant", content="Existing boundary")
        target = Message(chat_id=target_chat_id, role=role, content="Wrong target")
        later = Message(chat_id=chat.id, role="assistant", content="Later reply")
        session.add_all([boundary, target, later])
        await session.flush()
        chat.cut_above_message_id = boundary.id
        if delete_target:
            await session.delete(target)
        await session.commit()

        actions = _make_actions()
        message = _make_message(chat_id="tg-chat", bot_id="test-bot")
        with patch("mai_gram.bot.history_actions.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            assert await actions.get_message_preview(message, target.id) is None
            await actions.handle_cut_above(
                message, target.id, original_tg_msg_id="77", cached_original=("Wrong", None)
            )
            assert "no longer available" in _last_sent_text(actions)
            await actions.handle_cut_above(message, boundary.id)

        await session.refresh(chat)
        assert chat.cut_above_message_id == boundary.id
        visible = [later, target] if role == "user" else [later]
        assert (
            await MessageStore(session).get_recent(chat.id, after_message_id=boundary.id) == visible
        )
        actions._messenger.edit_message.assert_not_awaited()
        assert "no longer available" in _last_sent_text(actions)
