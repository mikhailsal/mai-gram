"""Tests for the regenerate service extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from mai_gram.bot.conversation_executor import AssistantTurnResult
from mai_gram.bot.regenerate_service import RegenerateService
from mai_gram.db.models import Chat, Message
from mai_gram.messenger.base import IncomingMessage, MessageType, SendResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def _make_message(*, chat_id: str, bot_id: str = "test-bot") -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id="test-user",
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.CALLBACK,
        callback_data="confirm_regen",
        bot_id=bot_id,
    )


def _make_service() -> tuple[RegenerateService, MagicMock, AsyncMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="42"))
    messenger.delete_message = AsyncMock()
    messenger.send_typing_indicator = AsyncMock()
    llm = MagicMock()
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=AssistantTurnResult(sent_message_ids=[]))
    settings = MagicMock()
    settings.get_model_params.return_value = {}

    service = RegenerateService(
        messenger,
        llm,
        executor,
        settings,
        resolve_chat_id=lambda message: f"{message.user_id}@{message.bot_id}",
        build_mcp_manager=MagicMock(return_value=MagicMock()),
        memory_data_dir="./data",
        wiki_context_limit=20,
        short_term_limit=500,
        test_mode=True,
    )
    return service, messenger, cast("AsyncMock", executor.execute)


def _last_outgoing_text(messenger: MagicMock) -> str:
    send_message = cast("AsyncMock", messenger.send_message)
    await_args = send_message.await_args
    assert await_args is not None
    return cast("str", await_args.args[0].text)


class TestRegenerateService:
    async def test_preserves_trailing_tool_chain(self, session: AsyncSession) -> None:
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
                Message(chat_id=chat.id, role="user", content="Remember this."),
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

        service, messenger, execute = _make_service()

        with (
            patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session,
            patch("mai_gram.bot.regenerate_service.PromptBuilder") as mock_prompt_builder,
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_prompt_builder.return_value.build_context = AsyncMock(return_value=[])

            sent_ids = await service.handle_regenerate(
                _make_message(chat_id=chat.id),
                previous_response_ids=["tool-display-1", "tool-display-2"],
            )

        persisted_messages = list(
            (
                await session.execute(
                    select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
                )
            ).scalars()
        )
        assert [item.role for item in persisted_messages] == ["user", "assistant", "tool"]
        assert sent_ids == []
        assert cast("AsyncMock", messenger.delete_message).await_count == 0
        execute.assert_awaited_once()

    async def test_normal_regenerate_deletes_previous_response_ids(
        self, session: AsyncSession
    ) -> None:
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
                Message(chat_id=chat.id, role="user", content="Question"),
                Message(chat_id=chat.id, role="assistant", content="Old answer"),
            ]
        )
        await session.commit()

        service, messenger, execute = _make_service()
        execute.return_value = AssistantTurnResult(sent_message_ids=["new-response"])

        with (
            patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session,
            patch("mai_gram.bot.regenerate_service.PromptBuilder") as mock_prompt_builder,
        ):
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_prompt_builder.return_value.build_context = AsyncMock(return_value=[])

            sent_ids = await service.handle_regenerate(
                _make_message(chat_id=chat.id),
                previous_response_ids=["old-1", "old-2"],
            )

        persisted_messages = list(
            (
                await session.execute(
                    select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
                )
            ).scalars()
        )
        assert [item.role for item in persisted_messages] == ["user"]
        assert sent_ids == ["new-response"]
        delete_message = cast("AsyncMock", messenger.delete_message)
        deleted_ids = [call.args[1] for call in delete_message.await_args_list]
        assert deleted_ids == ["old-1", "old-2"]

    async def test_missing_user_message_reports_error(self, session: AsyncSession) -> None:
        chat = Chat(
            id="test-user@test-bot",
            user_id="test-user",
            bot_id="test-bot",
            llm_model="test-model",
            system_prompt="test prompt",
        )
        session.add(chat)
        session.add(Message(chat_id=chat.id, role="assistant", content="Only answer"))
        await session.commit()

        service, messenger, execute = _make_service()

        with patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            sent_ids = await service.handle_regenerate(
                _make_message(chat_id=chat.id),
                previous_response_ids=["old-1"],
            )

        assert sent_ids == []
        assert "Cannot regenerate: no user message found." in _last_outgoing_text(messenger)
        execute.assert_not_awaited()
