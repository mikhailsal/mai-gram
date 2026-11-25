"""Tests for the regenerate service extraction."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mai_gram.bot.assistant_turn_builder import AssistantTurnBuilder
from mai_gram.bot.conversation_executor import AssistantTurnResult, ConversationExecutor
from mai_gram.bot.regenerate_service import RegenerateService
from mai_gram.bot.response_renderer import ResponseRenderer
from mai_gram.db.models import Base, Chat, Message
from mai_gram.llm.provider import LLMProviderError, StreamChunk, ToolCall
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, MessageType, SendResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.config import Settings
    from mai_gram.llm.provider import ChatMessage


def _make_message(*, chat_id: str, callback_data: str = "confirm_regen") -> IncomingMessage:
    return IncomingMessage(
        platform="telegram",
        user_id="test-user",
        chat_id=chat_id,
        message_id="msg-1",
        message_type=MessageType.CALLBACK,
        callback_data=callback_data,
        bot_id="test-bot",
    )


def _make_service() -> tuple[RegenerateService, MagicMock, AsyncMock, MagicMock]:
    messenger = _make_messenger()
    executor = MagicMock()
    executor.execute = AsyncMock(return_value=AssistantTurnResult(sent_message_ids=[]))
    turn_builder = MagicMock()
    turn_builder.build_request = AsyncMock(return_value=MagicMock())

    service = RegenerateService(
        messenger,
        executor,
        turn_builder=turn_builder,
        resolve_chat_id=lambda message: f"{message.user_id}@{message.bot_id}",
    )
    return service, messenger, cast("AsyncMock", executor.execute), turn_builder


def _last_outgoing_text(messenger: MagicMock) -> str:
    send_message = cast("AsyncMock", messenger.send_message)
    await_args = send_message.await_args
    assert await_args is not None
    return cast("str", await_args.args[0].text)


def _make_messenger() -> MagicMock:
    return MagicMock(
        max_message_length=4000,
        send_message=AsyncMock(return_value=SendResult(success=True, message_id="42")),
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id="42")),
        delete_message=AsyncMock(),
        send_typing_indicator=AsyncMock(),
    )


_CHAT_ID = "test-user@test-bot"
_STREAM = "mai_gram.bot.conversation_executor.run_with_tools_stream"


@pytest.fixture
async def regen_db(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'regen.db'}")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory.begin() as session:
            session.add(_make_chat(_CHAT_ID, "test-bot"))
        with patch("mai_gram.db.database.get_session_factory", return_value=factory):
            yield factory
    finally:
        await engine.dispose()


def _make_chat(chat_id: str, bot_id: str) -> Chat:
    return Chat(
        id=chat_id,
        user_id="test-user",
        bot_id=bot_id,
        llm_model="test-model",
        system_prompt="test prompt",
    )


@pytest.fixture
def real_regen(
    settings: Settings, tmp_path: Path
) -> tuple[RegenerateService, MagicMock, AssistantTurnBuilder]:
    messenger = _make_messenger()
    llm = MagicMock(count_tokens=AsyncMock(return_value=0))
    builder = AssistantTurnBuilder(
        llm,
        settings,
        build_mcp_manager=lambda *_: MagicMock(),
        memory_data_dir=str(tmp_path),
        wiki_context_limit=20,
        short_term_limit=50,
        test_mode=True,
    )
    executor = ConversationExecutor(
        messenger,
        llm,
        tool_max_iterations=5,
        renderer=ResponseRenderer(messenger, message_logger=MagicMock()),
    )
    return (
        RegenerateService(
            messenger, executor, turn_builder=builder, resolve_chat_id=lambda _: _CHAT_ID
        ),
        messenger,
        builder,
    )


async def _targeted_regen(service: RegenerateService, target_id: int) -> list[str] | None:
    return await service.handle_regenerate(
        _make_message(chat_id="tg-chat", callback_data=f"confirm_regen:{target_id}"),
        previous_response_ids=["old"],
        target_assistant_msg_id=target_id,
    )


async def _history(
    factory: async_sessionmaker[AsyncSession], chat_id: str = _CHAT_ID
) -> list[tuple[int, str, str]]:
    async with factory() as session:
        return [
            (row.id, row.role, row.content) for row in await MessageStore(session).get_all(chat_id)
        ]


class TestRegenerateService:
    async def test_targeted_regen_rolls_back_once_and_keeps_new_turns(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
    ) -> None:
        other_chat_id = "test-user@other-bot"
        async with regen_db.begin() as session:
            session.add(_make_chat(other_chat_id, "other-bot"))
            store = MessageStore(session)
            question = await store.save_message(
                _CHAT_ID, "user", "Original question", show_datetime=False
            )
            target = await store.save_message(_CHAT_ID, "assistant", "Old reply")
            await store.save_message(_CHAT_ID, "user", "Later question")
            await store.save_message(other_chat_id, "assistant", "Other bot")
            later_answer = await store.save_message(_CHAT_ID, "assistant", "Later reply")

        before = await _history(regen_db)
        service, messenger, _ = real_regen
        started = asyncio.Event()
        release = asyncio.Event()

        async def stream(
            _llm: object, _mcp: object, messages: list[ChatMessage], **_: object
        ) -> AsyncIterator[StreamChunk]:
            started.set()
            assert [msg.content for msg in messages[1:]] == ["Original question"]
            await release.wait()
            yield StreamChunk(content="Replacement reply")

        with (
            patch(_STREAM, side_effect=stream) as mock_stream,
        ):
            generation = asyncio.create_task(_targeted_regen(service, target.id))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                assert await _history(regen_db) == before

                async def write_other_chat() -> None:
                    async with regen_db.begin() as session:
                        await MessageStore(session).save_message(
                            other_chat_id, "user", "Concurrent"
                        )

                await asyncio.wait_for(write_other_chat(), timeout=1)
                duplicates = await asyncio.gather(
                    *(_targeted_regen(service, target.id) for _ in range(7))
                )
                assert duplicates == [None] * 7
            finally:
                release.set()
                result = await generation
            assert result == ["42"]
            history = await _history(regen_db)
            replacement = history[-1]
            assert replacement[0] > later_answer.id
            assert replacement[1:] == ("assistant", "Replacement reply")
            assert history == [(question.id, "user", "Original question"), replacement]
            async with regen_db() as session:
                persisted = await session.get(Message, replacement[0])
                assert persisted is not None
                reply_ref = persisted.history_ref
            messenger.build_inline_keyboard.assert_called_once_with(
                [
                    [
                        ("🔄 Regenerate", f"regen:{reply_ref}"),
                        ("✂ Cut this & above", f"cut:{reply_ref}"),
                    ]
                ]
            )

            async with regen_db.begin() as session:
                store = MessageStore(session)
                new_question = await store.save_message(_CHAT_ID, "user", "New question")
                new_reply = await store.save_message(_CHAT_ID, "assistant", "New reply")
            for stale_id in (target.id, later_answer.id):
                assert await _targeted_regen(service, stale_id) is None

        assert await _history(regen_db) == [
            *history,
            (new_question.id, "user", "New question"),
            (new_reply.id, "assistant", "New reply"),
        ]
        assert [
            (role, content) for _, role, content in await _history(regen_db, other_chat_id)
        ] == [
            ("assistant", "Other bot"),
            ("user", "Concurrent"),
        ]
        mock_stream.assert_called_once()
        messenger.delete_message.assert_not_awaited()
        messenger.send_typing_indicator.assert_awaited_once()
        assert "no longer available" in _last_outgoing_text(messenger)

    @pytest.mark.parametrize("action", ["target", "retry"])
    async def test_bound_generation_discards_rows_when_chat_advances(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
        action: str,
    ) -> None:
        async with regen_db.begin() as session:
            store = MessageStore(session)
            pending = await store.save_message(_CHAT_ID, "user", "Question")
            target = None
            if action == "target":
                target = await store.save_message(_CHAT_ID, "assistant", "Original answer")
        before = await _history(regen_db)
        service, messenger, _ = real_regen
        messenger.get_callback_source_message.return_value.message_id = "77"
        started = asyncio.Event()
        release = asyncio.Event()

        async def stream(
            *_: object,
            on_assistant_tool_call: Callable[..., Awaitable[None]],
            on_tool_result: Callable[..., Awaitable[None]],
            **__: object,
        ) -> AsyncIterator[StreamChunk]:
            await on_assistant_tool_call(
                content="", tool_calls=[ToolCall(id="call", name="lookup", arguments="{}")]
            )
            await on_tool_result(
                tool_call_id="call",
                tool_name="lookup",
                arguments="{}",
                result="found",
                content="found",
                error=None,
                server_name=None,
            )
            started.set()
            await release.wait()
            yield StreamChunk(content="Stale answer")

        async def append() -> int:
            async with regen_db.begin() as session:
                row = await MessageStore(session).save_message(_CHAT_ID, "user", "New question")
                return row.id

        with patch(_STREAM, side_effect=stream):
            if target is None:
                generation = service.handle_regenerate(
                    _make_message(chat_id="tg-chat", callback_data="confirm_retry"),
                    previous_response_ids=[],
                    retry_ref=pending.history_ref,
                )
            else:
                generation = _targeted_regen(service, target.id)
            task = asyncio.create_task(generation)
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                assert await _history(regen_db) == before
                new_id = await asyncio.wait_for(append(), timeout=1)
            finally:
                release.set()
                result = await task

        assert result is None
        assert await _history(regen_db) == [*before, (new_id, "user", "New question")]
        assert "no longer available" in _last_outgoing_text(messenger)

    async def test_targeted_regen_allows_writes_during_final_delivery(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
    ) -> None:
        other_chat_id = "test-user@other-bot"
        async with regen_db.begin() as session:
            session.add(_make_chat(other_chat_id, "other-bot"))
            store = MessageStore(session)
            await store.save_message(_CHAT_ID, "user", "Question")
            target = await store.save_message(_CHAT_ID, "assistant", "Original answer")
        before = await _history(regen_db)
        service, messenger, _ = real_regen

        async def stream(*_: object, **__: object) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Replacement")

        async def send(*_: object) -> SendResult:
            assert await _history(regen_db) == before

            async def append() -> None:
                async with regen_db.begin() as session:
                    await MessageStore(session).save_message(other_chat_id, "user", "Concurrent")

            await asyncio.wait_for(append(), timeout=1)
            return SendResult(success=True, message_id="42")

        messenger.send_message.side_effect = send
        with patch(_STREAM, side_effect=stream):
            assert await _targeted_regen(service, target.id) == ["42"]
        assert [(role, content) for _, role, content in await _history(regen_db)] == [
            ("user", "Question"),
            ("assistant", "Replacement"),
        ]

    @pytest.mark.parametrize("history", ["opener", "cut", "cut_tool_call"])
    async def test_targeted_regen_without_visible_input_preserves_history(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
        history: str,
    ) -> None:
        async with regen_db.begin() as session:
            store = MessageStore(session)
            if history != "opener":
                await store.save_message(_CHAT_ID, "user", "Hidden question")
            tool_call = None
            if history == "cut_tool_call":
                tool_call = await store.save_message(
                    _CHAT_ID,
                    "assistant",
                    "",
                    tool_calls=[ToolCall(id="old-tool", name="lookup", arguments="{}")],
                )
                await store.save_message(
                    _CHAT_ID, "tool", "Now orphaned tool result", tool_call_id="old-tool"
                )
            target = await store.save_message(_CHAT_ID, "assistant", "Target answer")
            if history != "opener":
                chat = await session.get(Chat, _CHAT_ID)
                assert chat is not None
                chat.cut_above_message_id = (tool_call or target).id
            await store.save_message(_CHAT_ID, "user", "Later history")

        before = await _history(regen_db)
        service, messenger, _ = real_regen
        with (
            patch(_STREAM) as stream,
        ):
            result = await _targeted_regen(service, target.id)

        assert await _history(regen_db) == before
        assert result is None
        assert "no user message found" in _last_outgoing_text(messenger)
        stream.assert_not_called()

    async def test_targeted_regen_build_failure_preserves_history(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
    ) -> None:
        async with regen_db.begin() as session:
            store = MessageStore(session)
            await store.save_message(_CHAT_ID, "user", "Question")
            target = await store.save_message(_CHAT_ID, "assistant", "Old answer")
            await store.save_message(_CHAT_ID, "user", "Later history")

        before = await _history(regen_db)
        service, _, builder = real_regen
        with (
            patch.object(builder, "build_request", side_effect=OSError("build failed")),
            pytest.raises(OSError, match="build failed"),
        ):
            await _targeted_regen(service, target.id)

        assert await _history(regen_db) == before

    @pytest.mark.parametrize("failure", ["provider", "validation"])
    async def test_targeted_regen_failure_rolls_back_tool_rows_and_can_retry(
        self,
        regen_db: async_sessionmaker[AsyncSession],
        real_regen: tuple[RegenerateService, MagicMock, AssistantTurnBuilder],
        failure: str,
    ) -> None:
        async with regen_db.begin() as session:
            store = MessageStore(session)
            await store.save_message(_CHAT_ID, "user", "Question")
            target = await store.save_message(_CHAT_ID, "assistant", "Old answer")
            await store.save_message(_CHAT_ID, "user", "Later history")

        before = await _history(regen_db)
        service, messenger, _ = real_regen

        async def fail_stream(
            *_: object,
            on_assistant_tool_call: Callable[..., Awaitable[None]],
            on_tool_result: Callable[..., Awaitable[None]],
            **__: object,
        ) -> AsyncIterator[StreamChunk]:
            await on_assistant_tool_call(
                content="", tool_calls=[ToolCall(id="call", name="lookup", arguments="{}")]
            )
            await on_tool_result(
                tool_call_id="call",
                tool_name="lookup",
                arguments="{}",
                result="found",
                content="found",
                error=None,
                server_name=None,
            )
            if failure == "provider":
                raise LLMProviderError("provider failed")
            yield StreamChunk(content="Incomplete", finish_reason="length")

        async def success_stream(*_: object, **__: object) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Replacement")

        with patch(_STREAM, side_effect=fail_stream):
            result = await _targeted_regen(service, target.id)
        assert await _history(regen_db) == before
        assert result is None
        assert "AI provider error" in _last_outgoing_text(messenger)
        messenger.build_inline_keyboard.assert_called_once_with(
            [[("🔄 Regenerate", f"regen:{target.id}")]]
        )
        with patch(_STREAM, side_effect=success_stream):
            assert await _targeted_regen(service, target.id) == ["42"]

        assert [(role, content) for _, role, content in await _history(regen_db)] == [
            ("user", "Question"),
            ("assistant", "Replacement"),
        ]

    @pytest.mark.parametrize("wrong_target", ["user", "tool", "other-user", "other-bot"])
    async def test_targeted_regen_rejects_unrelated_rows(
        self,
        session: AsyncSession,
        wrong_target: str,
    ) -> None:
        chat_id = "test-user@test-bot"
        session.add(
            Chat(
                id=chat_id,
                user_id="test-user",
                bot_id="test-bot",
                llm_model="test-model",
                system_prompt="test prompt",
            )
        )
        target_chat_id = {
            "other-user": "other-user@test-bot",
            "other-bot": "test-user@other-bot",
        }.get(wrong_target, chat_id)
        target = Message(
            chat_id=target_chat_id,
            role=wrong_target if target_chat_id == chat_id else "assistant",
            content="Wrong target",
        )
        later = Message(chat_id=chat_id, role="assistant", content="Keep this")
        session.add_all([target, later])
        await session.commit()
        service, messenger, execute, _ = _make_service()
        with patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await service.handle_regenerate(
                _make_message(chat_id="tg-chat"),
                previous_response_ids=["old"],
                target_assistant_msg_id=target.id,
            )

        assert result is None
        assert await session.get(Message, target.id) is target
        assert await session.get(Message, later.id) is later
        execute.assert_not_awaited()
        messenger.delete_message.assert_not_awaited()

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

        service, messenger, execute, turn_builder = _make_service()

        with patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

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
        turn_builder.build_request.assert_awaited_once()

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

        service, messenger, execute, turn_builder = _make_service()
        execute.return_value = AssistantTurnResult(sent_message_ids=["new-response"])

        with patch("mai_gram.bot.regenerate_service.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

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
        turn_builder.build_request.assert_awaited_once()

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

        service, messenger, execute, turn_builder = _make_service()

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
        turn_builder.build_request.assert_not_awaited()
