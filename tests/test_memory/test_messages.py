"""Tests for MessageStore."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.llm.provider import MessageRole, ToolCall
from mai_gram.memory.messages import (
    MessageStore,
    decode_persisted_message,
    parse_persisted_tool_calls,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def chat(session: AsyncSession) -> Chat:
    chat = Chat(
        id="test-user@testbot",
        user_id="test-user",
        bot_id="testbot",
        llm_model="openai/gpt-4o-mini",
        system_prompt="test",
    )
    session.add(chat)
    await session.flush()
    return chat


@pytest.fixture
def store(session: AsyncSession) -> MessageStore:
    return MessageStore(session)


class TestSaveMessage:
    async def test_basic_save(self, store: MessageStore, chat: Chat, session: AsyncSession) -> None:
        msg = await store.save_message(chat.id, "user", "Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.chat_id == chat.id

    async def test_save_with_timestamp(self, store: MessageStore, chat: Chat) -> None:
        ts = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        msg = await store.save_message(chat.id, "user", "Hello!", timestamp=ts)
        assert msg.timestamp == ts

    async def test_rejects_out_of_order_timestamp(self, store: MessageStore, chat: Chat) -> None:
        ts1 = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)
        await store.save_message(chat.id, "user", "First", timestamp=ts1)
        with pytest.raises(ValueError, match="not after"):
            await store.save_message(chat.id, "user", "Second", timestamp=ts2)

    async def test_save_with_tool_calls(self, store: MessageStore, chat: Chat) -> None:
        msg = await store.save_message(
            chat.id,
            MessageRole.ASSISTANT,
            "Checking...",
            tool_calls=[ToolCall(id="tc1", name="search", arguments="{}")],
        )
        assert msg.tool_calls is not None
        assert "search" in msg.tool_calls

    async def test_decode_persisted_message_round_trips_tool_calls(
        self,
        store: MessageStore,
        chat: Chat,
    ) -> None:
        saved = await store.save_message(
            chat.id,
            MessageRole.ASSISTANT,
            "Checking...",
            tool_calls=[ToolCall(id="tc1", name="search", arguments='{"q":"cats"}')],
            reasoning="thinking",
        )

        decoded = decode_persisted_message(saved)

        assert decoded.role == MessageRole.ASSISTANT
        assert decoded.tool_calls == [ToolCall(id="tc1", name="search", arguments='{"q":"cats"}')]
        assert decoded.reasoning == "thinking"

    def test_parse_persisted_tool_calls_returns_none_for_invalid_json(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        assert parse_persisted_tool_calls("not-json", message_id=7) is None
        assert "Failed to parse tool_calls for message 7" in caplog.text


class TestGetRecent:
    async def test_returns_recent(self, store: MessageStore, chat: Chat) -> None:
        for i in range(5):
            await store.save_message(chat.id, "user", f"msg {i}")
        recent = await store.get_recent(chat.id, limit=3)
        assert len(recent) == 3

    async def test_empty_chat(self, store: MessageStore, chat: Chat) -> None:
        recent = await store.get_recent(chat.id)
        assert recent == []


class TestGetAll:
    async def test_get_all_returns_all(self, store: MessageStore, chat: Chat) -> None:
        await store.save_message(chat.id, "user", "First")
        await store.save_message(chat.id, "assistant", "Second")
        all_msgs = await store.get_all(chat.id)
        assert len(all_msgs) == 2


class TestGetDatesWithMessages:
    async def test_returns_dates(self, store: MessageStore, chat: Chat) -> None:
        ts1 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
        await store.save_message(chat.id, "user", "Day 1", timestamp=ts1)
        await store.save_message(chat.id, "user", "Day 2", timestamp=ts2)
        dates = await store.get_dates_with_messages(chat.id)
        assert len(dates) == 2

    async def test_before_date_filter(self, store: MessageStore, chat: Chat) -> None:
        from datetime import date

        ts1 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 15, 14, 0, tzinfo=timezone.utc)
        await store.save_message(chat.id, "user", "Earlier", timestamp=ts1)
        await store.save_message(chat.id, "user", "Later", timestamp=ts2)
        dates = await store.get_dates_with_messages(chat.id, before_date=date(2026, 3, 12))
        assert len(dates) == 1


class TestSearch:
    async def test_basic_search(self, store: MessageStore, chat: Chat) -> None:
        await store.save_message(chat.id, "user", "I like Python")
        await store.save_message(chat.id, "user", "I like JavaScript")
        results = await store.search(chat.id, "Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_no_results(self, store: MessageStore, chat: Chat) -> None:
        await store.save_message(chat.id, "user", "Hello")
        results = await store.search(chat.id, "nonexistent")
        assert results == []


class TestGetMessageById:
    async def test_get_existing(self, store: MessageStore, chat: Chat) -> None:
        msg = await store.save_message(chat.id, "user", "Target")
        found = await store.get_message_by_id(chat.id, msg.id)
        assert found is not None
        assert found.content == "Target"

    async def test_get_nonexistent(self, store: MessageStore, chat: Chat) -> None:
        found = await store.get_message_by_id(chat.id, 99999)
        assert found is None


class TestGetMessageContext:
    async def test_context_around_message(self, store: MessageStore, chat: Chat) -> None:
        msgs = []
        for i in range(5):
            m = await store.save_message(chat.id, "user", f"msg-{i}")
            msgs.append(m)
        before, target, after = await store.get_message_context(
            chat.id, msgs[2].id, before=1, after=1
        )
        assert target is not None
        assert target.content == "msg-2"
        assert len(before) == 1
        assert len(after) == 1


class TestGetMessagesPaginated:
    async def test_pagination(self, store: MessageStore, chat: Chat) -> None:
        for i in range(10):
            await store.save_message(chat.id, "user", f"msg-{i}")
        messages, total = await store.get_messages_paginated(chat.id, limit=5, offset=0)
        assert len(messages) == 5
        assert total == 10
