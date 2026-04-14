"""Tests for MessageStore."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mai_gram.db.models import Chat
from mai_gram.memory.messages import MessageStore


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

    async def test_basic_save(
        self, store: MessageStore, chat: Chat, session: AsyncSession
    ) -> None:
        msg = await store.save_message(chat.id, "user", "Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.chat_id == chat.id

    async def test_save_with_timestamp(
        self, store: MessageStore, chat: Chat
    ) -> None:
        ts = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        msg = await store.save_message(chat.id, "user", "Hello!", timestamp=ts)
        assert msg.timestamp == ts

    async def test_rejects_out_of_order_timestamp(
        self, store: MessageStore, chat: Chat
    ) -> None:
        ts1 = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 15, 11, 0, tzinfo=timezone.utc)
        await store.save_message(chat.id, "user", "First", timestamp=ts1)
        with pytest.raises(ValueError, match="not after"):
            await store.save_message(chat.id, "user", "Second", timestamp=ts2)

    async def test_save_with_tool_calls(
        self, store: MessageStore, chat: Chat
    ) -> None:
        msg = await store.save_message(
            chat.id,
            "assistant",
            "Checking...",
            tool_calls='[{"id":"tc1","name":"search","arguments":"{}"}]',
        )
        assert msg.tool_calls is not None


class TestGetRecent:

    async def test_returns_recent(
        self, store: MessageStore, chat: Chat
    ) -> None:
        for i in range(5):
            await store.save_message(chat.id, "user", f"msg {i}")
        recent = await store.get_recent(chat.id, limit=3)
        assert len(recent) == 3

    async def test_empty_chat(self, store: MessageStore, chat: Chat) -> None:
        recent = await store.get_recent(chat.id)
        assert recent == []


class TestSearch:

    async def test_basic_search(
        self, store: MessageStore, chat: Chat
    ) -> None:
        await store.save_message(chat.id, "user", "I like Python")
        await store.save_message(chat.id, "user", "I like JavaScript")
        results = await store.search(chat.id, "Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_no_results(
        self, store: MessageStore, chat: Chat
    ) -> None:
        await store.save_message(chat.id, "user", "Hello")
        results = await store.search(chat.id, "nonexistent")
        assert results == []
