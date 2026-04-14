"""Tests for WikiStore."""

from __future__ import annotations

import tempfile

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore


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
def store(session: AsyncSession, tmp_path: object) -> WikiStore:
    return WikiStore(session, data_dir=str(tmp_path))


class TestWikiCreate:

    async def test_create_entry(
        self, store: WikiStore, chat: Chat
    ) -> None:
        entry = await store.create_entry(
            chat.id, key="name", content="Alice", importance=9000
        )
        assert entry.key == "name"
        assert entry.value == "Alice"
        assert entry.importance == 9000.0

    async def test_duplicate_key_raises(
        self, store: WikiStore, chat: Chat
    ) -> None:
        await store.create_entry(chat.id, key="name", content="Alice", importance=9000)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_entry(chat.id, key="name", content="Bob", importance=9000)


class TestWikiRead:

    async def test_read_existing(
        self, store: WikiStore, chat: Chat
    ) -> None:
        await store.create_entry(chat.id, key="color", content="Blue", importance=5000)
        content = await store.read_entry(chat.id, "color")
        assert content == "Blue"

    async def test_read_missing(
        self, store: WikiStore, chat: Chat
    ) -> None:
        content = await store.read_entry(chat.id, "nonexistent")
        assert content is None


class TestWikiSearch:

    async def test_search(
        self, store: WikiStore, chat: Chat
    ) -> None:
        await store.create_entry(chat.id, key="pet_cat", content="Has a cat named Whiskers", importance=7000)
        await store.create_entry(chat.id, key="pet_dog", content="Has a dog named Rex", importance=7000)
        results = await store.search_entries(chat.id, "cat")
        assert len(results) >= 1
        assert any("cat" in e.key or "cat" in e.value.lower() for e in results)

    async def test_search_no_results(
        self, store: WikiStore, chat: Chat
    ) -> None:
        results = await store.search_entries(chat.id, "nonexistent")
        assert results == []


class TestWikiTopEntries:

    async def test_top_entries_ordered_by_importance(
        self, store: WikiStore, chat: Chat
    ) -> None:
        await store.create_entry(chat.id, key="low", content="low", importance=100)
        await store.create_entry(chat.id, key="high", content="high", importance=9000)
        await store.create_entry(chat.id, key="mid", content="mid", importance=5000)
        top = await store.get_top_entries(chat.id, limit=2)
        assert len(top) == 2
        assert top[0].key == "high"
