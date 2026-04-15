"""Tests for WikiStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore

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
def store(session: AsyncSession, tmp_path: object) -> WikiStore:
    return WikiStore(session, data_dir=str(tmp_path))


class TestWikiCreate:
    async def test_create_entry(self, store: WikiStore, chat: Chat) -> None:
        entry = await store.create_entry(chat.id, key="name", content="Alice", importance=9000)
        assert entry.key == "name"
        assert entry.value == "Alice"
        assert entry.importance == 9000.0

    async def test_duplicate_key_raises(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="name", content="Alice", importance=9000)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_entry(chat.id, key="name", content="Bob", importance=9000)


class TestWikiRead:
    async def test_read_existing(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="color", content="Blue", importance=5000)
        content = await store.read_entry(chat.id, "color")
        assert content == "Blue"

    async def test_read_missing(self, store: WikiStore, chat: Chat) -> None:
        content = await store.read_entry(chat.id, "nonexistent")
        assert content is None


class TestWikiSearch:
    async def test_search(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(
            chat.id, key="pet_cat", content="Has a cat named Whiskers", importance=7000
        )
        await store.create_entry(
            chat.id, key="pet_dog", content="Has a dog named Rex", importance=7000
        )
        results = await store.search_entries(chat.id, "cat")
        assert len(results) >= 1
        assert any("cat" in e.key or "cat" in e.value.lower() for e in results)

    async def test_search_no_results(self, store: WikiStore, chat: Chat) -> None:
        results = await store.search_entries(chat.id, "nonexistent")
        assert results == []


class TestWikiEdit:
    async def test_edit_content(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="color", content="Blue", importance=5000)
        entry = await store.edit_entry(chat.id, "color", content="Red")
        assert entry is not None
        assert entry.value == "Red"

    async def test_edit_importance(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="city", content="Paris", importance=5000)
        entry = await store.edit_entry(chat.id, "city", importance=9000)
        assert entry is not None
        assert entry.importance == 9000.0

    async def test_edit_nonexistent_returns_none(self, store: WikiStore, chat: Chat) -> None:
        result = await store.edit_entry(chat.id, "nope", content="x")
        assert result is None


class TestWikiDelete:
    async def test_delete_existing(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="temp", content="bye", importance=100)
        deleted = await store.delete_entry(chat.id, "temp")
        assert deleted is True
        content = await store.read_entry(chat.id, "temp")
        assert content is None

    async def test_delete_nonexistent(self, store: WikiStore, chat: Chat) -> None:
        deleted = await store.delete_entry(chat.id, "nope")
        assert deleted is False


class TestWikiList:
    async def test_list_entries(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="a", content="A", importance=100)
        await store.create_entry(chat.id, key="b", content="B", importance=200)
        entries = await store.list_entries(chat.id)
        assert len(entries) == 2

    async def test_list_empty(self, store: WikiStore, chat: Chat) -> None:
        entries = await store.list_entries(chat.id)
        assert entries == []


class TestWikiDecay:
    async def test_decay_reduces_importance(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="fact", content="Test", importance=500)
        result = await store.decay_importance(chat.id, "fact", amount=100)
        assert result is not None
        assert result.importance == 400.0

    async def test_decay_to_zero_deletes(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="weak", content="Gone", importance=50)
        result = await store.decay_importance(chat.id, "weak", amount=50)
        assert result is None
        content = await store.read_entry(chat.id, "weak")
        assert content is None

    async def test_decay_nonexistent(self, store: WikiStore, chat: Chat) -> None:
        result = await store.decay_importance(chat.id, "nope")
        assert result is None


class TestWikiSanitizeKey:
    async def test_empty_key_raises(self, store: WikiStore, chat: Chat) -> None:
        with pytest.raises(ValueError, match="empty after sanitization"):
            await store.create_entry(chat.id, key="!!!", content="x", importance=1)


class TestWikiTopEntries:
    async def test_top_entries_ordered_by_importance(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="low", content="low", importance=100)
        await store.create_entry(chat.id, key="high", content="high", importance=9000)
        await store.create_entry(chat.id, key="mid", content="mid", importance=5000)
        top = await store.get_top_entries(chat.id, limit=2)
        assert len(top) == 2
        assert top[0].key == "high"
