"""Tests for WikiStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore

if TYPE_CHECKING:
    from pathlib import Path

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
        assert int(entry.importance) == 9000

    async def test_duplicate_key_raises(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="name", content="Alice", importance=9000)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_entry(chat.id, key="name", content="Bob", importance=9000)

    async def test_non_positive_importance_raises(self, store: WikiStore, chat: Chat) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            await store.create_entry(chat.id, key="name", content="Alice", importance=0)


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
        assert int(entry.importance) == 9000

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


class TestWikiListSorted:
    async def test_list_entries_sorted_by_importance(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="a", content="A", importance=100)
        await store.create_entry(chat.id, key="b", content="B", importance=200)
        entries, total = await store.list_entries_sorted(chat.id)
        assert total == 2
        assert len(entries) == 2
        assert entries[0].key == "b"

    async def test_list_entries_sorted_empty(self, store: WikiStore, chat: Chat) -> None:
        entries, total = await store.list_entries_sorted(chat.id)
        assert entries == []
        assert total == 0

    async def test_list_entries_sorted_by_key(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="zebra", content="Z", importance=100)
        await store.create_entry(chat.id, key="apple", content="A", importance=9000)
        entries, _ = await store.list_entries_sorted(chat.id, sort_by="key")
        assert entries[0].key == "apple"
        assert entries[1].key == "zebra"

    async def test_list_entries_sorted_with_limit(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="low", content="L", importance=100)
        await store.create_entry(chat.id, key="high", content="H", importance=9000)
        await store.create_entry(chat.id, key="mid", content="M", importance=5000)
        entries, total = await store.list_entries_sorted(chat.id, limit=2)
        assert total == 3
        assert len(entries) == 2
        assert entries[0].key == "high"

    async def test_list_entries_sorted_with_offset(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="low", content="L", importance=100)
        await store.create_entry(chat.id, key="high", content="H", importance=9000)
        entries, total = await store.list_entries_sorted(chat.id, offset=1)
        assert total == 2
        assert len(entries) == 1
        assert entries[0].key == "low"


class TestWikiDecay:
    async def test_decay_reduces_importance(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="fact", content="Test", importance=500)
        result = await store.decay_importance(chat.id, "fact", amount=100)
        assert result is not None
        assert int(result.importance) == 400

    async def test_decay_to_zero_deletes(self, store: WikiStore, chat: Chat) -> None:
        await store.create_entry(chat.id, key="weak", content="Gone", importance=50)
        result = await store.decay_importance(chat.id, "weak", amount=50)
        assert result is None
        content = await store.read_entry(chat.id, "weak")
        assert content is None

    async def test_decay_nonexistent(self, store: WikiStore, chat: Chat) -> None:
        result = await store.decay_importance(chat.id, "nope")
        assert result is None

    async def test_decay_with_non_positive_amount_raises(
        self,
        store: WikiStore,
        chat: Chat,
    ) -> None:
        await store.create_entry(chat.id, key="fact", content="Test", importance=500)

        with pytest.raises(ValueError, match="positive integer"):
            await store.decay_importance(chat.id, "fact", amount=0)


class TestWikiSanitizeKey:
    async def test_empty_key_raises(self, store: WikiStore, chat: Chat) -> None:
        with pytest.raises(ValueError, match="empty after sanitization"):
            await store.create_entry(chat.id, key="!!!", content="x", importance=1)


class TestWikiSync:
    async def test_sync_creates_db_rows_from_disk(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        wiki_dir = tmp_path / chat.id / "wiki"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "9999_human_name.md").write_text("Alice", encoding="utf-8")
        (wiki_dir / "5000_favorite_color.md").write_text("Blue", encoding="utf-8")

        report = await store.sync_from_disk(chat.id)

        assert len(report.created) == 2
        assert "human_name" in report.created
        assert "favorite_color" in report.created
        _entries, total = await store.list_entries_sorted(chat.id)
        assert total == 2

    async def test_sync_removes_orphaned_db_rows(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        await store.create_entry(chat.id, key="old_fact", content="Gone", importance=1000)
        wiki_dir = tmp_path / chat.id / "wiki"
        (wiki_dir / "1000_old_fact.md").unlink()

        report = await store.sync_from_disk(chat.id)

        assert "old_fact" in report.db_rows_deleted
        _entries, total = await store.list_entries_sorted(chat.id)
        assert total == 0

    async def test_sync_updates_content_from_disk(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        await store.create_entry(chat.id, key="fact", content="Old text", importance=5000)
        wiki_dir = tmp_path / chat.id / "wiki"
        (wiki_dir / "5000_fact.md").write_text("New text from disk", encoding="utf-8")

        report = await store.sync_from_disk(chat.id)

        assert "fact" in report.updated
        content = await store.read_entry(chat.id, "fact")
        assert content == "New text from disk"

    async def test_sync_updates_importance_from_filename(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        await store.create_entry(chat.id, key="fact", content="Text", importance=5000)
        wiki_dir = tmp_path / chat.id / "wiki"
        (wiki_dir / "5000_fact.md").rename(wiki_dir / "9000_fact.md")

        report = await store.sync_from_disk(chat.id)

        assert "fact" in report.updated
        entries, _ = await store.list_entries_sorted(chat.id)
        assert int(entries[0].importance) == 9000

    async def test_sync_noop_when_already_in_sync(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        await store.create_entry(chat.id, key="fact", content="Text", importance=5000)

        report = await store.sync_from_disk(chat.id)

        assert report.total_changes == 0

    async def test_sync_no_wiki_dir_cleans_db(
        self, store: WikiStore, chat: Chat, session: AsyncSession
    ) -> None:
        from mai_gram.db.models import KnowledgeEntry

        entry = KnowledgeEntry(
            chat_id=chat.id, category="wiki", key="orphan", value="x", importance=1.0
        )
        session.add(entry)
        await session.flush()

        report = await store.sync_from_disk(chat.id)

        assert "orphan" in report.db_rows_deleted

    async def test_sync_skips_non_wiki_files(
        self, store: WikiStore, chat: Chat, tmp_path: Path
    ) -> None:
        wiki_dir = tmp_path / chat.id / "wiki"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "readme.md").write_text("not a wiki file", encoding="utf-8")
        (wiki_dir / "5000_valid.md").write_text("Valid", encoding="utf-8")

        report = await store.sync_from_disk(chat.id)

        assert len(report.created) == 1
        assert "valid" in report.created
        assert "readme.md" in report.skipped_files
