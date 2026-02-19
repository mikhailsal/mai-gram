"""Tests for WikiStore."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import Companion, KnowledgeEntry
from mai_companion.memory.knowledge_base import WikiStore


async def _create_companion(session: AsyncSession, companion_id: str = "comp-wiki") -> str:
    companion = Companion(id=companion_id, name="Wiki Companion")
    session.add(companion)
    await session.flush()
    return companion_id


class TestWikiStore:
    """WikiStore behavior."""

    async def test_create_entry_file(self, session: AsyncSession, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 42)
        expected = tmp_path / companion_id / "wiki" / "0042_human_name.md"
        assert expected.exists()

    async def test_create_entry_content(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        created = tmp_path / companion_id / "wiki" / "9999_human_name.md"
        assert created.read_text(encoding="utf-8") == "Alex"

    async def test_create_entry_db_record(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "favorite_city", "Paris", 3000)
        result = await session.execute(
            select(KnowledgeEntry).where(KnowledgeEntry.companion_id == companion_id)
        )
        row = result.scalar_one()
        assert row.key == "favorite_city"
        assert row.value == "Paris"
        assert row.importance == 3000

    async def test_create_duplicate_key_error(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_entry(companion_id, "human_name", "Alice", 9000)

    async def test_edit_entry_content(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        await store.edit_entry(companion_id, "human_name", content="Alice")
        created = tmp_path / companion_id / "wiki" / "9999_human_name.md"
        assert created.read_text(encoding="utf-8") == "Alice"

    async def test_edit_entry_importance_renames_file(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        await store.edit_entry(companion_id, "human_name", importance=9500)

        old_path = tmp_path / companion_id / "wiki" / "9999_human_name.md"
        new_path = tmp_path / companion_id / "wiki" / "9500_human_name.md"
        assert not old_path.exists()
        assert new_path.exists()

    async def test_read_entry(self, session: AsyncSession, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        assert await store.read_entry(companion_id, "human_name") == "Alex"
        assert await store.read_entry(companion_id, "missing_key") is None

    async def test_search_entries_by_key(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        await store.create_entry(companion_id, "favorite_city", "Paris", 5000)

        results = await store.search_entries(companion_id, "human")
        assert [entry.key for entry in results] == ["human_name"]

    async def test_search_entries_by_content(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        await store.create_entry(companion_id, "favorite_city", "Paris", 5000)

        results = await store.search_entries(companion_id, "Alex")
        assert [entry.key for entry in results] == ["human_name"]

    async def test_get_top_entries(self, session: AsyncSession, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "k1", "v1", 100)
        await store.create_entry(companion_id, "k2", "v2", 9000)
        await store.create_entry(companion_id, "k3", "v3", 5000)

        results = await store.get_top_entries(companion_id, limit=2)
        assert [entry.key for entry in results] == ["k2", "k3"]

    async def test_list_entries(self, session: AsyncSession, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "k1", "v1", 100)
        await store.create_entry(companion_id, "k2", "v2", 9000)

        results = await store.list_entries(companion_id)
        assert len(results) == 2
        assert {entry.key for entry in results} == {"k1", "k2"}

    async def test_delete_entry(self, session: AsyncSession, tmp_path: Path) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "human_name", "Alex", 9999)
        path = tmp_path / companion_id / "wiki" / "9999_human_name.md"

        deleted = await store.delete_entry(companion_id, "human_name")
        assert deleted is True
        assert not path.exists()
        assert await store.read_entry(companion_id, "human_name") is None

    async def test_decay_importance(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(companion_id, "human_name", "Alex", 2500)

        updated = await store.decay_importance(companion_id, "human_name", amount=100)
        assert updated is not None
        assert updated.importance == 2400
        assert (tmp_path / companion_id / "wiki" / "2400_human_name.md").exists()
        assert not (tmp_path / companion_id / "wiki" / "2500_human_name.md").exists()

    async def test_key_sanitization(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        entry = await store.create_entry(companion_id, "Human Name! 2026", "Alex", 9999)
        assert entry.key == "human_name_2026"
        assert (tmp_path / companion_id / "wiki" / "9999_human_name_2026.md").exists()

    async def test_edit_restores_missing_file(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        """Ensure edit restores file content from DB if file was missing."""
        companion_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)

        # Create entry
        await store.create_entry(companion_id, "key", "content", 1000)
        file_path = tmp_path / companion_id / "wiki" / "1000_key.md"
        assert file_path.exists()

        # Delete file manually to simulate data loss
        file_path.unlink()
        assert not file_path.exists()

        # Edit importance only (content is None)
        await store.edit_entry(companion_id, "key", importance=2000)

        # New file should exist AND have content restored from DB
        new_path = tmp_path / companion_id / "wiki" / "2000_key.md"
        assert new_path.exists()
        assert new_path.read_text(encoding="utf-8") == "content"
