"""Wiki-style knowledge base store backed by markdown files + DB metadata."""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mai_gram.db.models import KnowledgeEntry


def _escape_like_pattern(query: str) -> str:
    """Escape SQL LIKE wildcard characters (%, _) in user input."""
    return query.replace("%", r"\%").replace("_", r"\_")


class WikiStore:
    """File-backed knowledge entry store with DB metadata."""

    def __init__(self, session: AsyncSession, data_dir: str | Path = "./data") -> None:
        self._session = session
        self._data_dir = Path(data_dir)

    @property
    def data_dir(self) -> Path:
        """Expose the configured wiki base directory."""
        return self._data_dir

    async def create_entry(
        self,
        chat_id: str,
        key: str,
        content: str,
        importance: int | float,
        *,
        category: str = "wiki",
    ) -> KnowledgeEntry:
        """Create a new wiki entry on disk and in the database."""
        safe_key = self._sanitize_key(key)
        existing = await self._get_entry(chat_id, safe_key)
        if existing is not None:
            raise ValueError(f"Wiki entry with key '{safe_key}' already exists")

        wiki_dir = self._wiki_dir(chat_id)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        target_file = wiki_dir / self._filename(importance, safe_key)
        if target_file.exists():
            raise ValueError(f"Wiki file already exists for key '{safe_key}'")

        target_file.write_text(content, encoding="utf-8")
        entry = KnowledgeEntry(
            chat_id=chat_id,
            category=category,
            key=safe_key,
            value=content,
            importance=float(importance),
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def edit_entry(
        self,
        chat_id: str,
        key: str,
        *,
        content: str | None = None,
        importance: int | float | None = None,
    ) -> KnowledgeEntry | None:
        """Edit an existing wiki entry's content and/or importance."""
        safe_key = self._sanitize_key(key)
        entry = await self._get_entry(chat_id, safe_key)
        if entry is None:
            return None

        current_file = self._find_entry_file(chat_id, safe_key)
        if content is not None:
            entry.value = content
        if importance is not None:
            entry.importance = float(importance)

        wiki_dir = self._wiki_dir(chat_id)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        target_file = wiki_dir / self._filename(entry.importance, safe_key)

        if current_file is not None and current_file != target_file:
            current_file.rename(target_file)
        elif current_file is None:
            # File was missing, so we must write content regardless
            pass
            
        if content is not None:
             target_file.write_text(content, encoding="utf-8")
        elif not target_file.exists():
             # If file didn't exist (and we didn't rename one to here), restore from DB value
             target_file.write_text(entry.value, encoding="utf-8")

        await self._session.flush()
        return entry

    async def read_entry(self, chat_id: str, key: str) -> str | None:
        """Read an entry by key, preferring file content over DB value."""
        safe_key = self._sanitize_key(key)
        file_path = self._find_entry_file(chat_id, safe_key)
        if file_path is not None and file_path.exists():
            return file_path.read_text(encoding="utf-8")

        entry = await self._get_entry(chat_id, safe_key)
        if entry is None:
            return None
        return entry.value

    async def search_entries(
        self,
        chat_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[KnowledgeEntry]:
        """Search entries by key or content."""
        escaped = _escape_like_pattern(query)
        pattern = f"%{escaped}%"
        result = await self._session.execute(
            select(KnowledgeEntry)
            .where(
                and_(
                    KnowledgeEntry.chat_id == chat_id,
                    or_(
                        KnowledgeEntry.key.like(pattern, escape="\\"),
                        KnowledgeEntry.value.like(pattern, escape="\\"),
                    ),
                )
            )
            .order_by(desc(KnowledgeEntry.importance), desc(KnowledgeEntry.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_top_entries(
        self,
        chat_id: str,
        *,
        limit: int = 20,
    ) -> list[KnowledgeEntry]:
        """Return highest-importance entries first."""
        result = await self._session.execute(
            select(KnowledgeEntry)
            .where(KnowledgeEntry.chat_id == chat_id)
            .order_by(desc(KnowledgeEntry.importance), desc(KnowledgeEntry.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_entries(self, chat_id: str) -> list[KnowledgeEntry]:
        """Return all entries for one companion."""
        result = await self._session.execute(
            select(KnowledgeEntry)
            .where(KnowledgeEntry.chat_id == chat_id)
            .order_by(desc(KnowledgeEntry.importance), KnowledgeEntry.key.asc())
        )
        return list(result.scalars().all())

    async def delete_entry(self, chat_id: str, key: str) -> bool:
        """Delete an entry from both disk and DB."""
        safe_key = self._sanitize_key(key)
        entry = await self._get_entry(chat_id, safe_key)
        file_path = self._find_entry_file(chat_id, safe_key)

        if file_path is not None and file_path.exists():
            file_path.unlink()
        if entry is not None:
            await self._session.delete(entry)
            await self._session.flush()
            return True
        return False

    async def decay_importance(
        self,
        chat_id: str,
        key: str,
        *,
        amount: int | float = 100,
    ) -> KnowledgeEntry | None:
        """Decrease importance and rename file accordingly.

        If importance reaches 0 or below, the entry is deleted and None is returned.
        """
        safe_key = self._sanitize_key(key)
        entry = await self._get_entry(chat_id, safe_key)
        if entry is None:
            return None

        new_importance = entry.importance - float(amount)
        if new_importance <= 0:
            await self.delete_entry(chat_id, safe_key)
            return None

        return await self.edit_entry(chat_id, safe_key, importance=new_importance)

    async def _get_entry(self, chat_id: str, safe_key: str) -> KnowledgeEntry | None:
        result = await self._session.execute(
            select(KnowledgeEntry).where(
                and_(
                    KnowledgeEntry.chat_id == chat_id,
                    KnowledgeEntry.key == safe_key,
                )
            )
        )
        return result.scalar_one_or_none()

    def _wiki_dir(self, chat_id: str) -> Path:
        return self._data_dir / chat_id / "wiki"

    def _find_entry_file(self, chat_id: str, safe_key: str) -> Path | None:
        wiki_dir = self._wiki_dir(chat_id)
        if not wiki_dir.exists():
            return None
        matches = sorted(wiki_dir.glob(f"*_{safe_key}.md"))
        if not matches:
            return None
        return matches[0]

    def _filename(self, importance: int | float, safe_key: str) -> str:
        return f"{int(float(importance)):04d}_{safe_key}.md"

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Sanitize a wiki key for use as a filename.

        Preserves Unicode letters (Cyrillic, CJK, etc.) while removing
        characters that are problematic for filesystems.
        """
        # Strip and lowercase
        normalized = key.strip().lower()

        # Replace filesystem-unsafe characters with underscores.
        # Keep: Unicode letters (\w includes them), digits, underscores, hyphens.
        # The \w pattern in Python regex with no ASCII flag matches Unicode letters.
        normalized = re.sub(r"[^\w\-]+", "_", normalized, flags=re.UNICODE)

        # Collapse multiple underscores and strip leading/trailing underscores
        normalized = re.sub(r"_+", "_", normalized).strip("_")

        if not normalized:
            raise ValueError("Wiki key becomes empty after sanitization")
        return normalized
