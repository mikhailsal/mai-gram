"""Wiki-style knowledge base store backed by markdown files + DB metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import and_, desc, or_, select

from mai_gram.db.models import KnowledgeEntry

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Describes what changed during a disk-to-DB sync."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    db_rows_deleted: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.created) + len(self.updated) + len(self.db_rows_deleted)

    def summary(self) -> str:
        parts: list[str] = []
        if self.created:
            parts.append(f"{len(self.created)} created")
        if self.updated:
            parts.append(f"{len(self.updated)} updated")
        if self.db_rows_deleted:
            parts.append(f"{len(self.db_rows_deleted)} orphaned DB rows removed")
        if self.skipped_files:
            parts.append(f"{len(self.skipped_files)} files skipped")
        return ", ".join(parts) if parts else "no changes needed"


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

    async def list_entries_sorted(
        self,
        chat_id: str,
        *,
        sort_by: str = "importance",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[KnowledgeEntry], int]:
        """Return entries with flexible sorting and pagination.

        Returns a tuple of (entries, total_count).
        sort_by options: "importance" (default), "key", "updated".
        """
        from sqlalchemy import func as sql_func

        count_result = await self._session.execute(
            select(sql_func.count())
            .select_from(KnowledgeEntry)
            .where(KnowledgeEntry.chat_id == chat_id)
        )
        total_count = count_result.scalar() or 0

        query = select(KnowledgeEntry).where(KnowledgeEntry.chat_id == chat_id)

        if sort_by == "key":
            query = query.order_by(KnowledgeEntry.key.asc(), desc(KnowledgeEntry.importance))
        elif sort_by == "updated":
            query = query.order_by(desc(KnowledgeEntry.updated_at), desc(KnowledgeEntry.importance))
        else:
            query = query.order_by(desc(KnowledgeEntry.importance), KnowledgeEntry.key.asc())

        query = query.offset(offset).limit(limit)
        result = await self._session.execute(query)
        return list(result.scalars().all()), total_count

    async def sync_from_disk(self, chat_id: str) -> SyncReport:
        """Synchronise DB rows with .md files on disk (disk is source of truth).

        For each wiki file on disk:
        - If no DB row exists, create one from the file content and filename.
        - If a DB row exists but content or importance differs, update it.

        For each DB row with no corresponding file on disk:
        - Delete the orphaned DB row.

        Returns a SyncReport describing what changed.
        """
        report = SyncReport()
        wiki_dir = self._wiki_dir(chat_id)
        if not wiki_dir.exists():
            db_entries = await self._all_entries_map(chat_id)
            for key, entry in db_entries.items():
                await self._session.delete(entry)
                report.db_rows_deleted.append(key)
            if report.db_rows_deleted:
                await self._session.flush()
            return report

        disk_entries: dict[str, tuple[int, str, Path]] = {}
        for md_file in sorted(wiki_dir.glob("*.md")):
            if md_file.name == "changelog.jsonl":
                continue
            parsed = self._parse_wiki_filename(md_file.name)
            if parsed is None:
                report.skipped_files.append(md_file.name)
                continue
            importance, safe_key = parsed
            content = md_file.read_text(encoding="utf-8")
            disk_entries[safe_key] = (importance, content, md_file)

        db_entries = await self._all_entries_map(chat_id)

        for safe_key, (importance, content, _path) in disk_entries.items():
            db_entry = db_entries.pop(safe_key, None)
            if db_entry is None:
                new_entry = KnowledgeEntry(
                    chat_id=chat_id,
                    category="wiki",
                    key=safe_key,
                    value=content,
                    importance=float(importance),
                )
                self._session.add(new_entry)
                report.created.append(safe_key)
            else:
                changed = False
                if db_entry.value != content:
                    db_entry.value = content
                    changed = True
                if int(db_entry.importance) != importance:
                    db_entry.importance = float(importance)
                    changed = True
                if changed:
                    report.updated.append(safe_key)

        for key, entry in db_entries.items():
            await self._session.delete(entry)
            report.db_rows_deleted.append(key)

        await self._session.flush()
        return report

    async def _all_entries_map(self, chat_id: str) -> dict[str, KnowledgeEntry]:
        """Return all DB entries for a chat_id as a {key: entry} dict."""
        result = await self._session.execute(
            select(KnowledgeEntry).where(KnowledgeEntry.chat_id == chat_id)
        )
        return {entry.key: entry for entry in result.scalars().all()}

    @staticmethod
    def _parse_wiki_filename(filename: str) -> tuple[int, str] | None:
        """Parse ``NNNN_key.md`` into (importance, key), or None if unparseable."""
        if not filename.endswith(".md"):
            return None
        stem = filename[:-3]
        match = re.match(r"^(\d+)_(.+)$", stem)
        if not match:
            return None
        importance = int(match.group(1))
        key = match.group(2)
        return importance, key

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
