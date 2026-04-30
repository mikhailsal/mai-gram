"""Shared history and wiki inspection services for CLI and other adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mai_gram.memory.knowledge_base import SyncReport, WikiStore
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class HistoryMessageView:
    """Transport-neutral view of a stored chat message."""

    role: str
    content: str
    timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class WikiEntryView:
    """Transport-neutral view of a wiki entry."""

    key: str
    value: str
    importance: int


@dataclass(frozen=True, slots=True)
class WikiInspectionResult:
    """Wiki listing plus any sync report produced while loading it."""

    entries: list[WikiEntryView]
    sync_report: SyncReport


class ChatInspectionService:
    """Provide adapter-neutral history and wiki inspection workflows."""

    def __init__(self, *, data_dir: str | Path = "./data") -> None:
        self._data_dir = Path(data_dir)

    async def list_history(
        self,
        session: AsyncSession,
        *,
        chat_id: str,
    ) -> list[HistoryMessageView]:
        """Return all stored messages for a chat in chronological order."""
        messages = await MessageStore(session).get_all(chat_id)
        return [
            HistoryMessageView(
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
            )
            for message in messages
        ]

    async def list_wiki(
        self,
        session: AsyncSession,
        *,
        chat_id: str,
    ) -> WikiInspectionResult:
        """Sync wiki entries from disk and return the sorted listing.

        ``sync_from_disk`` may change DB rows; callers should commit when
        ``result.sync_report.total_changes > 0`` to persist reconciliation.
        """
        wiki_store = WikiStore(session, data_dir=self._data_dir)
        sync_report = await wiki_store.sync_from_disk(chat_id)
        entries, _ = await wiki_store.list_entries_sorted(chat_id)
        return WikiInspectionResult(
            entries=[
                WikiEntryView(
                    key=entry.key,
                    value=entry.value,
                    importance=int(entry.importance),
                )
                for entry in entries
            ],
            sync_report=sync_report,
        )

    async def repair_wiki(
        self,
        session: AsyncSession,
        *,
        chat_id: str,
    ) -> SyncReport:
        """Sync wiki entries from disk and return the applied repair report.

        This is an explicit mutating workflow; callers should commit after
        invocation to persist any created/updated/deleted rows.
        """
        wiki_store = WikiStore(session, data_dir=self._data_dir)
        return await wiki_store.sync_from_disk(chat_id)
