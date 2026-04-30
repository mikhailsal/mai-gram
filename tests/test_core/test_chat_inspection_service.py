"""Tests for the shared chat inspection service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from mai_gram.core.chat_inspection_service import ChatInspectionService
from mai_gram.db.models import Chat, KnowledgeEntry
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


async def _create_chat(session: AsyncSession, chat_id: str) -> None:
    session.add(
        Chat(
            id=chat_id,
            user_id="test-user",
            bot_id="test-bot",
            llm_model="openai/test-model",
            system_prompt="Test prompt",
            timezone="UTC",
        )
    )
    await session.commit()


@pytest.mark.asyncio
async def test_list_history_returns_chronological_messages(session: AsyncSession) -> None:
    await _create_chat(session, "history-chat")
    store = MessageStore(session)
    now = datetime.now(timezone.utc)
    await store.save_message("history-chat", "user", "Hello", timestamp=now)
    await store.save_message(
        "history-chat",
        "assistant",
        "Hi there",
        timestamp=now + timedelta(seconds=1),
    )
    await session.commit()

    history = await ChatInspectionService().list_history(session, chat_id="history-chat")

    assert [item.role for item in history] == ["user", "assistant"]
    assert [item.content for item in history] == ["Hello", "Hi there"]


@pytest.mark.asyncio
async def test_list_wiki_syncs_disk_and_returns_entries(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    await _create_chat(session, "wiki-chat")
    wiki_dir = tmp_path / "wiki-chat" / "wiki"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "900_color.md").write_text("Favorite color: orange", encoding="utf-8")

    result = await ChatInspectionService(data_dir=tmp_path).list_wiki(session, chat_id="wiki-chat")
    await session.commit()

    assert result.sync_report.created == ["color"]
    assert [(entry.key, entry.value) for entry in result.entries] == [
        ("color", "Favorite color: orange")
    ]


@pytest.mark.asyncio
async def test_repair_wiki_reports_removed_orphan_rows(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    await _create_chat(session, "repair-chat")
    session.add(
        KnowledgeEntry(
            chat_id="repair-chat",
            category="wiki",
            key="travel",
            value="Favorite destination: Kyoto.",
            importance=0.7,
        )
    )
    await session.commit()

    report = await ChatInspectionService(data_dir=tmp_path).repair_wiki(
        session,
        chat_id="repair-chat",
    )
    await session.commit()

    assert report.db_rows_deleted == ["travel"]
