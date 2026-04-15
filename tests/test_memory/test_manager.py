"""Tests for MemoryManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.manager import MemoryManager
from mai_gram.memory.messages import MessageStore

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
def manager(session: AsyncSession, tmp_path: object) -> MemoryManager:
    message_store = MessageStore(session)
    wiki_store = WikiStore(session, data_dir=str(tmp_path))
    return MemoryManager(message_store, wiki_store)


class TestMemoryManager:
    async def test_save_and_get_recent(self, manager: MemoryManager, chat: Chat) -> None:
        await manager.save_message(chat.id, "user", "Hello!")
        await manager.save_message(chat.id, "assistant", "Hi there!")
        recent = await manager.get_recent(chat.id, limit=10)
        assert len(recent) == 2

    async def test_search_messages(self, manager: MemoryManager, chat: Chat) -> None:
        await manager.save_message(chat.id, "user", "I love Python programming")
        await manager.save_message(chat.id, "user", "JavaScript is also nice")
        results = await manager.search_messages(chat.id, "Python")
        assert len(results) == 1
