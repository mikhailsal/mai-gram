"""Tests for MemoryManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mai_gram.db.models import Chat
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.manager import MemoryManager
from mai_gram.memory.messages import MessageStore

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
def manager(session: AsyncSession, tmp_path: Path) -> MemoryManager:
    message_store = MessageStore(session)
    wiki_store = WikiStore(session, data_dir=str(tmp_path))
    return MemoryManager(message_store, wiki_store)


@pytest.fixture
def wiki_store(session: AsyncSession, tmp_path: Path) -> WikiStore:
    return WikiStore(session, data_dir=str(tmp_path))


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

    async def test_get_wiki_top(
        self, manager: MemoryManager, wiki_store: WikiStore, chat: Chat
    ) -> None:
        await wiki_store.create_entry(chat.id, "hobby", "Playing guitar", 8, category="facts")
        await wiki_store.create_entry(chat.id, "food", "Loves pizza", 5, category="facts")
        await wiki_store.create_entry(chat.id, "color", "Blue", 3, category="facts")

        top = await manager.get_wiki_top(chat.id, limit=2)

        assert len(top) == 2
        assert top[0].importance >= top[1].importance

    async def test_get_wiki_top_empty(self, manager: MemoryManager, chat: Chat) -> None:
        top = await manager.get_wiki_top(chat.id, limit=10)
        assert top == []

    async def test_save_message_with_tool_calls(self, manager: MemoryManager, chat: Chat) -> None:
        from mai_gram.llm.provider import ToolCall

        tool_calls = [ToolCall(id="call_1", name="search", arguments='{"q":"test"}')]
        msg = await manager.save_message(chat.id, "assistant", "", tool_calls=tool_calls)
        assert msg.tool_calls is not None

    async def test_save_message_with_tool_call_id(self, manager: MemoryManager, chat: Chat) -> None:
        msg = await manager.save_message(chat.id, "tool", "result data", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"
