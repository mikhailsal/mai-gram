"""Tests for WikiMCPServer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mai_gram.db.models import Chat
from mai_gram.mcp_servers.wiki_server import WikiMCPServer
from mai_gram.memory.knowledge_base import WikiStore

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


async def _create_companion(session: AsyncSession, chat_id: str = "test@testbot") -> str:
    chat = Chat(
        id=chat_id, user_id="test", bot_id="testbot", llm_model="test/model", system_prompt="test"
    )
    session.add(chat)
    await session.flush()
    return chat_id


class TestWikiMCPServer:
    async def test_list_tools(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        server = WikiMCPServer(WikiStore(session, data_dir=tmp_path), chat_id)

        tools = await server.list_tools()

        assert [tool.name for tool in tools] == [
            "wiki_create",
            "wiki_edit",
            "wiki_read",
            "wiki_search",
        ]

    async def test_call_wiki_create(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        server = WikiMCPServer(WikiStore(session, data_dir=tmp_path), chat_id)

        result = await server.call_tool(
            "wiki_create",
            {"key": "human_name", "content": "Alex", "importance": 9999},
        )

        assert "Created wiki entry 'human_name'" in result
        assert (tmp_path / chat_id / "wiki" / "9999_human_name.md").exists()

    async def test_call_wiki_edit(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "human_name", "Alex", 9999)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_edit", {"key": "human_name", "content": "Alice"})

        assert "Updated wiki entry 'human_name'" in result
        assert await store.read_entry(chat_id, "human_name") == "Alice"

    async def test_call_wiki_read(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "human_name", "Alex", 9999)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_read", {"key": "human_name"})

        assert result == "Alex"

    async def test_call_wiki_read_not_found(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        server = WikiMCPServer(WikiStore(session, data_dir=tmp_path), chat_id)

        result = await server.call_tool("wiki_read", {"key": "missing"})

        assert "not found" in result.lower()

    async def test_call_wiki_search(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "human_name", "Alex", 9999)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_search", {"query": "Alex"})

        assert "human_name (9999): Alex" in result

    async def test_changelog_appends_for_create_and_edit(
        self,
        session: AsyncSession,
        tmp_path: Path,
    ) -> None:
        chat_id = await _create_companion(session)
        server = WikiMCPServer(WikiStore(session, data_dir=tmp_path), chat_id)

        await server.call_tool(
            "wiki_create",
            {"key": "human_name", "content": "Alex", "importance": 9999},
        )
        await server.call_tool(
            "wiki_edit",
            {"key": "human_name", "content": "Alexander", "importance": 9999},
        )

        changelog_path = tmp_path / chat_id / "wiki" / "changelog.jsonl"
        assert changelog_path.exists()
        lines = [
            json.loads(line)
            for line in changelog_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(lines) == 2
        assert lines[0]["action"] == "create"
        assert lines[0]["key"] == "human_name"
        assert lines[1]["action"] == "edit"
        assert lines[1]["content"] == "Alexander"
