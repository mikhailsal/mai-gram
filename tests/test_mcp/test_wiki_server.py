"""Tests for WikiMCPServer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mai_gram.db.models import Chat
from mai_gram.mcp_servers.wiki_server import WikiMCPServer, _content_preview
from mai_gram.memory.knowledge_base import WikiStore

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


class TestContentPreview:
    """Test the _content_preview helper."""

    def test_short_content_unchanged(self) -> None:
        assert _content_preview("Hello world") == "Hello world"

    def test_long_single_line_truncated(self) -> None:
        text = "A" * 200
        result = _content_preview(text)
        assert len(result) == 121  # 120 + "…"
        assert result.endswith("…")

    def test_multiline_uses_first_line(self) -> None:
        text = "First line\nSecond line\nThird line"
        assert _content_preview(text) == "First line"

    def test_custom_max_chars(self) -> None:
        text = "A" * 50
        result = _content_preview(text, max_chars=10)
        assert result == "A" * 10 + "…"

    def test_empty_string(self) -> None:
        assert _content_preview("") == ""

    def test_first_line_long_truncated(self) -> None:
        text = "B" * 200 + "\nshort second line"
        result = _content_preview(text)
        assert result == "B" * 120 + "…"


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
            "wiki_list",
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

    async def test_call_wiki_search_truncates_long_content(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        long_content = "Searchable " + "X" * 500
        await store.create_entry(chat_id, "long_entry", long_content, 5000)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_search", {"query": "Searchable"})

        assert "long_entry (5000):" in result
        assert "X" * 500 not in result
        assert "…" in result

    async def test_call_wiki_list_empty(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        server = WikiMCPServer(WikiStore(session, data_dir=tmp_path), chat_id)

        result = await server.call_tool("wiki_list", {})

        assert "No wiki entries found" in result

    async def test_call_wiki_list_returns_entries(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "human_name", "Alex", 9999)
        await store.create_entry(chat_id, "favorite_color", "Blue", 5000)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_list", {})

        assert "1-2 of 2 total" in result
        assert "[9999] human_name: Alex" in result
        assert "[5000] favorite_color: Blue" in result

    async def test_call_wiki_list_truncates_long_content(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        long_content = "A" * 500
        await store.create_entry(chat_id, "long_entry", long_content, 5000)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_list", {})

        assert "[5000] long_entry:" in result
        assert "A" * 500 not in result
        assert "…" in result

    async def test_call_wiki_list_sorted_by_key(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "zebra", "A zebra", 100)
        await store.create_entry(chat_id, "apple", "An apple", 9000)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_list", {"sort_by": "key"})

        lines = result.strip().split("\n")
        entry_lines = [line for line in lines if line.startswith("[")]
        assert entry_lines[0].startswith("[9000] apple")
        assert entry_lines[1].startswith("[100] zebra")

    async def test_call_wiki_list_pagination(self, session: AsyncSession, tmp_path: Path) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        for i in range(5):
            await store.create_entry(chat_id, f"entry_{i}", f"Content {i}", 1000 + i)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_list", {"limit": 2, "offset": 0})
        assert "1-2 of 5 total" in result
        assert "offset=2" in result

        result2 = await server.call_tool("wiki_list", {"limit": 2, "offset": 4})
        assert "5-5 of 5 total" in result2

    async def test_call_wiki_list_offset_past_end(
        self, session: AsyncSession, tmp_path: Path
    ) -> None:
        chat_id = await _create_companion(session)
        store = WikiStore(session, data_dir=tmp_path)
        await store.create_entry(chat_id, "only_entry", "Test", 5000)
        server = WikiMCPServer(store, chat_id)

        result = await server.call_tool("wiki_list", {"offset": 10})
        assert "No more wiki entries" in result

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
