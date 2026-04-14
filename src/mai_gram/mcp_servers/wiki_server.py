"""MCP wiki server backed by WikiStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.mcp_servers.messages_server import MCPToolSpec


class WikiMCPServer:
    """Expose wiki CRUD/search tools for MCP-style tool calling."""

    def __init__(self, store: WikiStore, chat_id: str) -> None:
        self._store = store
        self._chat_id = chat_id

    def _append_changelog(self, action: str, payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        entry: dict[str, Any] = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "action": action,
        }
        for key in ("key", "content", "importance"):
            if key in payload and payload[key] is not None:
                entry[key] = payload[key]

        changelog_path = (
            Path(self._store.data_dir) / self._chat_id / "wiki" / "changelog.jsonl"
        )
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        with changelog_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def list_tools(self) -> list[MCPToolSpec]:
        """Return tools exposed by this server."""
        return [
            MCPToolSpec(
                name="wiki_create",
                description=(
                    "Save a new piece of knowledge to your personal wiki. Use this when you "
                    "learn something important about your human or want to remember something. "
                    "Choose importance carefully: 9999 = human's name, 9000+ = family/close "
                    "relationships, 8000+ = major life events, 7000+ = important dates, "
                    "5000+ = strong preferences and hobbies, 3000+ = casual preferences, "
                    "1000+ = minor details."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "A short descriptive title for the entry",
                        },
                        "content": {
                            "type": "string",
                            "description": "The knowledge content to store",
                        },
                        "importance": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 9999,
                            "description": "Importance score (see tool description for scale)",
                        },
                    },
                    "required": ["key", "content", "importance"],
                    "additionalProperties": False,
                },
            ),
            MCPToolSpec(
                name="wiki_edit",
                description=(
                    "Update an existing wiki entry with new or corrected information. "
                    "Use this to fix mistakes or add details to existing knowledge."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key of the entry to edit",
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (optional if only changing importance)",
                        },
                        "importance": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 9999,
                            "description": "New importance score (optional)",
                        },
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            ),
            MCPToolSpec(
                name="wiki_read",
                description="Read a specific entry from your wiki by its key.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key of the entry to read",
                        },
                    },
                    "required": ["key"],
                    "additionalProperties": False,
                },
            ),
            MCPToolSpec(
                name="wiki_search",
                description=(
                    "Search your wiki for entries matching a query. "
                    "Searches both keys and content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 20,
                            "description": "Maximum number of results",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a named wiki tool."""
        if tool_name == "wiki_create":
            result = await self._call_create(arguments)
            self._append_changelog("create", arguments)
            return result
        if tool_name == "wiki_edit":
            result = await self._call_edit(arguments)
            if "not found" not in result.lower():
                self._append_changelog("edit", arguments)
            return result
        if tool_name == "wiki_read":
            return await self._call_read(arguments)
        if tool_name == "wiki_search":
            return await self._call_search(arguments)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _call_create(self, arguments: dict[str, Any]) -> str:
        key = arguments.get("key")
        content = arguments.get("content")
        importance = arguments.get("importance")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("wiki_create requires non-empty string 'key'")
        if not isinstance(content, str):
            raise ValueError("wiki_create requires string 'content'")
        if not isinstance(importance, int):
            raise ValueError("wiki_create requires integer 'importance'")

        created = await self._store.create_entry(
            self._chat_id,
            key=key,
            content=content,
            importance=importance,
        )
        return f"Created wiki entry '{created.key}' with importance {int(created.importance)}."

    async def _call_edit(self, arguments: dict[str, Any]) -> str:
        key = arguments.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("wiki_edit requires non-empty string 'key'")

        content = arguments.get("content")
        importance = arguments.get("importance")
        if content is not None and not isinstance(content, str):
            raise ValueError("wiki_edit optional 'content' must be a string")
        if importance is not None and not isinstance(importance, int):
            raise ValueError("wiki_edit optional 'importance' must be an integer")
        if content is None and importance is None:
            raise ValueError("wiki_edit requires 'content' and/or 'importance'")

        updated = await self._store.edit_entry(
            self._chat_id,
            key=key,
            content=content,
            importance=importance,
        )
        if updated is None:
            return f"Wiki entry '{key}' not found."

        return f"Updated wiki entry '{updated.key}' with importance {int(updated.importance)}."

    async def _call_read(self, arguments: dict[str, Any]) -> str:
        key = arguments.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("wiki_read requires non-empty string 'key'")

        content = await self._store.read_entry(self._chat_id, key)
        if content is None:
            return f"Wiki entry '{key}' not found."
        return content

    async def _call_search(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("wiki_search requires non-empty string 'query'")

        raw_limit = arguments.get("limit", 20)
        if not isinstance(raw_limit, int):
            raise ValueError("wiki_search optional 'limit' must be an integer")
        limit = max(1, raw_limit)

        matches = await self._store.search_entries(self._chat_id, query.strip(), limit=limit)
        if not matches:
            return "No wiki entries found."

        lines = [f"{entry.key} ({int(entry.importance)}): {entry.value}" for entry in matches]
        return "\n".join(lines)
