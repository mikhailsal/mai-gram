"""Tests for the shared dialogue import logic."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from mai_gram.core.importer import (
    ImportError as ImportParseError,
)
from mai_gram.core.importer import (
    extract_system_prompt,
    parse_import_json,
    save_imported_messages,
)
from mai_gram.db.models import Chat, Message
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TestParseImportJson:
    """Tests for parse_import_json()."""

    def test_parse_simple_array(self) -> None:
        data = json.dumps(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        result = parse_import_json(data)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_parse_bytes_input(self) -> None:
        data = json.dumps([{"role": "user", "content": "Hello"}]).encode()
        result = parse_import_json(data)
        assert len(result) == 1

    def test_parse_proxy_request_format(self) -> None:
        data = json.dumps(
            {
                "request_body": {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                    ]
                },
                "response_body": {
                    "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
                },
                "timestamp": "2024-01-15T14:30:00Z",
            }
        )
        result = parse_import_json(data)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Hi!"

    def test_parse_proxy_with_client_prefix(self) -> None:
        data = json.dumps(
            {
                "client_request_body": {"messages": [{"role": "user", "content": "Hey"}]},
            }
        )
        result = parse_import_json(data)
        assert len(result) == 1

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(ImportParseError, match="Invalid JSON"):
            parse_import_json("not json at all")

    def test_parse_unknown_object_raises(self) -> None:
        with pytest.raises(ImportParseError, match="does not look like a proxy"):
            parse_import_json(json.dumps({"some_key": "value"}))

    def test_parse_non_list_non_dict_raises(self) -> None:
        with pytest.raises(ImportParseError, match="Expected a JSON array"):
            parse_import_json(json.dumps("just a string"))

    def test_parse_proxy_missing_messages_raises(self) -> None:
        data = json.dumps({"request_body": {"no_messages": True}})
        with pytest.raises(ImportParseError, match="no messages array"):
            parse_import_json(data)

    def test_parse_proxy_missing_request_body_raises(self) -> None:
        data = json.dumps({"request_body": "not a dict"})
        with pytest.raises(ImportParseError, match="no request_body"):
            parse_import_json(data)

    def test_parse_with_tool_calls(self) -> None:
        data = json.dumps(
            [
                {"role": "user", "content": "Search for cats"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "search", "arguments": '{"q": "cats"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "Found cats.", "tool_call_id": "tc1"},
            ]
        )
        result = parse_import_json(data)
        assert len(result) == 3

    def test_parse_with_reasoning(self) -> None:
        data = json.dumps(
            [
                {"role": "user", "content": "Think hard"},
                {"role": "assistant", "content": "Done.", "reasoning": "Let me think..."},
            ]
        )
        result = parse_import_json(data)
        assert result[1]["reasoning"] == "Let me think..."


class TestExtractSystemPrompt:
    """Tests for extract_system_prompt()."""

    def test_extract_from_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "Hello"},
        ]
        assert extract_system_prompt(messages) == "You are a helpful AI."

    def test_returns_none_when_no_system(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        assert extract_system_prompt(messages) is None

    def test_returns_none_for_empty_system(self) -> None:
        messages = [{"role": "system", "content": "  "}, {"role": "user", "content": "Hi"}]
        assert extract_system_prompt(messages) is None

    def test_returns_first_system_message(self) -> None:
        messages = [
            {"role": "system", "content": "First prompt"},
            {"role": "system", "content": "Second prompt"},
        ]
        assert extract_system_prompt(messages) == "First prompt"


class TestSaveImportedMessages:
    """Tests for save_imported_messages()."""

    @pytest.fixture
    async def chat(self, session: AsyncSession) -> Chat:
        chat = Chat(
            id="test-import",
            user_id="user1",
            bot_id="bot1",
            llm_model="test/model",
            system_prompt="test prompt",
        )
        session.add(chat)
        await session.flush()
        return chat

    async def test_saves_user_and_assistant(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        store = MessageStore(session)
        before = datetime.now(tz=timezone.utc)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 2

        result = await session.execute(
            select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
        )
        saved = list(result.scalars().all())
        assert len(saved) == 2
        assert saved[1].timestamp > saved[0].timestamp
        ts0 = saved[0].timestamp.replace(tzinfo=timezone.utc)
        assert ts0 >= before
        assert ts0 - before < timedelta(seconds=5)

    async def test_skips_system_messages(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 1

    async def test_skips_invalid_entries(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            "not a dict",
            {"role": "invalid_role", "content": "Bad"},
            {"role": "user", "content": "Good"},
        ]
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 1

    async def test_saves_tool_calls(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {
                            "name": "wiki_create",
                            "arguments": '{"key":"test"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Created.",
                "tool_call_id": "tc1",
            },
        ]
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 2

        result = await session.execute(select(Message).where(Message.chat_id == chat.id))
        saved = list(result.scalars().all())
        assistant_msg = next(m for m in saved if m.role == "assistant")
        assert assistant_msg.tool_calls is not None
        assert "wiki_create" in assistant_msg.tool_calls

    async def test_saves_reasoning(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {
                "role": "assistant",
                "content": "Answer",
                "reasoning": "I thought about it.",
            },
        ]
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 1

        result = await session.execute(select(Message).where(Message.chat_id == chat.id))
        saved = list(result.scalars().all())
        assert saved[0].reasoning == "I thought about it."

    async def test_show_datetime_is_false(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {"role": "user", "content": "Hello"},
        ]
        store = MessageStore(session)
        await save_imported_messages(chat.id, messages_data, store)

        result = await session.execute(select(Message).where(Message.chat_id == chat.id))
        saved = list(result.scalars().all())
        assert saved[0].show_datetime is False

    async def test_empty_list_returns_zero(self, session: AsyncSession, chat: Chat) -> None:
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, [], store)
        assert count == 0

    async def test_assigns_sequential_import_timestamps(
        self, session: AsyncSession, chat: Chat
    ) -> None:
        """All imported messages get sequential timestamps based on the import moment."""
        messages_data = [
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Msg 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Msg 4"},
        ]
        store = MessageStore(session)
        before = datetime.now(tz=timezone.utc)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 4

        result = await session.execute(
            select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
        )
        saved = list(result.scalars().all())
        assert len(saved) == 4

        for j in range(1, len(saved)):
            diff = saved[j].timestamp - saved[j - 1].timestamp
            assert diff == timedelta(seconds=1)

        ts0 = saved[0].timestamp.replace(tzinfo=timezone.utc)
        assert ts0 >= before
        assert ts0 - before < timedelta(seconds=5)

    async def test_ignores_original_timestamps(self, session: AsyncSession, chat: Chat) -> None:
        """Original timestamps from the JSON are ignored; import date is used instead."""
        messages_data = [
            {"role": "user", "content": "Msg 1", "timestamp": "2024-06-15T10:00:00Z"},
            {"role": "assistant", "content": "Msg 2", "timestamp": "2024-09-20T15:30:00Z"},
        ]
        store = MessageStore(session)
        before = datetime.now(tz=timezone.utc)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 2

        result = await session.execute(
            select(Message).where(Message.chat_id == chat.id).order_by(Message.id)
        )
        saved = list(result.scalars().all())
        ts0 = saved[0].timestamp.replace(tzinfo=timezone.utc)
        assert ts0 >= before
        assert saved[0].timestamp.year == before.year

    async def test_handles_non_string_content(self, session: AsyncSession, chat: Chat) -> None:
        messages_data = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": 42},
        ]
        store = MessageStore(session)
        count = await save_imported_messages(chat.id, messages_data, store)
        assert count == 2
