"""Tests for PromptBuilder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.llm.provider import ChatMessage, MessageRole


def _make_message(
    message_id: int,
    role: str,
    content: str,
    *,
    show_datetime: bool = False,
    tz_name: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: str | None = None,
    reasoning: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        role=role,
        content=content,
        show_datetime=show_datetime,
        timezone=tz_name,
        timestamp=datetime.now(timezone.utc),
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
        reasoning=reasoning,
    )


@pytest.mark.asyncio
async def test_build_context_loads_history_and_prepends_system_prompt() -> None:
    llm = MagicMock()
    llm.count_tokens = AsyncMock(return_value=12)

    message_store = MagicMock()
    message_store.get_recent = AsyncMock(return_value=[_make_message(2, "user", "Hello there")])

    wiki_store = MagicMock()
    wiki_store.sync_from_disk = AsyncMock()
    wiki_store.list_entries_sorted = AsyncMock(return_value=([], 0))

    builder = PromptBuilder(llm, message_store, wiki_store)
    chat = cast("Any", SimpleNamespace(id="test-chat", system_prompt="System prompt"))

    context = await builder.build_context(chat, cut_above_message_id=77)

    wiki_store.sync_from_disk.assert_awaited_once_with("test-chat")
    message_store.get_recent.assert_awaited_once_with(
        "test-chat",
        limit=500,
        after_message_id=77,
    )
    assert context[0].role == MessageRole.SYSTEM
    assert context[0].content.startswith("System prompt")
    assert "[HISTORY NOTE]" in context[0].content
    assert context[1].role == MessageRole.USER
    assert context[1].content == "Hello there"


@pytest.mark.asyncio
async def test_build_context_truncates_oldest_messages_when_over_budget() -> None:
    llm = MagicMock()
    llm.count_tokens = AsyncMock(side_effect=[12, 4])

    message_store = MagicMock()
    message_store.get_recent = AsyncMock(
        return_value=[
            _make_message(1, "user", "oldest"),
            _make_message(2, "assistant", "middle"),
            _make_message(3, "user", "latest"),
        ]
    )

    wiki_store = MagicMock()
    wiki_store.sync_from_disk = AsyncMock()
    wiki_store.list_entries_sorted = AsyncMock(return_value=([], 0))

    builder = PromptBuilder(
        llm,
        message_store,
        wiki_store,
        max_context_tokens=5,
    )
    chat = cast("Any", SimpleNamespace(id="test-chat", system_prompt="System prompt"))

    context = await builder.build_context(chat)

    assert [message.content for message in context] == ["System prompt", "middle", "latest"]
    assert llm.count_tokens.await_count == 2


def test_normalize_conversation_handles_empty_and_merges_consecutive_users() -> None:
    builder = PromptBuilder(MagicMock(), MagicMock(), MagicMock())

    assert builder._normalize_conversation([]) == []

    normalized = builder._normalize_conversation(
        [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.USER, content="again"),
            ChatMessage(role=MessageRole.ASSISTANT, content="done"),
        ]
    )

    assert [message.content for message in normalized] == ["hello\nagain", "done"]


def test_message_to_chat_message_covers_datetime_tool_and_tool_call_parsing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    builder = PromptBuilder(MagicMock(), MagicMock(), MagicMock(), test_mode=True)
    builder._chat_timezone = "Bad/Timezone"

    dated_message = _make_message(1, "user", "hello", show_datetime=True)
    dated_message.timestamp = datetime(2024, 1, 2, 3, 4)
    user_chat_message = builder._message_to_chat_message(dated_message)

    tool_chat_message = builder._message_to_chat_message(
        _make_message(2, "tool", "tool output", tool_call_id="call-1")
    )

    assistant_tool_calls = json.dumps([{"id": "call-2", "name": "wiki_search", "arguments": "{}"}])
    assistant_chat_message = builder._message_to_chat_message(
        _make_message(
            3,
            "assistant",
            "assistant output",
            tool_calls=assistant_tool_calls,
            reasoning="thought",
        )
    )

    invalid_assistant_chat_message = builder._message_to_chat_message(
        _make_message(4, "assistant", "broken", tool_calls="not-json")
    )

    assert user_chat_message.content == "[2024-01-02 03:04 UTC] hello"
    assert tool_chat_message.role == MessageRole.TOOL
    assert tool_chat_message.tool_call_id == "call-1"
    assert assistant_chat_message.tool_calls is not None
    assert assistant_chat_message.tool_calls[0].name == "wiki_search"
    assert assistant_chat_message.reasoning == "thought"
    assert invalid_assistant_chat_message.tool_calls is None
    assert "Failed to parse tool_calls" in caplog.text


def test_build_system_prompt_adds_test_mode_banner() -> None:
    builder = PromptBuilder(MagicMock(), MagicMock(), MagicMock(), test_mode=True)
    chat = cast("Any", SimpleNamespace(system_prompt="System prompt"))

    prompt = builder._build_system_prompt(chat, [], datetime.now(timezone.utc))

    assert prompt.startswith("[TEST MODE]")
    assert prompt.endswith("System prompt")
