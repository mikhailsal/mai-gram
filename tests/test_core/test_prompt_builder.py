"""Tests for PromptBuilder."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.llm.provider import MessageRole


def _make_message(
    message_id: int,
    role: str,
    content: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        role=role,
        content=content,
        show_datetime=False,
        timestamp=datetime.now(timezone.utc),
        tool_call_id=None,
        tool_calls=None,
        reasoning=None,
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
