"""Tests for the rate-limited message replay engine."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mai_gram.core.replay import (
    _build_cut_keyboard,
    _format_assistant_message,
    _format_user_message,
    replay_imported_messages,
)
from mai_gram.messenger.base import SendResult


def _make_mock_message(
    *,
    msg_id: int,
    role: str,
    content: str,
    tool_calls: str | None = None,
    tool_call_id: str | None = None,
) -> MagicMock:
    """Create a mock Message ORM object."""
    msg = MagicMock()
    msg.id = msg_id
    msg.role = role
    msg.content = content
    msg.tool_calls = tool_calls
    msg.tool_call_id = tool_call_id
    return msg


def _make_mock_messenger() -> AsyncMock:
    """Create a mock Messenger that returns success on every send."""
    messenger = AsyncMock()
    call_count = 0

    async def _send_message(msg: Any) -> SendResult:
        nonlocal call_count
        call_count += 1
        return SendResult(success=True, message_id=f"msg-{call_count}")

    messenger.send_message = AsyncMock(side_effect=_send_message)
    return messenger


class TestFormatFunctions:
    """Tests for message formatting functions."""

    def test_format_user_message(self) -> None:
        result = _format_user_message("Hello there!")
        assert "[You]" in result
        assert "Hello there!" in result

    def test_format_user_message_truncates_long_content(self) -> None:
        long_text = "x" * 5000
        result = _format_user_message(long_text)
        assert len(result) < 5000
        assert "..." in result

    def test_format_assistant_message(self) -> None:
        result = _format_assistant_message("Hi!")
        assert "[AI]" in result
        assert "Hi!" in result

    def test_format_assistant_escapes_html(self) -> None:
        result = _format_assistant_message("<b>bold</b>")
        assert "&lt;b&gt;" in result

    def test_build_cut_keyboard(self) -> None:
        kb = _build_cut_keyboard(42)
        assert len(kb) == 1
        assert len(kb[0]) == 1
        assert "cut:42" in kb[0][0][1]


class TestReplayImportedMessages:
    """Tests for replay_imported_messages()."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_zero(self) -> None:
        messenger = _make_mock_messenger()
        count = await replay_imported_messages(messenger, "chat-123", [])
        assert count == 0
        messenger.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_replays_user_and_assistant(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(msg_id=2, role="assistant", content="Hi!"),
        ]
        count = await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        assert count == 2
        # 1 intro + 2 messages + 1 summary = 4 calls
        assert messenger.send_message.call_count == 4

    @pytest.mark.asyncio
    async def test_skips_system_messages(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="system", content="You are helpful."),
            _make_mock_message(msg_id=2, role="user", content="Hello"),
        ]
        count = await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        assert count == 1

    @pytest.mark.asyncio
    async def test_skips_tool_messages_by_default(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(
                msg_id=2,
                role="assistant",
                content="",
                tool_calls='[{"id":"tc1","name":"search","arguments":"{}"}]',
            ),
            _make_mock_message(msg_id=3, role="tool", content="Result", tool_call_id="tc1"),
            _make_mock_message(msg_id=4, role="assistant", content="Answer"),
        ]
        count = await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        # user + assistant(with content) = 2 sent
        assert count == 2

    @pytest.mark.asyncio
    async def test_shows_tool_messages_when_enabled(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(
                msg_id=2,
                role="assistant",
                content="",
                tool_calls='[{"id":"tc1","name":"search","arguments":"{}"}]',
            ),
            _make_mock_message(msg_id=3, role="tool", content="Result", tool_call_id="tc1"),
            _make_mock_message(msg_id=4, role="assistant", content="Answer"),
        ]
        count = await replay_imported_messages(
            messenger, "chat-123", messages, delay_seconds=0.01, show_tool_calls=True
        )
        # user + tool_call_assistant + tool + assistant = 4
        assert count == 4

    @pytest.mark.asyncio
    async def test_sends_progress_updates(self) -> None:
        messenger = _make_mock_messenger()
        messages = []
        for i in range(60):
            messages.append(
                _make_mock_message(msg_id=i * 2 + 1, role="user", content=f"Message {i}")
            )
            messages.append(
                _make_mock_message(msg_id=i * 2 + 2, role="assistant", content=f"Reply {i}")
            )
        count = await replay_imported_messages(
            messenger, "chat-123", messages, delay_seconds=0.001, progress_interval=25
        )
        assert count == 120

        all_texts = [call.args[0].text for call in messenger.send_message.call_args_list]
        progress_msgs = [t for t in all_texts if "progress" in t.lower()]
        assert len(progress_msgs) >= 2

    @pytest.mark.asyncio
    async def test_sends_summary_at_end(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(msg_id=2, role="assistant", content="Hi!"),
        ]
        await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        last_call = messenger.send_message.call_args_list[-1]
        assert "complete" in last_call.args[0].text.lower()

    @pytest.mark.asyncio
    async def test_assistant_messages_have_cut_button(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(msg_id=2, role="assistant", content="Hi!"),
        ]
        await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        all_calls = messenger.send_message.call_args_list
        assistant_calls = [c for c in all_calls if c.args[0].keyboard is not None]
        assert len(assistant_calls) >= 1
        kb = assistant_calls[0].args[0].keyboard
        assert any("cut:2" in str(row) for row in kb)

    @pytest.mark.asyncio
    async def test_respects_delay(self) -> None:
        messenger = _make_mock_messenger()
        messages = [
            _make_mock_message(msg_id=1, role="user", content="Hello"),
            _make_mock_message(msg_id=2, role="assistant", content="Hi!"),
        ]
        start = time.monotonic()
        await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.1)
        elapsed = time.monotonic() - start
        # At least 3 delays (intro + 2 messages)
        assert elapsed >= 0.2
