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
    _send_with_retry,
    replay_imported_messages,
)
from mai_gram.core.telegram_limits import split_html_safe
from mai_gram.messenger.base import OutgoingMessage, SendResult


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


def _make_mock_messenger(*, fail_pattern: list[bool] | None = None) -> AsyncMock:
    """Create a mock Messenger.

    If *fail_pattern* is given, calls fail/succeed according to the list
    (True = success, False = failure with "flood control" error). The pattern
    repeats if shorter than the number of calls.
    """
    messenger = AsyncMock()
    call_count = 0

    async def _send_message(msg: Any, **kwargs: Any) -> SendResult:
        nonlocal call_count
        call_count += 1
        if fail_pattern is not None:
            idx = (call_count - 1) % len(fail_pattern)
            if not fail_pattern[idx]:
                return SendResult(
                    success=False,
                    error="Flood control exceeded. Retry in 1 seconds",
                )
        return SendResult(success=True, message_id=f"msg-{call_count}")

    messenger.send_message = AsyncMock(side_effect=_send_message)
    return messenger


class TestSplitHtmlSafe:
    """Tests for split_html_safe()."""

    def test_short_text_not_split(self) -> None:
        result = split_html_safe("Hello world", max_len=100)
        assert result == ["Hello world"]

    def test_splits_at_paragraph_boundary(self) -> None:
        text = "A" * 50 + "\n\n" + "B" * 50
        result = split_html_safe(text, max_len=60)
        assert len(result) == 2
        assert result[0].startswith("A")
        assert result[1].startswith("B")

    def test_splits_at_newline(self) -> None:
        text = "A" * 50 + "\n" + "B" * 50
        result = split_html_safe(text, max_len=60)
        assert len(result) == 2

    def test_splits_at_space(self) -> None:
        text = "word " * 20
        result = split_html_safe(text, max_len=30)
        assert len(result) > 1
        assert all(len(chunk) <= 30 for chunk in result)

    def test_hard_cut_when_no_boundary(self) -> None:
        text = "X" * 200
        result = split_html_safe(text, max_len=50)
        assert len(result) == 4
        assert all(len(chunk) <= 50 for chunk in result)

    def test_empty_text(self) -> None:
        result = split_html_safe("")
        assert result == [""]


class TestFormatFunctions:
    """Tests for message formatting functions."""

    def test_format_user_message_returns_list(self) -> None:
        result = _format_user_message("Hello there!")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "[You]" in result[0]
        assert "Hello there!" in result[0]

    def test_format_user_message_splits_long_content(self) -> None:
        long_text = "word " * 2000
        result = _format_user_message(long_text)
        assert len(result) > 1
        assert "[You]" in result[0]
        assert "[You]" not in result[1]

    def test_format_assistant_message_returns_list(self) -> None:
        result = _format_assistant_message("Hi!")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "[AI]" in result[0]

    def test_format_assistant_splits_long_content(self) -> None:
        long_text = "word " * 2000
        result = _format_assistant_message(long_text)
        assert len(result) > 1
        assert "[AI]" in result[0]

    def test_format_assistant_with_reasoning(self) -> None:
        result = _format_assistant_message("Answer", reasoning="Thinking...")
        assert len(result) >= 1
        combined = " ".join(result)
        assert "Reasoning" in combined
        assert "Answer" in combined

    def test_build_cut_keyboard(self) -> None:
        kb = _build_cut_keyboard(42)
        assert len(kb) == 1
        assert len(kb[0]) == 1
        assert "cut:42" in kb[0][0][1]


class TestSendWithRetry:
    """Tests for _send_with_retry()."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        messenger = _make_mock_messenger()
        msg = OutgoingMessage(text="test", chat_id="123")
        ok = await _send_with_retry(messenger, msg, max_retries=3, base_delay=0.01)
        assert ok is True
        assert messenger.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_flood_control(self) -> None:
        messenger = _make_mock_messenger(fail_pattern=[False, True])
        msg = OutgoingMessage(text="test", chat_id="123")
        ok = await _send_with_retry(messenger, msg, max_retries=3, base_delay=0.01)
        assert ok is True
        assert messenger.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self) -> None:
        messenger = _make_mock_messenger(fail_pattern=[False])
        msg = OutgoingMessage(text="test", chat_id="123")
        ok = await _send_with_retry(messenger, msg, max_retries=2, base_delay=0.01)
        assert ok is False
        assert messenger.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_truncates_on_too_long_error(self) -> None:
        messenger = AsyncMock()
        call_count = 0

        async def _send(msg: Any, **kw: Any) -> SendResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SendResult(success=False, error="Message is too long")
            return SendResult(success=True, message_id="ok")

        messenger.send_message = AsyncMock(side_effect=_send)
        long_text = "X" * 5000
        msg = OutgoingMessage(text=long_text, chat_id="123")
        ok = await _send_with_retry(messenger, msg, max_retries=3, base_delay=0.01)
        assert ok is True


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
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_long_message_is_split(self) -> None:
        """Messages exceeding 4096 chars should be split into multiple sends."""
        messenger = _make_mock_messenger()
        long_text = "word " * 2000
        messages = [
            _make_mock_message(msg_id=1, role="assistant", content=long_text),
        ]
        count = await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        assert count == 1
        all_calls = messenger.send_message.call_args_list
        html_calls = [c for c in all_calls if c.args[0].parse_mode == "html"]
        assert len(html_calls) > 1
        kb_calls = [c for c in html_calls if c.args[0].keyboard is not None]
        assert len(kb_calls) == 1

    @pytest.mark.asyncio
    async def test_cut_button_only_on_last_part(self) -> None:
        """When an assistant message is split, only the last part has the Cut button."""
        messenger = _make_mock_messenger()
        long_text = "sentence. " * 1000
        messages = [
            _make_mock_message(msg_id=42, role="assistant", content=long_text),
        ]
        await replay_imported_messages(messenger, "chat-123", messages, delay_seconds=0.01)
        all_calls = messenger.send_message.call_args_list
        html_calls = [c for c in all_calls if c.args[0].parse_mode == "html"]
        for call in html_calls[:-1]:
            assert call.args[0].keyboard is None
        assert html_calls[-1].args[0].keyboard is not None
