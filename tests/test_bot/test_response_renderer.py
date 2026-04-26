"""Unit tests for the response renderer service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from mai_gram.bot.response_renderer import ResponseRenderer
from mai_gram.llm.provider import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
)
from mai_gram.messenger.base import SendResult


def _make_renderer() -> tuple[ResponseRenderer, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))
    messenger.edit_message = AsyncMock(return_value=SendResult(success=True))
    message_logger = MagicMock()
    renderer = ResponseRenderer(messenger, message_logger=message_logger)
    return renderer, messenger, message_logger


class TestResponseRenderer:
    def test_build_intermediate_display_prefers_reasoning_when_enabled(self) -> None:
        renderer, _, _ = _make_renderer()

        display = renderer._build_intermediate_display("Final answer", "Step 1", True)

        assert display == "💭 Reasoning:\nStep 1\n\n───\n\nFinal answer"

    def test_format_usage_footer_includes_tokens_and_cost(self) -> None:
        renderer, _, _ = _make_renderer()
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=34)

        footer = renderer._format_usage_footer(usage, 0.1234, False)

        assert footer == "12/34 tokens | $0.1234"

    def test_user_friendly_error_maps_known_llm_failures(self) -> None:
        renderer, _, _ = _make_renderer()

        assert (
            "authentication failed"
            in renderer._user_friendly_error(LLMAuthenticationError("bad key")).lower()
        )
        assert "~7s" in renderer._user_friendly_error(
            LLMRateLimitError("slow down", retry_after=7.9)
        )
        assert (
            "selected model is no longer available"
            in renderer._user_friendly_error(LLMModelNotFoundError("gone")).lower()
        )
        assert (
            "conversation is too long"
            in renderer._user_friendly_error(LLMContextLengthError("too long")).lower()
        )
        assert (
            "http 503"
            in renderer._user_friendly_error(LLMProviderError("busy", status_code=503)).lower()
        )
        assert "something went wrong" in renderer._user_friendly_error(LLMError("provider")).lower()
        assert "unexpected error" in renderer._user_friendly_error(RuntimeError("boom")).lower()

    async def test_deliver_error_retries_failed_placeholder_edits(self) -> None:
        renderer, messenger, _ = _make_renderer()
        messenger.edit_message = AsyncMock(
            side_effect=[
                SendResult(success=False, error="temporary"),
                SendResult(success=True),
            ]
        )
        sent_msg_ids: list[str] = []

        with patch("mai_gram.bot.response_renderer.asyncio.sleep", new=AsyncMock()) as sleep:
            await renderer._deliver_error(
                "test-chat",
                "error text",
                placeholder_msg_id="placeholder-1",
                sent_msg_ids=sent_msg_ids,
                max_attempts=2,
            )

        assert sent_msg_ids == ["placeholder-1"]
        assert messenger.edit_message.await_count == 2
        sleep.assert_awaited_once_with(2.0)

    async def test_send_part_uses_split_fallback_for_oversized_messages(self) -> None:
        renderer, messenger, _ = _make_renderer()
        messenger.send_message = AsyncMock(
            return_value=SendResult(success=False, error="Message is too long")
        )
        renderer._send_part_split = AsyncMock(return_value="split-last")

        message_id = await renderer._send_part("test-chat", "<b>very long</b>")

        assert message_id == "split-last"
        renderer._send_part_split.assert_awaited_once_with(
            "test-chat",
            "<b>very long</b>",
            keyboard=None,
        )

    async def test_send_part_falls_back_to_plain_text_for_bad_html(self) -> None:
        renderer, messenger, _ = _make_renderer()
        messenger.send_message = AsyncMock(
            side_effect=[
                SendResult(success=False, error="Can't find end tag"),
                SendResult(success=True, message_id="plain-1"),
            ]
        )

        message_id = await renderer._send_part("test-chat", "<b>hello</b>")

        assert message_id == "plain-1"
        first_call = messenger.send_message.await_args_list[0].args[0]
        second_call = messenger.send_message.await_args_list[1].args[0]
        assert first_call.parse_mode == "html"
        assert second_call.text == "hello"
        assert second_call.parse_mode is None

    async def test_send_long_message_sends_header_as_separate_part_when_needed(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._send_part = AsyncMock(side_effect=["header-id", "part-1", "part-2"])

        with (
            patch("mai_gram.core.telegram_limits.SAFE_MAX_LENGTH", 15),
            patch(
                "mai_gram.core.telegram_limits.split_html_safe",
                side_effect=[["first", "second"], ["second"]],
            ),
            patch("mai_gram.core.md_to_telegram.markdown_to_html", side_effect=lambda text: text),
        ):
            sent_ids = await renderer._send_long_message(
                "test-chat",
                "ignored raw text",
                header_html="HEADER-TOO-LONG",
                keyboard="kbd",
            )

        assert sent_ids == ["header-id", "part-1", "part-2"]
        assert renderer._send_part.await_args_list[0].args == ("test-chat", "HEADER-TOO-LONG")
        assert renderer._send_part.await_args_list[1].kwargs["keyboard"] is None
        assert renderer._send_part.await_args_list[2].kwargs["keyboard"] == "kbd"

    async def test_edit_part_falls_back_to_plain_text_for_bad_html(self) -> None:
        renderer, messenger, _ = _make_renderer()
        messenger.edit_message = AsyncMock(
            side_effect=[
                SendResult(success=False, error="parse entities"),
                SendResult(success=True),
            ]
        )

        edited = await renderer._edit_part("test-chat", "msg-1", "<i>hello</i>")

        assert edited is True
        second_call = messenger.edit_message.await_args_list[1]
        assert second_call.args == ("test-chat", "msg-1", "hello")

    async def test_finalize_placeholder_sends_remaining_parts_after_edit(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._edit_part = AsyncMock(return_value=True)
        renderer._send_part = AsyncMock(return_value="extra-1")

        with (
            patch(
                "mai_gram.core.telegram_limits.split_html_safe",
                side_effect=[["first", "second"], ["second"]],
            ),
            patch(
                "mai_gram.core.md_to_telegram.markdown_to_html",
                side_effect=lambda text: f"<{text}>",
            ),
        ):
            extra_ids = await renderer._finalize_placeholder(
                "test-chat",
                "placeholder-1",
                "ignored raw text",
                keyboard="kbd",
            )

        assert extra_ids == ["extra-1"]
        renderer._edit_part.assert_awaited_once_with("test-chat", "placeholder-1", "<first>")
        renderer._send_part.assert_awaited_once_with("test-chat", "<second>", keyboard="kbd")

    async def test_finalize_placeholder_keeps_header_in_placeholder_when_it_overflows(self) -> None:
        renderer, _, _ = _make_renderer()
        renderer._edit_part = AsyncMock(return_value=True)
        renderer._send_part = AsyncMock(return_value="part-1")

        with (
            patch("mai_gram.core.telegram_limits.SAFE_MAX_LENGTH", 10),
            patch("mai_gram.core.telegram_limits.split_html_safe", return_value=["first"]),
            patch("mai_gram.core.md_to_telegram.markdown_to_html", side_effect=lambda text: text),
        ):
            extra_ids = await renderer._finalize_placeholder(
                "test-chat",
                "placeholder-1",
                "ignored raw text",
                header_html="HEADER-LONG",
                keyboard="kbd",
            )

        assert extra_ids == ["part-1"]
        renderer._edit_part.assert_awaited_once_with("test-chat", "placeholder-1", "HEADER-LONG")
        renderer._send_part.assert_awaited_once_with("test-chat", "first", keyboard="kbd")

    async def test_commit_overflow_commits_chunks_and_creates_new_placeholder(self) -> None:
        renderer, messenger, _ = _make_renderer()
        renderer._edit_part = AsyncMock(return_value=True)
        renderer._send_part = AsyncMock(return_value="sent-1")
        messenger.send_message = AsyncMock(
            return_value=SendResult(success=True, message_id="placeholder-2")
        )
        sent_msg_ids: list[str] = []
        current_content = "12345678901234567890123456789012345"

        with (
            patch("mai_gram.core.telegram_limits.SAFE_MAX_LENGTH", 220),
            patch("mai_gram.core.md_to_telegram.markdown_to_html", side_effect=lambda text: text),
        ):
            offset, new_placeholder = await renderer._commit_overflow(
                tg_chat_id="test-chat",
                header_html="HEADER",
                reasoning_committed=False,
                placeholder_msg_id="placeholder-1",
                sent_msg_ids=sent_msg_ids,
                remaining_content=current_content,
                current_content=current_content,
                committed_content_offset=0,
            )

        assert offset == 20
        assert new_placeholder == "placeholder-2"
        assert sent_msg_ids == ["placeholder-1", "sent-1"]
        renderer._edit_part.assert_awaited_once_with("test-chat", "placeholder-1", "HEADER")
        renderer._send_part.assert_awaited_once_with("test-chat", current_content[:20])

    async def test_send_response_formats_reasoning_header_and_logs_outgoing(self) -> None:
        renderer, _, message_logger = _make_renderer()
        renderer._send_long_message = AsyncMock(return_value=["msg-1", "msg-2"])

        with patch(
            "mai_gram.core.md_to_telegram.format_reasoning_html",
            return_value="<reasoning>",
        ):
            sent_ids = await renderer._send_response(
                "test-chat",
                response_text="Final answer",
                response_reasoning="Hidden chain",
                show_reasoning=True,
                keyboard="kbd",
            )

        assert sent_ids == ["msg-1", "msg-2"]
        renderer._send_long_message.assert_awaited_once_with(
            "test-chat",
            "Final answer",
            header_html="<reasoning>",
            keyboard="kbd",
        )
        message_logger.log_outgoing.assert_called_once_with(
            "test-chat",
            "Final answer",
            success=True,
            message_id="msg-2",
        )
