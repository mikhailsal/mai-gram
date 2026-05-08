"""Unit tests for the shared conversation executor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mai_gram.bot.conversation_executor import (
    AssistantTurnRequest,
    ConversationExecutor,
    _StreamOutcome,
    _StreamState,
)
from mai_gram.bot.tool_activity_notifier import ToolActivityNotifier
from mai_gram.db.models import Chat
from mai_gram.llm.provider import LLMProviderError, ToolCall
from mai_gram.messenger.base import SendResult


def _make_request(
    *,
    message_store: Any | None = None,
    mcp_manager: Any | None = None,
    llm_messages: list[object] | None = None,
    show_reasoning: bool = False,
    show_tool_calls: bool = False,
    extra_params: dict[str, Any] | None = None,
    failure_log_message: str = "Failed to generate response",
    resolved_model: str | None = None,
) -> AssistantTurnRequest:
    chat = Chat(
        id="test-user@test-bot",
        user_id="test-user",
        bot_id="test-bot",
        llm_model="test-model",
        system_prompt="test prompt",
        show_reasoning=show_reasoning,
        show_tool_calls=show_tool_calls,
    )
    if message_store is None:
        message_store = MagicMock(save_message=AsyncMock(return_value=SimpleNamespace(id=7)))
    if mcp_manager is None:
        mcp_manager = MagicMock()
    llm_messages_value: Any = llm_messages or []
    assert message_store is not None
    assert mcp_manager is not None
    return AssistantTurnRequest(
        chat=chat,
        message_store=message_store,
        mcp_manager=mcp_manager,
        llm_messages=llm_messages_value,
        telegram_chat_id="test-telegram-chat",
        timezone_name="UTC",
        show_datetime=True,
        show_reasoning=show_reasoning,
        show_tool_calls=show_tool_calls,
        extra_params=extra_params,
        failure_log_message=failure_log_message,
        resolved_model=resolved_model,
    )


def _make_executor() -> tuple[ConversationExecutor, MagicMock, MagicMock]:
    messenger = MagicMock()
    messenger.max_message_length = 4000
    messenger.send_message = AsyncMock(return_value=SendResult(success=True, message_id="sent-1"))
    messenger.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="edited-1"))

    renderer = MagicMock()
    renderer._send_response = AsyncMock(return_value=["final-id"])
    renderer._finalize_placeholder = AsyncMock(return_value=["overflow-id"])
    renderer._deliver_error = AsyncMock()
    renderer._commit_overflow = AsyncMock(return_value=(5, "placeholder-2"))
    renderer._build_intermediate_display = MagicMock(return_value="intermediate")
    renderer._format_usage_footer = MagicMock(return_value="")
    renderer._user_friendly_error = MagicMock(return_value="Friendly error")

    executor = ConversationExecutor(
        messenger,
        MagicMock(),
        tool_max_iterations=5,
        renderer=renderer,
    )
    return executor, messenger, renderer


class TestConversationExecutor:
    async def test_execute_success_saves_final_message(self) -> None:
        executor, _, renderer = _make_executor()
        save_message = AsyncMock(return_value=SimpleNamespace(id=7))
        request = _make_request(message_store=MagicMock(save_message=save_message))

        with patch.object(
            executor,
            "_stream_response",
            AsyncMock(
                return_value=_StreamOutcome(
                    response_text="Hello there",
                    response_reasoning=None,
                    placeholder_msg_id=None,
                    committed_content_offset=0,
                    reasoning_committed=False,
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
                    cost=0.12,
                    is_byok=False,
                )
            ),
        ):
            result = await executor.execute(request)

        save_message.assert_awaited_once_with(
            request.chat.id,
            "assistant",
            "Hello there",
            reasoning=None,
            timezone_name="UTC",
            show_datetime=True,
        )
        renderer._send_response.assert_awaited_once()
        assert result.sent_message_ids == ["final-id"]

    async def test_execute_delivers_user_friendly_error_on_empty_response(self) -> None:
        executor, _, renderer = _make_executor()
        request = _make_request()

        with patch.object(
            executor,
            "_stream_response",
            AsyncMock(side_effect=LLMProviderError("The model returned an empty response")),
        ):
            result = await executor.execute(request)

        renderer._deliver_error.assert_awaited_once()
        assert result.sent_message_ids == []

    async def test_execute_propagates_unexpected_exceptions(self) -> None:
        executor, _, renderer = _make_executor()
        request = _make_request()

        with (
            patch.object(
                executor,
                "_stream_response",
                AsyncMock(side_effect=ValueError("bug")),
            ),
            pytest.raises(ValueError, match="bug"),
        ):
            await executor.execute(request)

        renderer._deliver_error.assert_not_called()

    async def test_tool_callbacks_persist_and_display_activity(self) -> None:
        executor, messenger, _ = _make_executor()
        save_message = AsyncMock(return_value=SimpleNamespace(id=9))
        request = _make_request(
            message_store=MagicMock(save_message=save_message),
            show_tool_calls=True,
        )
        messenger.send_message = AsyncMock(
            side_effect=[
                SendResult(success=True, message_id="tool-call-id"),
                SendResult(success=True, message_id="tool-result-id"),
            ]
        )
        sent_msg_ids: list[str] = []

        tool_call_cb, tool_result_cb = executor._tool_activity.build_callbacks(
            request,
            sent_msg_ids,
        )

        await tool_call_cb(
            content="",
            tool_calls=[ToolCall(id="call-1", name="wiki_create", arguments='{"title":"Test"}')],
        )
        await tool_result_cb(
            tool_call_id="call-1",
            tool_name="wiki_create",
            arguments="{}",
            result={"ok": True},
            content="Saved",
            error=None,
            server_name=None,
        )

        assert save_message.await_count == 2
        roles = [call.args[1] for call in save_message.await_args_list]
        assert roles == ["assistant", "tool"]
        assert save_message.await_args_list[0].kwargs["tool_calls"] == [
            ToolCall(id="call-1", name="wiki_create", arguments='{"title":"Test"}')
        ]
        assert sent_msg_ids == ["tool-call-id", "tool-result-id"]

    async def test_tool_display_helpers_cover_guard_and_fallback_branches(self) -> None:
        executor, messenger, _ = _make_executor()
        hidden_request = _make_request(show_tool_calls=False)
        visible_request = _make_request(show_tool_calls=True)
        sent_msg_ids: list[str] = []

        await executor._tool_activity._maybe_send_tool_call_display(
            hidden_request,
            sent_msg_ids,
            [],
        )
        await executor._tool_activity._maybe_send_tool_result_display(
            hidden_request,
            sent_msg_ids,
            tool_name="wiki_create",
            result=None,
            error=None,
        )
        await executor._tool_activity._maybe_send_tool_call_display(
            visible_request,
            sent_msg_ids,
            [],
        )

        assert messenger.send_message.await_count == 0
        assert ToolActivityNotifier.tool_call_lines(
            [ToolCall(id="call-1", name="wiki_create", arguments="{bad json")]
        ) == ["🔧 wiki_create({bad json)"]
        assert (
            ToolActivityNotifier.tool_result_text(
                tool_name="wiki_create",
                result=None,
                error="boom",
            )
            == "❌ wiki_create: boom"
        )
        assert ToolActivityNotifier.tool_result_text(
            tool_name="wiki_create",
            result="x" * 205,
            error=None,
        ).endswith("…")

    async def test_stream_response_collects_chunks_and_usage(self) -> None:
        executor, _, _ = _make_executor()
        request = _make_request()

        async def _stream(*args: object, **kwargs: object) -> Any:
            del args, kwargs
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
                cost=0.01,
                is_byok=True,
                turn_complete=False,
                reasoning="think",
                content="First",
                finish_reason=None,
            )
            yield SimpleNamespace(
                usage=None,
                cost=None,
                is_byok=False,
                turn_complete=True,
                reasoning="",
                content="",
                finish_reason=None,
            )
            yield SimpleNamespace(
                usage=None,
                cost=None,
                is_byok=False,
                turn_complete=False,
                reasoning="",
                content="Second",
                finish_reason="stop",
            )

        with (
            patch("mai_gram.bot.conversation_executor.run_with_tools_stream", _stream),
            patch.object(executor, "_handle_turn_complete", AsyncMock()),
            patch.object(executor, "_maybe_update_live_display", AsyncMock()),
        ):
            outcome = await executor._stream_response(
                request,
                [],
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        assert outcome.response_text == "FirstSecond"
        assert outcome.response_reasoning == "think"
        assert outcome.cost == 0.01
        assert outcome.is_byok is True
        assert outcome.finish_reason == "stop"

    async def test_resolved_model_passed_to_streaming_call(self) -> None:
        executor, _, _ = _make_executor()
        request = _make_request(resolved_model="real/api-model-id")

        assert request.model_for_api == "real/api-model-id"

        captured_kwargs: dict[str, Any] = {}

        async def _stream(*args: object, **kwargs: object) -> Any:
            captured_kwargs.update(kwargs)
            yield SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
                cost=0.01,
                is_byok=False,
                turn_complete=False,
                reasoning="",
                content="Hello",
                finish_reason="stop",
            )

        with (
            patch("mai_gram.bot.conversation_executor.run_with_tools_stream", _stream),
            patch.object(executor, "_maybe_update_live_display", AsyncMock()),
        ):
            outcome = await executor._stream_response(
                request,
                [],
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        assert captured_kwargs["model"] == "real/api-model-id"
        assert outcome.response_text == "Hello"

    async def test_model_for_api_falls_back_to_chat_llm_model(self) -> None:
        request = _make_request()
        assert request.resolved_model is None
        assert request.model_for_api == "test-model"

    async def test_maybe_update_live_display_covers_non_edit_and_placeholder_path(self) -> None:
        executor, _, _ = _make_executor()
        request = _make_request(show_reasoning=True)
        quiet_state = _StreamState(
            content_parts=["content"],
            reasoning_parts=["reasoning"],
            last_edit_time=5.0,
            last_display_len=len("contentreasoning"),
        )

        with patch("mai_gram.bot.conversation_executor.time.monotonic", return_value=5.1):
            await executor._maybe_update_live_display(request, quiet_state, [])

        active_state = _StreamState(
            content_parts=["content"],
            reasoning_parts=["reasoning"],
        )
        with (
            patch("mai_gram.bot.conversation_executor.time.monotonic", return_value=8.0),
            patch.object(executor, "_render_live_text", return_value=None),
        ):
            await executor._maybe_update_live_display(request, active_state, [])

        with (
            patch("mai_gram.bot.conversation_executor.time.monotonic", return_value=10.0),
            patch.object(
                executor,
                "_render_live_text",
                return_value=("short", "fallback", "", "content"),
            ),
            patch.object(
                executor,
                "_send_or_edit_placeholder",
                AsyncMock(return_value="placeholder-3"),
            ) as send_or_edit,
        ):
            await executor._maybe_update_live_display(request, active_state, [])

        send_or_edit.assert_awaited_once()
        assert active_state.placeholder_msg_id == "placeholder-3"

    async def test_send_or_edit_placeholder_edits_existing_message(self) -> None:
        executor, messenger, _ = _make_executor()
        request = _make_request()
        messenger.edit_message = AsyncMock(
            side_effect=[
                SendResult(success=False, error="bad html"),
                SendResult(success=True, message_id="placeholder-1"),
            ]
        )

        placeholder = await executor._send_or_edit_placeholder(
            request,
            placeholder_msg_id="placeholder-1",
            live_text="<b>html</b>",
            fallback="plain text",
        )

        assert placeholder == "placeholder-1"
        assert messenger.edit_message.await_count == 2

    async def test_handle_turn_complete_flushes_placeholder_state(self) -> None:
        executor, messenger, renderer = _make_executor()
        request = _make_request()
        state = _StreamState(
            content_parts=["Partial answer"],
            reasoning_parts=["Reasoning"],
            placeholder_msg_id="placeholder-1",
            last_edit_time=2.0,
            last_display_len=20,
            committed_content_offset=10,
            reasoning_committed=True,
        )
        sent_msg_ids: list[str] = []

        await executor._handle_turn_complete(request, state, sent_msg_ids)

        messenger.edit_message.assert_awaited_once_with(
            request.telegram_chat_id,
            "placeholder-1",
            "intermediate",
        )
        assert sent_msg_ids == ["placeholder-1"]
        assert state.content_parts == []
        assert state.reasoning_parts == []
        assert state.placeholder_msg_id is None
        assert state.committed_content_offset == 0
        assert state.reasoning_committed is False
        renderer._build_intermediate_display.assert_called_once()

    async def test_maybe_update_live_display_commits_overflow(self) -> None:
        executor, _, renderer = _make_executor()
        request = _make_request(show_reasoning=True)
        state = _StreamState(
            content_parts=["content"],
            reasoning_parts=["reasoning"],
            placeholder_msg_id="placeholder-1",
        )

        with (
            patch.object(
                executor,
                "_render_live_text",
                return_value=("x" * 5000, "fallback", "<b>reasoning</b>", "content"),
            ),
            patch("mai_gram.bot.conversation_executor.time.monotonic", return_value=5.0),
            patch("mai_gram.core.telegram_limits.SAFE_MAX_LENGTH", 100),
        ):
            await executor._maybe_update_live_display(request, state, [])

        renderer._commit_overflow.assert_awaited_once()
        assert state.committed_content_offset == 5
        assert state.placeholder_msg_id == "placeholder-2"
        assert state.reasoning_committed is True

    async def test_send_or_edit_placeholder_falls_back_to_plain_text(self) -> None:
        executor, messenger, _ = _make_executor()
        request = _make_request()
        messenger.send_message = AsyncMock(
            side_effect=[
                SendResult(success=False, error="bad html"),
                SendResult(success=True, message_id="fallback-id"),
            ]
        )

        placeholder = await executor._send_or_edit_placeholder(
            request,
            placeholder_msg_id=None,
            live_text="<b>html</b>",
            fallback="plain text",
        )

        assert placeholder == "fallback-id"
        assert messenger.send_message.await_count == 2

    async def test_send_or_edit_placeholder_returns_none_when_both_edits_fail(self) -> None:
        executor, messenger, _ = _make_executor()
        request = _make_request()
        messenger.edit_message = AsyncMock(
            return_value=SendResult(success=False, error="permanently broken")
        )

        placeholder = await executor._send_or_edit_placeholder(
            request,
            placeholder_msg_id="dead-placeholder",
            live_text="<b>html</b>",
            fallback="plain text",
        )

        assert placeholder is None
        assert messenger.edit_message.await_count == 2

    async def test_finalize_response_updates_existing_placeholder(self) -> None:
        executor, _, renderer = _make_executor()
        request = _make_request(show_reasoning=True)
        outcome = _StreamOutcome(
            response_text="Final answer",
            response_reasoning="Reasoning trace",
            placeholder_msg_id="placeholder-1",
            committed_content_offset=0,
            reasoning_committed=False,
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            cost=0.05,
            is_byok=False,
        )
        renderer._format_usage_footer.return_value = "1/2 tokens"
        sent_msg_ids: list[str] = []

        await executor._finalize_response(request, sent_msg_ids, outcome, 11)

        renderer._finalize_placeholder.assert_awaited_once()
        renderer._send_response.assert_not_called()
        assert sent_msg_ids == ["overflow-id", "placeholder-1"]

    async def test_finalize_response_appends_usage_footer_without_placeholder(self) -> None:
        executor, messenger, renderer = _make_executor()
        request = _make_request(show_reasoning=False)
        outcome = _StreamOutcome(
            response_text="Final answer",
            response_reasoning=None,
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
            cost=0.05,
            is_byok=False,
        )
        renderer._format_usage_footer.return_value = "1/2 tokens"
        sent_msg_ids: list[str] = []

        await executor._finalize_response(request, sent_msg_ids, outcome, 11)

        renderer._send_response.assert_awaited_once_with(
            request.telegram_chat_id,
            response_text="Final answer\n\n1/2 tokens",
            response_reasoning=None,
            show_reasoning=False,
            keyboard=messenger.build_inline_keyboard.return_value,
        )
        messenger.build_inline_keyboard.assert_called_once()
        assert sent_msg_ids == ["final-id"]

    def test_helper_formatters_cover_tool_and_render_paths(self) -> None:
        executor, _, _ = _make_executor()
        live_text, fallback, header_html, remaining = executor._render_live_text(
            current_reasoning="Reasoning",
            current_content="Answer",
            committed_content_offset=0,
            show_reasoning=True,
            reasoning_committed=False,
        ) or ("", "", "", "")

        assert header_html
        assert remaining == "Answer"
        assert fallback.endswith(" ▍")
        assert live_text.endswith(" ▍")
        assert (
            executor._render_live_text(
                current_reasoning="Only reasoning",
                current_content="",
                committed_content_offset=0,
                show_reasoning=True,
                reasoning_committed=False,
            )
            or ("", "", "", "")
        )[0].endswith(" ▍")
        assert (
            executor._render_live_text(
                current_reasoning="",
                current_content="Only content",
                committed_content_offset=0,
                show_reasoning=False,
                reasoning_committed=False,
            )
            or ("", "", "", "")
        )[0].endswith(" ▍")
        assert (
            executor._render_live_text(
                current_reasoning="",
                current_content="   ",
                committed_content_offset=0,
                show_reasoning=False,
                reasoning_committed=False,
            )
            is None
        )
        assert ToolActivityNotifier.tool_call_lines([]) == []
        assert (
            ToolActivityNotifier.tool_result_text(
                tool_name="wiki_create",
                result="ok",
                error=None,
            )
            == "✅ wiki_create: ok"
        )
        request = _make_request(show_reasoning=False)
        outcome = _StreamOutcome(
            response_text="Answer",
            response_reasoning="Reasoning",
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
        )
        assert ConversationExecutor._final_header_html(request, outcome) == ""
        request = _make_request(show_reasoning=True)
        outcome = _StreamOutcome(
            response_text="Answer",
            response_reasoning=" ",
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
        )
        assert ConversationExecutor._final_header_html(request, outcome) == ""
        outcome = _StreamOutcome(
            response_text="Answer",
            response_reasoning="Reasoning",
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=True,
            usage=None,
            cost=None,
            is_byok=False,
        )
        assert ConversationExecutor._final_header_html(request, outcome) == ""


class TestTierCorrection:
    """Tests for the two-tier correction logic in _stream_with_validation."""

    async def test_tier1_regex_sanitization_fixes_output(self) -> None:
        """Tier 1 regex sanitization should fix malformed XML before parsing."""
        executor, _, _ = _make_executor()

        from mai_gram.response_templates.registry import get_template

        template = get_template("xml")
        request = _make_request()

        malformed = "<thought>thinking</thought>\n<content>answer<///content>"
        stream_outcome = _StreamOutcome(
            response_text=malformed,
            response_reasoning=None,
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
        )

        with patch.object(
            executor,
            "_stream_response",
            AsyncMock(return_value=stream_outcome),
        ):
            outcome, parsed = await executor._stream_with_validation(
                request,
                [],
                template=template,
                total_attempts=1,
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        assert "</content>" in outcome.response_text
        assert "<///content>" not in outcome.response_text
        assert parsed.fields["content"] == "answer"

    async def test_tier2_llm_repair_called_when_tier1_insufficient(self) -> None:
        """When Tier 1 cannot fix the output, Tier 2 LLM repair is invoked."""
        executor, _, _ = _make_executor()
        executor._format_repair_config = {
            "model": "openrouter/free",
            "temperature": 0.0,
            "max_tokens": 8192,
            "enabled": True,
        }

        from mai_gram.response_templates.registry import get_template

        template = get_template("xml")
        request = _make_request()

        # Missing <thought> entirely -- regex can't fix this
        malformed = "<content>answer</content>"
        stream_outcome = _StreamOutcome(
            response_text=malformed,
            response_reasoning=None,
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
        )

        repaired_text = "<thought>thinking</thought>\n<content>answer</content>"
        with (
            patch.object(
                executor,
                "_stream_response",
                AsyncMock(return_value=stream_outcome),
            ),
            patch(
                "mai_gram.bot.conversation_executor.llm_repair",
                AsyncMock(return_value=repaired_text),
            ) as mock_repair,
        ):
            outcome, parsed = await executor._stream_with_validation(
                request,
                [],
                template=template,
                total_attempts=1,
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        mock_repair.assert_awaited_once()
        assert outcome.response_text == repaired_text
        assert parsed.fields["thought"] == "thinking"
        assert parsed.fields["content"] == "answer"

    async def test_tier2_disabled_skips_llm_repair(self) -> None:
        """When format_repair is disabled, Tier 2 is skipped."""
        executor, _, _ = _make_executor()
        executor._format_repair_config = {
            "model": "openrouter/free",
            "temperature": 0.0,
            "max_tokens": 8192,
            "enabled": False,
        }

        from mai_gram.response_templates.registry import get_template

        template = get_template("xml")
        request = _make_request()

        malformed = "<content>answer</content>"
        stream_outcome = _StreamOutcome(
            response_text=malformed,
            response_reasoning=None,
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
        )

        with (
            patch.object(
                executor,
                "_stream_response",
                AsyncMock(return_value=stream_outcome),
            ),
            patch(
                "mai_gram.bot.conversation_executor.llm_repair",
                AsyncMock(),
            ) as mock_repair,
            pytest.raises(LLMProviderError, match="Format error"),
        ):
            await executor._stream_with_validation(
                request,
                [],
                template=template,
                total_attempts=1,
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        mock_repair.assert_not_awaited()

    async def test_finish_reason_length_skips_repair_and_retries(self) -> None:
        """Responses truncated by token limit bypass repair tiers entirely."""
        executor, messenger, _ = _make_executor()

        from mai_gram.response_templates.registry import get_template

        template = get_template("xml")
        request = _make_request()

        truncated_outcome = _StreamOutcome(
            response_text="//_api:search_" * 100,
            response_reasoning=None,
            placeholder_msg_id="ph-1",
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
            finish_reason="length",
        )
        good_outcome = _StreamOutcome(
            response_text="<thought>thinking</thought>\n<content>answer</content>",
            response_reasoning=None,
            placeholder_msg_id="ph-2",
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
            finish_reason="stop",
        )

        stream_mock = AsyncMock(side_effect=[truncated_outcome, good_outcome])
        with (
            patch.object(executor, "_stream_response", stream_mock),
            patch(
                "mai_gram.bot.conversation_executor.llm_repair",
                AsyncMock(),
            ) as mock_repair,
        ):
            outcome, parsed = await executor._stream_with_validation(
                request,
                [],
                template=template,
                total_attempts=3,
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )

        assert stream_mock.await_count == 2
        mock_repair.assert_not_awaited()
        assert outcome.finish_reason == "stop"
        assert parsed.fields["content"] == "answer"
        messenger.edit_message.assert_awaited_with(
            request.telegram_chat_id,
            "ph-1",
            "\u26a0\ufe0f Response truncated, retrying...",
        )

    async def test_finish_reason_length_all_attempts_raises(self) -> None:
        """All attempts truncated raises LLMProviderError with length message."""
        executor, _, _ = _make_executor()

        from mai_gram.response_templates.registry import get_template

        template = get_template("xml")
        request = _make_request()

        truncated = _StreamOutcome(
            response_text="garbage" * 50,
            response_reasoning=None,
            placeholder_msg_id=None,
            committed_content_offset=0,
            reasoning_committed=False,
            usage=None,
            cost=None,
            is_byok=False,
            finish_reason="length",
        )

        with (
            patch.object(
                executor,
                "_stream_response",
                AsyncMock(return_value=truncated),
            ),
            pytest.raises(LLMProviderError, match="truncated by token limit"),
        ):
            await executor._stream_with_validation(
                request,
                [],
                template=template,
                total_attempts=2,
                on_tool_call_display=AsyncMock(),
                on_tool_result_display=AsyncMock(),
            )
