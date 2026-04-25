"""Unit tests for the shared conversation executor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mai_gram.bot.conversation_executor import (
    AssistantTurnRequest,
    ConversationExecutor,
    _StreamOutcome,
    _StreamState,
)
from mai_gram.db.models import Chat
from mai_gram.llm.provider import LLMProviderError
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
    )


def _make_executor() -> tuple[ConversationExecutor, MagicMock, MagicMock]:
    messenger = MagicMock()
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

        tool_call_cb, tool_result_cb = executor._build_tool_callbacks(request, sent_msg_ids)

        await tool_call_cb(
            content="",
            tool_calls_json='[{"name":"wiki_create","arguments":{"title":"Test"}}]',
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
        assert sent_msg_ids == ["tool-call-id", "tool-result-id"]

    async def test_tool_display_helpers_cover_guard_and_fallback_branches(self) -> None:
        executor, messenger, _ = _make_executor()
        hidden_request = _make_request(show_tool_calls=False)
        visible_request = _make_request(show_tool_calls=True)
        sent_msg_ids: list[str] = []

        await executor._maybe_send_tool_call_display(hidden_request, sent_msg_ids, "[]")
        await executor._maybe_send_tool_result_display(
            hidden_request,
            sent_msg_ids,
            tool_name="wiki_create",
            result=None,
            error=None,
        )
        await executor._maybe_send_tool_call_display(visible_request, sent_msg_ids, "[]")

        assert messenger.send_message.await_count == 0
        assert ConversationExecutor._tool_call_lines(
            '[{"name":"wiki_create","arguments":"{bad json"}]'
        ) == ["🔧 wiki_create({bad json)"]
        assert (
            ConversationExecutor._tool_result_text(
                tool_name="wiki_create",
                result=None,
                error="boom",
            )
            == "❌ wiki_create: boom"
        )
        assert ConversationExecutor._tool_result_text(
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
            )
            yield SimpleNamespace(
                usage=None,
                cost=None,
                is_byok=False,
                turn_complete=True,
                reasoning="",
                content="",
            )
            yield SimpleNamespace(
                usage=None,
                cost=None,
                is_byok=False,
                turn_complete=False,
                reasoning="",
                content="Second",
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
        executor, _, renderer = _make_executor()
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
            keyboard=renderer._send_response.await_args.kwargs["keyboard"],
        )
        assert sent_msg_ids == ["final-id"]

    def test_helper_formatters_cover_tool_and_render_paths(self) -> None:
        live_text, fallback, header_html, remaining = ConversationExecutor._render_live_text(
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
            ConversationExecutor._render_live_text(
                current_reasoning="Only reasoning",
                current_content="",
                committed_content_offset=0,
                show_reasoning=True,
                reasoning_committed=False,
            )
            or ("", "", "", "")
        )[0].endswith(" ▍")
        assert (
            ConversationExecutor._render_live_text(
                current_reasoning="",
                current_content="Only content",
                committed_content_offset=0,
                show_reasoning=False,
                reasoning_committed=False,
            )
            or ("", "", "", "")
        )[0].endswith(" ▍")
        assert (
            ConversationExecutor._render_live_text(
                current_reasoning="",
                current_content="   ",
                committed_content_offset=0,
                show_reasoning=False,
                reasoning_committed=False,
            )
            is None
        )
        assert ConversationExecutor._tool_call_lines("not-json") == []
        assert (
            ConversationExecutor._tool_result_text(
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
