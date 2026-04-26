"""Shared assistant-turn execution for conversation and regenerate flows."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mai_gram.bot.tool_activity_notifier import ToolActivityNotifier
from mai_gram.llm.provider import LLMProviderError
from mai_gram.mcp_servers.bridge import run_with_tools_stream
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.db.models import Chat
    from mai_gram.llm.provider import ChatMessage, LLMProvider
    from mai_gram.mcp_servers.manager import MCPManager
    from mai_gram.memory.messages import MessageStore
    from mai_gram.messenger.base import Messenger

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AssistantTurnRequest:
    """Prepared inputs for one assistant response generation pass."""

    chat: Chat
    message_store: MessageStore
    mcp_manager: MCPManager
    llm_messages: list[ChatMessage]
    telegram_chat_id: str
    timezone_name: str
    show_datetime: bool
    show_reasoning: bool
    show_tool_calls: bool
    extra_params: dict[str, Any] | None
    failure_log_message: str


@dataclass(frozen=True, slots=True)
class AssistantTurnResult:
    """Result of running one assistant response generation pass."""

    sent_message_ids: list[str]


@dataclass(slots=True)
class _StreamState:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    placeholder_msg_id: str | None = None
    last_edit_time: float = 0.0
    last_display_len: int = 0
    committed_content_offset: int = 0
    reasoning_committed: bool = False


@dataclass(frozen=True, slots=True)
class _StreamOutcome:
    response_text: str
    response_reasoning: str | None
    placeholder_msg_id: str | None
    committed_content_offset: int
    reasoning_committed: bool
    usage: object | None
    cost: float | None
    is_byok: bool


class ConversationRenderer(Protocol):
    def _build_intermediate_display(
        self,
        content: str,
        reasoning: str,
        show_reasoning: bool,
    ) -> str: ...

    def _format_usage_footer(self, usage: object, cost: float | None, is_byok: bool) -> str: ...

    def _user_friendly_error(self, exc: BaseException) -> str: ...

    async def _deliver_error(
        self,
        chat_id: str,
        error_text: str,
        *,
        placeholder_msg_id: str | None,
        keyboard: object | None = None,
        sent_msg_ids: list[str],
        max_attempts: int = 5,
    ) -> None: ...

    async def _commit_overflow(
        self,
        *,
        tg_chat_id: str,
        header_html: str,
        reasoning_committed: bool,
        placeholder_msg_id: str | None,
        sent_msg_ids: list[str],
        remaining_content: str,
        current_content: str,
        committed_content_offset: int,
    ) -> tuple[int, str | None]: ...

    async def _finalize_placeholder(
        self,
        chat_id: str,
        placeholder_msg_id: str,
        raw_text: str,
        *,
        header_html: str = "",
        keyboard: object = None,
    ) -> list[str]: ...

    async def _send_response(
        self,
        chat_id: str,
        *,
        response_text: str | None,
        response_reasoning: str | None = None,
        show_reasoning: bool = False,
        keyboard: object = None,
    ) -> list[str]: ...


class ConversationExecutor:
    """Execute one assistant turn against the shared LLM/tool pipeline."""

    def __init__(
        self,
        messenger: Messenger,
        llm_provider: LLMProvider,
        *,
        tool_max_iterations: int,
        renderer: ConversationRenderer,
    ) -> None:
        self._messenger = messenger
        self._llm = llm_provider
        self._tool_max_iterations = tool_max_iterations
        self._renderer = renderer
        self._tool_activity = ToolActivityNotifier(messenger)

    async def execute(self, request: AssistantTurnRequest) -> AssistantTurnResult:
        sent_msg_ids: list[str] = []
        tool_call_cb, tool_result_cb = self._tool_activity.build_callbacks(request, sent_msg_ids)

        try:
            outcome = await self._stream_response(
                request,
                sent_msg_ids,
                on_tool_call_display=tool_call_cb,
                on_tool_result_display=tool_result_cb,
            )
            saved_msg = await request.message_store.save_message(
                request.chat.id,
                "assistant",
                outcome.response_text,
                reasoning=outcome.response_reasoning,
                timezone_name=request.timezone_name,
                show_datetime=request.show_datetime,
            )
        except Exception as exc:
            logger.exception(request.failure_log_message)
            error_text = self._renderer._user_friendly_error(exc)
            await self._deliver_error(request, sent_msg_ids, error_text)
            return AssistantTurnResult(sent_message_ids=sent_msg_ids)

        await self._finalize_response(
            request,
            sent_msg_ids,
            outcome,
            saved_msg.id,
        )
        return AssistantTurnResult(sent_message_ids=sent_msg_ids)

    async def _stream_response(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        *,
        on_tool_call_display: Callable[..., Awaitable[None]],
        on_tool_result_display: Callable[..., Awaitable[None]],
    ) -> _StreamOutcome:
        state = _StreamState()
        stream_usage = None
        stream_cost: float | None = None
        stream_is_byok = False

        async for chunk in run_with_tools_stream(
            self._llm,
            request.mcp_manager,
            request.llm_messages,
            model=request.chat.llm_model,
            max_iterations=self._tool_max_iterations,
            extra_params=request.extra_params or None,
            on_assistant_tool_call=on_tool_call_display,
            on_tool_result=on_tool_result_display,
        ):
            if chunk.usage is not None:
                stream_usage = chunk.usage
                stream_cost = chunk.cost
                stream_is_byok = chunk.is_byok
            if chunk.turn_complete:
                await self._handle_turn_complete(request, state, sent_msg_ids)
                continue
            if chunk.reasoning:
                state.reasoning_parts.append(chunk.reasoning)
            if chunk.content:
                state.content_parts.append(chunk.content)
            await self._maybe_update_live_display(request, state, sent_msg_ids)

        response_text = "".join(state.content_parts)
        if not response_text or not response_text.strip():
            raise LLMProviderError("The model returned an empty response")
        return _StreamOutcome(
            response_text=response_text,
            response_reasoning="".join(state.reasoning_parts) or None,
            placeholder_msg_id=state.placeholder_msg_id,
            committed_content_offset=state.committed_content_offset,
            reasoning_committed=state.reasoning_committed,
            usage=stream_usage,
            cost=stream_cost,
            is_byok=stream_is_byok,
        )

    async def _handle_turn_complete(
        self,
        request: AssistantTurnRequest,
        state: _StreamState,
        sent_msg_ids: list[str],
    ) -> None:
        if state.placeholder_msg_id:
            turn_text = self._renderer._build_intermediate_display(
                "".join(state.content_parts),
                "".join(state.reasoning_parts),
                request.show_reasoning,
            )
            if turn_text.strip():
                await self._messenger.edit_message(
                    request.telegram_chat_id,
                    state.placeholder_msg_id,
                    turn_text,
                )
            sent_msg_ids.append(state.placeholder_msg_id)
        state.content_parts.clear()
        state.reasoning_parts.clear()
        state.placeholder_msg_id = None
        state.last_edit_time = 0.0
        state.last_display_len = 0
        state.committed_content_offset = 0
        state.reasoning_committed = False

    async def _maybe_update_live_display(
        self,
        request: AssistantTurnRequest,
        state: _StreamState,
        sent_msg_ids: list[str],
    ) -> None:
        current_reasoning = "".join(state.reasoning_parts)
        current_content = "".join(state.content_parts)
        display_len = len(current_reasoning) + len(current_content)
        now_mono = time.monotonic()
        chars_since_edit = display_len - state.last_display_len
        time_since_edit = now_mono - state.last_edit_time

        should_edit = (
            (current_content.strip() or current_reasoning.strip())
            and (time_since_edit >= 1.0 or chars_since_edit >= 60)
            and chars_since_edit > 0
        )
        if not should_edit:
            return

        rendered = self._render_live_text(
            current_reasoning=current_reasoning,
            current_content=current_content,
            committed_content_offset=state.committed_content_offset,
            show_reasoning=request.show_reasoning,
            reasoning_committed=state.reasoning_committed,
        )
        if rendered is None:
            return

        live_text, fallback, header_html, remaining_content = rendered
        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        if len(live_text) > SAFE_MAX_LENGTH:
            (
                state.committed_content_offset,
                state.placeholder_msg_id,
            ) = await self._renderer._commit_overflow(
                tg_chat_id=request.telegram_chat_id,
                header_html=header_html,
                reasoning_committed=state.reasoning_committed,
                placeholder_msg_id=state.placeholder_msg_id,
                sent_msg_ids=sent_msg_ids,
                remaining_content=remaining_content,
                current_content=current_content,
                committed_content_offset=state.committed_content_offset,
            )
            if header_html and not state.reasoning_committed:
                state.reasoning_committed = True
        else:
            state.placeholder_msg_id = await self._send_or_edit_placeholder(
                request,
                placeholder_msg_id=state.placeholder_msg_id,
                live_text=live_text,
                fallback=fallback,
            )

        state.last_edit_time = now_mono
        state.last_display_len = display_len

    @staticmethod
    def _render_live_text(
        *,
        current_reasoning: str,
        current_content: str,
        committed_content_offset: int,
        show_reasoning: bool,
        reasoning_committed: bool,
    ) -> tuple[str, str, str, str] | None:
        from mai_gram.core.md_to_telegram import format_reasoning_html, markdown_to_html

        remaining_content = current_content[committed_content_offset:]
        header_html = ""
        if show_reasoning and current_reasoning.strip() and not reasoning_committed:
            header_html = format_reasoning_html(current_reasoning)
        content_html = markdown_to_html(remaining_content) if remaining_content.strip() else ""
        if header_html and content_html:
            live_text = header_html + "\n\n" + content_html + " ▍"
        elif header_html:
            live_text = header_html + " ▍"
        elif content_html:
            live_text = content_html + " ▍"
        else:
            return None

        from mai_gram.core.telegram_limits import SAFE_MAX_LENGTH

        raw_fallback = remaining_content or current_reasoning
        fallback = raw_fallback[:SAFE_MAX_LENGTH] + " ▍"
        return live_text, fallback, header_html, remaining_content

    async def _send_or_edit_placeholder(
        self,
        request: AssistantTurnRequest,
        *,
        placeholder_msg_id: str | None,
        live_text: str,
        fallback: str,
    ) -> str | None:
        if placeholder_msg_id is None:
            result = await self._messenger.send_message(
                OutgoingMessage(
                    text=live_text,
                    chat_id=request.telegram_chat_id,
                    parse_mode="html",
                )
            )
            if not result.success:
                result = await self._messenger.send_message(
                    OutgoingMessage(text=fallback, chat_id=request.telegram_chat_id)
                )
            return result.message_id if result.success else None

        edit_result = await self._messenger.edit_message(
            request.telegram_chat_id,
            placeholder_msg_id,
            live_text,
            parse_mode="html",
        )
        if not edit_result.success:
            await self._messenger.edit_message(
                request.telegram_chat_id,
                placeholder_msg_id,
                fallback,
            )
        return placeholder_msg_id

    async def _deliver_error(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        error_text: str,
    ) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        error_keyboard = build_inline_keyboard([[("🔄 Regenerate", "regen")]])
        await self._renderer._deliver_error(
            request.telegram_chat_id,
            error_text,
            placeholder_msg_id=None,
            keyboard=error_keyboard,
            sent_msg_ids=sent_msg_ids,
        )

    async def _finalize_response(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: _StreamOutcome,
        saved_msg_id: int,
    ) -> None:
        from mai_gram.messenger.telegram import build_inline_keyboard

        keyboard_buttons = [
            [("🔄 Regenerate", "regen"), ("✂ Cut this & above", f"cut:{saved_msg_id}")]
        ]
        action_keyboard = build_inline_keyboard(keyboard_buttons)
        usage_footer = self._renderer._format_usage_footer(
            outcome.usage,
            outcome.cost,
            outcome.is_byok,
        )

        if outcome.placeholder_msg_id and outcome.response_text.strip():
            remaining_raw = outcome.response_text[outcome.committed_content_offset :]
            if usage_footer:
                remaining_raw += f"\n\n{usage_footer}"
            header_html = self._final_header_html(request, outcome)
            extra_ids = await self._renderer._finalize_placeholder(
                request.telegram_chat_id,
                outcome.placeholder_msg_id,
                remaining_raw,
                header_html=header_html,
                keyboard=action_keyboard,
            )
            sent_msg_ids.extend(extra_ids)
            sent_msg_ids.append(outcome.placeholder_msg_id)
            return

        text_with_footer = outcome.response_text
        if usage_footer:
            text_with_footer += f"\n\n{usage_footer}"
        final_msg_ids = await self._renderer._send_response(
            request.telegram_chat_id,
            response_text=text_with_footer,
            response_reasoning=outcome.response_reasoning,
            show_reasoning=request.show_reasoning,
            keyboard=action_keyboard,
        )
        sent_msg_ids.extend(final_msg_ids)

    @staticmethod
    def _final_header_html(request: AssistantTurnRequest, outcome: _StreamOutcome) -> str:
        if not request.show_reasoning:
            return ""
        if not outcome.response_reasoning or not outcome.response_reasoning.strip():
            return ""
        if outcome.reasoning_committed:
            return ""
        from mai_gram.core.md_to_telegram import format_reasoning_html

        return format_reasoning_html(outcome.response_reasoning, expandable=True)
