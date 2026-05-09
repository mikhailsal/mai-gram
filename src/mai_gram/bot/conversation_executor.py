"""Shared assistant-turn execution for conversation and regenerate flows."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from mai_gram.bot.executor_types import (
    AssistantTurnRequest,
    AssistantTurnResult,
    ConversationRenderer,
    StreamOutcome,
    StreamState,
    build_structured_parts,
    parse_hidden_fields,
    parse_template_params,
    replace_response_text,
)
from mai_gram.bot.tool_activity_notifier import ToolActivityNotifier
from mai_gram.llm.provider import LLMError, LLMProviderError
from mai_gram.mcp_servers.bridge import run_with_tools_stream
from mai_gram.messenger.base import OutgoingMessage
from mai_gram.response_templates._sanitize import llm_repair
from mai_gram.response_templates.registry import get_template

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mai_gram.llm.provider import LLMProvider
    from mai_gram.messenger.base import Messenger
    from mai_gram.response_templates.base import ParsedResponse, ResponseTemplate

logger = logging.getLogger(__name__)

__all__ = [
    "AssistantTurnRequest",
    "AssistantTurnResult",
    "ConversationExecutor",
    "ConversationRenderer",
    "StreamOutcome",
    "StreamState",
]


class ConversationExecutor:
    """Execute one assistant turn against the shared LLM/tool pipeline."""

    def __init__(
        self,
        messenger: Messenger,
        llm_provider: LLMProvider,
        *,
        tool_max_iterations: int,
        renderer: ConversationRenderer,
        format_repair_config: dict[str, Any] | None = None,
    ) -> None:
        self._messenger = messenger
        self._llm = llm_provider
        self._tool_max_iterations = tool_max_iterations
        self._renderer = renderer
        self._tool_activity = ToolActivityNotifier(messenger)
        self._format_repair_config = format_repair_config or {
            "model": "openrouter/free",
            "temperature": 0.0,
            "max_tokens": 8192,
            "enabled": True,
        }
        from mai_gram.bot.stream_display import StreamDisplayManager

        self._stream_display = StreamDisplayManager(messenger, renderer)

    async def execute(
        self,
        request: AssistantTurnRequest,
        *,
        max_format_retries: int = 2,
    ) -> AssistantTurnResult:
        sent_msg_ids: list[str] = []
        tool_call_cb, tool_result_cb = self._tool_activity.build_callbacks(request, sent_msg_ids)
        tpl_params = parse_template_params(request.chat)
        template = get_template(request.chat.response_template, tpl_params)
        total_attempts = 1 + max_format_retries

        try:
            outcome, parsed = await self._stream_with_validation(
                request,
                sent_msg_ids,
                template=template,
                total_attempts=total_attempts,
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
        except (LLMError, RuntimeError, OSError, asyncio.TimeoutError) as exc:
            logger.exception(request.failure_log_message)
            error_text = self._renderer._user_friendly_error(exc)
            await self._deliver_error(request, sent_msg_ids, error_text)
            return AssistantTurnResult(sent_message_ids=sent_msg_ids)

        await self._finalize_response(
            request,
            sent_msg_ids,
            outcome,
            saved_msg.id,
            parsed=parsed,
            template=template,
        )
        return AssistantTurnResult(sent_message_ids=sent_msg_ids)

    async def _stream_with_validation(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        *,
        template: ResponseTemplate,
        total_attempts: int,
        on_tool_call_display: Callable[..., Awaitable[None]],
        on_tool_result_display: Callable[..., Awaitable[None]],
    ) -> tuple[StreamOutcome, ParsedResponse]:
        """Stream response and validate against template, retrying on failure."""
        last_errors: list[str] = []
        prefill = template.assistant_prefill() or ""

        for attempt in range(1, total_attempts + 1):
            outcome = await self._stream_response(
                request,
                sent_msg_ids,
                template=template,
                on_tool_call_display=on_tool_call_display,
                on_tool_result_display=on_tool_result_display,
            )
            if prefill:
                outcome = replace_response_text(outcome, prefill + outcome.response_text)

            if outcome.finish_reason == "length":
                last_errors = await self._handle_truncated(
                    request,
                    outcome,
                    attempt,
                    total_attempts,
                )
                continue

            outcome, parsed, errors = self._apply_tier1(request, outcome, template)
            if not errors:
                return outcome, parsed

            outcome, parsed, errors = await self._apply_tier2(request, outcome, template, errors)
            if not errors:
                return outcome, parsed

            last_errors = errors
            await self._notify_validation_fail(request, outcome, attempt, total_attempts, errors)

        raise LLMProviderError(
            f"Format error after {total_attempts} attempts: {'; '.join(last_errors)}"
        )

    async def _handle_truncated(
        self,
        request: AssistantTurnRequest,
        outcome: StreamOutcome,
        attempt: int,
        total_attempts: int,
    ) -> list[str]:
        logger.warning(
            "Response truncated (finish_reason=length) on attempt %d/%d for chat %s",
            attempt,
            total_attempts,
            request.chat.id,
        )
        if attempt < total_attempts and outcome.placeholder_msg_id:
            await self._messenger.edit_message(
                request.telegram_chat_id,
                outcome.placeholder_msg_id,
                "\u26a0\ufe0f Response truncated, retrying...",
            )
        return ["response truncated by token limit (finish_reason=length)"]

    def _apply_tier1(
        self,
        request: AssistantTurnRequest,
        outcome: StreamOutcome,
        template: ResponseTemplate,
    ) -> tuple[StreamOutcome, ParsedResponse, list[str]]:
        sanitized = template.sanitize(outcome.response_text)
        if sanitized != outcome.response_text:
            logger.info("Tier 1 (regex) corrected format for chat %s", request.chat.id)
            outcome = replace_response_text(outcome, sanitized)
        parsed = template.parse(outcome.response_text)
        return outcome, parsed, template.validate(parsed)

    async def _apply_tier2(
        self,
        request: AssistantTurnRequest,
        outcome: StreamOutcome,
        template: ResponseTemplate,
        errors: list[str],
    ) -> tuple[StreamOutcome, ParsedResponse, list[str]]:
        cfg = self._format_repair_config
        spec = template.llm_repair_prompt()
        if not (cfg["enabled"] and spec):
            return outcome, template.parse(outcome.response_text), errors

        logger.info("Tier 2 (LLM repair) for chat %s: %s", request.chat.id, "; ".join(errors))
        repaired = await llm_repair(
            self._llm,
            outcome.response_text,
            spec,
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            extra_params=cfg.get("extra_params"),
        )
        if repaired != outcome.response_text:
            outcome = replace_response_text(outcome, repaired)
            parsed = template.parse(outcome.response_text)
            errors = template.validate(parsed)
            if not errors:
                logger.info("Tier 2 (LLM) fixed format for chat %s", request.chat.id)
        return outcome, template.parse(outcome.response_text), errors

    async def _notify_validation_fail(
        self,
        request: AssistantTurnRequest,
        outcome: StreamOutcome,
        attempt: int,
        total_attempts: int,
        errors: list[str],
    ) -> None:
        logger.warning(
            "Template validation failed (%d/%d) for chat %s: %s",
            attempt,
            total_attempts,
            request.chat.id,
            "; ".join(errors),
        )
        if attempt < total_attempts and outcome.placeholder_msg_id:
            await self._messenger.edit_message(
                request.telegram_chat_id,
                outcome.placeholder_msg_id,
                "\u26a0\ufe0f Format error, retrying...",
            )

    async def _stream_response(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        *,
        template: ResponseTemplate | None = None,
        on_tool_call_display: Callable[..., Awaitable[None]],
        on_tool_result_display: Callable[..., Awaitable[None]],
    ) -> StreamOutcome:
        state = StreamState()
        stream_usage = None
        stream_cost: float | None = None
        stream_is_byok = False
        stream_finish_reason: str | None = None

        async for chunk in run_with_tools_stream(
            self._llm,
            request.mcp_manager,
            request.llm_messages,
            model=request.model_for_api,
            max_iterations=self._tool_max_iterations,
            extra_params=request.extra_params or None,
            on_assistant_tool_call=on_tool_call_display,
            on_tool_result=on_tool_result_display,
        ):
            if chunk.usage is not None:
                stream_usage = chunk.usage
                stream_cost = chunk.cost
                stream_is_byok = chunk.is_byok
            if chunk.finish_reason:
                stream_finish_reason = chunk.finish_reason
            if chunk.turn_complete:
                await self._stream_display.handle_turn_complete(request, state, sent_msg_ids)
                continue
            if chunk.reasoning:
                state.reasoning_parts.append(chunk.reasoning)
            if chunk.content:
                state.content_parts.append(chunk.content)
            await self._stream_display.maybe_update_live_display(
                request,
                state,
                sent_msg_ids,
                template=template,
            )

        response_text = "".join(state.content_parts)
        if not response_text or not response_text.strip():
            raise LLMProviderError("The model returned an empty response")
        return StreamOutcome(
            response_text=response_text,
            response_reasoning="".join(state.reasoning_parts) or None,
            placeholder_msg_id=state.placeholder_msg_id,
            committed_content_offset=state.committed_content_offset,
            reasoning_committed=state.reasoning_committed,
            usage=stream_usage,
            cost=stream_cost,
            is_byok=stream_is_byok,
            finish_reason=stream_finish_reason,
        )

    async def _deliver_error(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        error_text: str,
    ) -> None:
        kb = self._messenger.build_inline_keyboard([[("🔄 Regenerate", "regen")]])
        await self._renderer._deliver_error(
            request.telegram_chat_id,
            error_text,
            placeholder_msg_id=None,
            keyboard=kb,
            sent_msg_ids=sent_msg_ids,
        )

    async def _finalize_response(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: StreamOutcome,
        saved_msg_id: int,
        *,
        parsed: ParsedResponse | None = None,
        template: ResponseTemplate | None = None,
    ) -> None:
        kb = self._build_action_keyboard(saved_msg_id)
        footer = self._renderer._format_usage_footer(outcome.usage, outcome.cost, outcome.is_byok)
        has_structured = (
            template is not None
            and parsed is not None
            and template.name != "empty"
            and len(parsed.fields) > 1
        )
        if has_structured and template is not None and parsed is not None:
            await self._finalize_structured(
                request,
                sent_msg_ids,
                outcome,
                parsed=parsed,
                template=template,
                usage_footer=footer,
                keyboard=kb,
            )
        elif outcome.placeholder_msg_id and outcome.response_text.strip():
            await self._finalize_with_placeholder(request, sent_msg_ids, outcome, footer, kb)
        else:
            await self._finalize_fresh(request, sent_msg_ids, outcome, footer, kb)

    def _build_action_keyboard(self, saved_msg_id: int) -> object:
        return self._messenger.build_inline_keyboard(
            [
                [
                    ("🔄 Regenerate", f"regen:{saved_msg_id}"),
                    ("✂ Cut this & above", f"cut:{saved_msg_id}"),
                ]
            ]
        )

    async def _finalize_with_placeholder(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: StreamOutcome,
        footer: str,
        kb: object,
    ) -> None:
        remaining = outcome.response_text[outcome.committed_content_offset :]
        if footer:
            remaining += f"\n\n{footer}"
        header = self._final_header_html(request, outcome)
        extra = await self._renderer._finalize_placeholder(
            request.telegram_chat_id,
            outcome.placeholder_msg_id,  # type: ignore[arg-type]
            remaining,
            header_html=header,
            keyboard=kb,
        )
        sent_msg_ids.extend(extra)
        sent_msg_ids.append(outcome.placeholder_msg_id)  # type: ignore[arg-type]

    async def _finalize_fresh(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: StreamOutcome,
        footer: str,
        kb: object,
    ) -> None:
        text = outcome.response_text + (f"\n\n{footer}" if footer else "")
        ids = await self._renderer._send_response(
            request.telegram_chat_id,
            response_text=text,
            response_reasoning=outcome.response_reasoning,
            show_reasoning=request.show_reasoning,
            keyboard=kb,
        )
        sent_msg_ids.extend(ids)

    async def _finalize_structured(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: StreamOutcome,
        *,
        parsed: ParsedResponse,
        template: ResponseTemplate,
        usage_footer: str,
        keyboard: object,
    ) -> None:
        hidden = parse_hidden_fields(request.chat.hidden_template_fields)
        header_html, content_text = build_structured_parts(template, parsed, hidden)
        if outcome.placeholder_msg_id:
            sent_msg_ids.append(outcome.placeholder_msg_id)
        if content_text:
            if usage_footer:
                content_text += f"\n\n{usage_footer}"
            await self._send_structured(
                request,
                sent_msg_ids,
                outcome,
                header_html=header_html,
                content_text=content_text,
                keyboard=keyboard,
            )

    async def _send_structured(
        self,
        request: AssistantTurnRequest,
        sent_msg_ids: list[str],
        outcome: StreamOutcome,
        *,
        header_html: str,
        content_text: str,
        keyboard: object,
    ) -> None:
        if outcome.placeholder_msg_id:
            extra = await self._renderer._finalize_placeholder(
                request.telegram_chat_id,
                outcome.placeholder_msg_id,
                content_text,
                header_html=header_html,
                keyboard=keyboard,
            )
            sent_msg_ids.extend(extra)
        else:
            ids = await self._renderer._send_response(
                request.telegram_chat_id,
                response_text=content_text,
                response_reasoning=None,
                show_reasoning=False,
                keyboard=keyboard,
            )
            if header_html and ids:
                r = await self._messenger.send_message(
                    OutgoingMessage(
                        text=header_html,
                        chat_id=request.telegram_chat_id,
                        parse_mode="html",
                    )
                )
                if r.success and r.message_id:
                    sent_msg_ids.append(r.message_id)
            sent_msg_ids.extend(ids)

    @staticmethod
    def _final_header_html(request: AssistantTurnRequest, outcome: StreamOutcome) -> str:
        if not request.show_reasoning:
            return ""
        if not outcome.response_reasoning or not outcome.response_reasoning.strip():
            return ""
        if outcome.reasoning_committed:
            return ""
        from mai_gram.core.md_to_telegram import format_reasoning_html

        return format_reasoning_html(outcome.response_reasoning, expandable=True)
