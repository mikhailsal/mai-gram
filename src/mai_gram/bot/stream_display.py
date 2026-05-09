"""Live streaming display logic for assistant turns.

Handles the real-time message editing, overflow commit, and placeholder
management during LLM response streaming.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from mai_gram.bot.executor_types import (
        AssistantTurnRequest,
        ConversationRenderer,
        StreamState,
    )
    from mai_gram.messenger.base import Messenger
    from mai_gram.response_templates.base import ResponseTemplate, StreamingParseResult

logger = logging.getLogger(__name__)


class StreamDisplayManager:
    """Manage live display updates during LLM response streaming."""

    def __init__(self, messenger: Messenger, renderer: ConversationRenderer) -> None:
        self._messenger = messenger
        self._renderer = renderer

    async def handle_turn_complete(
        self,
        request: AssistantTurnRequest,
        state: StreamState,
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
        state.template_fields_committed.clear()

    async def maybe_update_live_display(
        self,
        request: AssistantTurnRequest,
        state: StreamState,
        sent_msg_ids: list[str],
        *,
        template: ResponseTemplate | None = None,
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

        rendered = self._choose_renderer(
            request,
            state,
            template,
            current_reasoning,
            current_content,
        )
        if rendered is None:
            return

        await self._apply_rendered(request, state, sent_msg_ids, rendered, current_content)
        state.last_edit_time = now_mono
        state.last_display_len = display_len

    def _choose_renderer(
        self,
        request: AssistantTurnRequest,
        state: StreamState,
        template: ResponseTemplate | None,
        current_reasoning: str,
        current_content: str,
    ) -> tuple[str, str, str, str] | None:
        use_template = (
            template is not None and template.name != "empty" and not current_reasoning.strip()
        )
        if use_template:
            return self._render_template_live_text(
                template=template,  # type: ignore[arg-type]
                current_content=current_content,
                state=state,
            )
        return self._render_live_text(
            current_reasoning=current_reasoning,
            current_content=current_content,
            committed_content_offset=state.committed_content_offset,
            show_reasoning=request.show_reasoning,
            reasoning_committed=state.reasoning_committed,
        )

    async def _apply_rendered(
        self,
        request: AssistantTurnRequest,
        state: StreamState,
        sent_msg_ids: list[str],
        rendered: tuple[str, str, str, str],
        current_content: str,
    ) -> None:
        live_text, fallback, header_html, remaining_content = rendered
        if len(live_text) > self._messenger.max_message_length:
            await self._handle_overflow(
                request,
                state,
                sent_msg_ids,
                header_html,
                remaining_content,
                current_content,
            )
        else:
            state.placeholder_msg_id = await self._send_or_edit_placeholder(
                request,
                placeholder_msg_id=state.placeholder_msg_id,
                live_text=live_text,
                fallback=fallback,
            )

    async def _handle_overflow(
        self,
        request: AssistantTurnRequest,
        state: StreamState,
        sent_msg_ids: list[str],
        header_html: str,
        remaining_content: str,
        current_content: str,
    ) -> None:
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

    def _render_live_text(
        self,
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

        max_len = self._messenger.max_message_length
        raw_fallback = remaining_content or current_reasoning
        fallback = raw_fallback[:max_len] + " ▍"
        return live_text, fallback, header_html, remaining_content

    def _render_template_live_text(
        self,
        *,
        template: ResponseTemplate,
        current_content: str,
        state: StreamState,
    ) -> tuple[str, str, str, str] | None:
        """Render live text using template-aware incremental parsing.

        Returns the same 4-tuple as ``_render_live_text`` so the caller
        can handle overflow and placeholder logic identically.
        """
        from mai_gram.core.md_to_telegram import markdown_to_html

        result = template.parse_streaming(current_content)
        content_field = template.content_field_name()
        header_html = self._build_template_header(template, result, content_field, state)

        active_text = ""
        if result.active_field is not None:
            active_text = result.active_content
        elif content_field in result.completed_fields:
            active_text = result.completed_fields[content_field]
        elif result.preamble:
            active_text = result.preamble

        remaining = active_text[state.committed_content_offset :]
        content_html = markdown_to_html(remaining) if remaining.strip() else ""

        return self._assemble_live_text(
            header_html,
            content_html,
            remaining,
            current_content,
            result.preamble,
        )

    @staticmethod
    def _build_template_header(
        template: ResponseTemplate,
        result: StreamingParseResult,
        content_field: str,
        state: StreamState,
    ) -> str:
        parts: list[str] = []
        for descriptor in sorted(template.get_fields(), key=lambda f: f.order):
            name = descriptor.name
            if name == content_field:
                continue
            value = result.completed_fields.get(name, "")
            if not value.strip() or name in state.template_fields_committed:
                continue
            parts.append(template.render_field_html(name, value, expandable=descriptor.expandable))
        return "\n\n".join(parts) if parts else ""

    def _assemble_live_text(
        self,
        header_html: str,
        content_html: str,
        remaining: str,
        current_content: str,
        preamble: str,
    ) -> tuple[str, str, str, str] | None:
        from mai_gram.core.md_to_telegram import markdown_to_html

        max_len = self._messenger.max_message_length
        if header_html and content_html:
            live_text = header_html + "\n\n" + content_html + " ▍"
        elif header_html:
            live_text = header_html + " ▍"
        elif content_html:
            live_text = content_html + " ▍"
        elif preamble.strip():
            live_text = markdown_to_html(preamble) + " ▍"
            return live_text, preamble[:max_len] + " ▍", "", preamble
        else:
            return None
        fallback = (remaining or current_content)[:max_len] + " ▍"
        return live_text, fallback, header_html, remaining

    async def _send_or_edit_placeholder(
        self,
        request: AssistantTurnRequest,
        *,
        placeholder_msg_id: str | None,
        live_text: str,
        fallback: str,
    ) -> str | None:
        if placeholder_msg_id is None:
            return await self._send_new(request, live_text, fallback)
        return await self._edit_existing(request, placeholder_msg_id, live_text, fallback)

    async def _send_new(
        self,
        request: AssistantTurnRequest,
        live_text: str,
        fallback: str,
    ) -> str | None:
        result = await self._messenger.send_message(
            OutgoingMessage(text=live_text, chat_id=request.telegram_chat_id, parse_mode="html")
        )
        if not result.success:
            result = await self._messenger.send_message(
                OutgoingMessage(text=fallback, chat_id=request.telegram_chat_id)
            )
        return result.message_id if result.success else None

    async def _edit_existing(
        self,
        request: AssistantTurnRequest,
        placeholder_msg_id: str,
        live_text: str,
        fallback: str,
    ) -> str | None:
        edit_result = await self._messenger.edit_message(
            request.telegram_chat_id,
            placeholder_msg_id,
            live_text,
            parse_mode="html",
        )
        if not edit_result.success:
            fb = await self._messenger.edit_message(
                request.telegram_chat_id,
                placeholder_msg_id,
                fallback,
            )
            if not fb.success:
                return None
        return placeholder_msg_id
