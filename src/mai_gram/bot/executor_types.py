"""Data types and protocols for the conversation executor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mai_gram.db.models import Chat
    from mai_gram.llm.provider import ChatMessage
    from mai_gram.mcp_servers.manager import MCPManager
    from mai_gram.memory.messages import MessageStore
    from mai_gram.response_templates.base import ParsedResponse, ResponseTemplate


def parse_template_params(chat: Any) -> dict[str, Any] | None:
    """Extract JSON template_params from a Chat object, returning None on failure."""
    raw = getattr(chat, "template_params", None)
    if not raw:
        return None
    try:
        data: dict[str, Any] = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data


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
    resolved_model: str | None = None

    @property
    def model_for_api(self) -> str:
        """Return the OpenRouter model identifier to use for the API call."""
        return self.resolved_model or self.chat.llm_model


@dataclass(frozen=True, slots=True)
class AssistantTurnResult:
    """Result of running one assistant response generation pass."""

    sent_message_ids: list[str]


@dataclass(slots=True)
class StreamState:
    content_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    placeholder_msg_id: str | None = None
    last_edit_time: float = 0.0
    last_display_len: int = 0
    committed_content_offset: int = 0
    reasoning_committed: bool = False
    template_fields_committed: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class StreamOutcome:
    response_text: str
    response_reasoning: str | None
    placeholder_msg_id: str | None
    committed_content_offset: int
    reasoning_committed: bool
    usage: object | None
    cost: float | None
    is_byok: bool
    finish_reason: str | None = None


class ConversationRenderer(Protocol):
    def _build_intermediate_display(
        self,
        content: str,
        reasoning: str,
        show_reasoning: bool,
    ) -> str: ...

    def _format_usage_footer(
        self,
        usage: object,
        cost: float | None,
        is_byok: bool,
    ) -> str: ...

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


def replace_response_text(outcome: StreamOutcome, new_text: str) -> StreamOutcome:
    """Return a copy of *outcome* with response_text replaced."""
    return StreamOutcome(
        response_text=new_text,
        response_reasoning=outcome.response_reasoning,
        placeholder_msg_id=outcome.placeholder_msg_id,
        committed_content_offset=outcome.committed_content_offset,
        reasoning_committed=outcome.reasoning_committed,
        usage=outcome.usage,
        cost=outcome.cost,
        is_byok=outcome.is_byok,
        finish_reason=outcome.finish_reason,
    )


def parse_hidden_fields(hidden_json: str | None) -> set[str]:
    """Parse the JSON-encoded hidden fields list from the chat model."""
    if not hidden_json:
        return set()
    try:
        data = json.loads(hidden_json)
        if isinstance(data, list):
            return {str(item) for item in data}
    except (json.JSONDecodeError, TypeError):
        pass
    return set()


def build_structured_parts(
    template: ResponseTemplate,
    parsed: ParsedResponse,
    hidden: set[str],
) -> tuple[str, str]:
    """Build header HTML and content text from structured template fields."""
    content_field = template.content_field_name()
    fields_ordered = sorted(template.get_fields(), key=lambda f: f.order)
    header_parts: list[str] = []
    content_text = ""

    for descriptor in fields_ordered:
        value = parsed.fields.get(descriptor.name, "")
        if not value.strip() or descriptor.name in hidden:
            continue
        if descriptor.name == content_field:
            content_text = value
            continue
        html = template.render_field_html(
            descriptor.name,
            value,
            expandable=descriptor.expandable,
        )
        header_parts.append(html)

    header_html = "\n\n".join(header_parts) if header_parts else ""
    return header_html, content_text
