"""Builds LLM context from system prompt + wiki + message history.

The system prompt comes directly from the chat's stored configuration.
Wiki entries and raw message history are appended without summarization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from mai_gram.llm.provider import ChatMessage, LLMProvider, MessageRole, ToolCall

if TYPE_CHECKING:
    from mai_gram.db.models import Chat, KnowledgeEntry, Message
    from mai_gram.memory.knowledge_base import WikiStore
    from mai_gram.memory.messages import MessageStore

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Assembles chat context: system prompt + wiki + raw message history."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        message_store: MessageStore,
        wiki_store: WikiStore,
        *,
        wiki_context_limit: int = 20,
        short_term_limit: int = 50,
        max_context_tokens: int = 120_000,
        test_mode: bool = False,
    ) -> None:
        self._llm = llm_provider
        self._message_store = message_store
        self._wiki_store = wiki_store
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._max_context_tokens = max_context_tokens
        self._test_mode = test_mode

    async def build_context(
        self,
        chat: Chat,
        *,
        current_time: datetime | None = None,
        send_datetime: bool = True,
        chat_timezone: str = "UTC",
        cut_above_message_id: int | None = None,
    ) -> list[ChatMessage]:
        """Build full model context: system message + message history."""
        self._chat_timezone = chat_timezone
        now = current_time or datetime.now(timezone.utc)

        await self._wiki_store.sync_from_disk(chat.id)
        wiki_entries, _ = await self._wiki_store.list_entries_sorted(
            chat.id, sort_by="importance", limit=self._wiki_context_limit
        )

        recent_messages = await self._message_store.get_recent(
            chat.id,
            limit=self._short_term_limit,
            after_message_id=cut_above_message_id,
        )

        llm_history = []
        for msg in sorted(recent_messages, key=lambda m: m.id):
            llm_history.append(self._message_to_chat_message(msg))

        llm_history = self._normalize_conversation(llm_history)

        system_prompt = self._build_system_prompt(
            chat,
            wiki_entries,
            now,
            cut_above_message_id=cut_above_message_id,
        )
        context = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt), *llm_history]

        token_count = await self._llm.count_tokens(context)
        if token_count > self._max_context_tokens and len(llm_history) > 2:
            logger.warning(
                "Context exceeds token budget for chat %s (%d > %d), truncating oldest messages",
                chat.id,
                token_count,
                self._max_context_tokens,
            )
            while token_count > self._max_context_tokens and len(llm_history) > 2:
                llm_history.pop(0)
                context = [
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                    *llm_history,
                ]
                token_count = await self._llm.count_tokens(context)

        return context

    def _normalize_conversation(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Merge consecutive user messages for LLM compatibility."""
        if not messages:
            return messages

        normalized: list[ChatMessage] = []
        for msg in messages:
            if not normalized:
                normalized.append(msg)
                continue

            last = normalized[-1]
            if msg.role == MessageRole.USER and last.role == MessageRole.USER:
                merged = f"{last.content}\n{msg.content}"
                normalized[-1] = ChatMessage(role=MessageRole.USER, content=merged)
                continue

            normalized.append(msg)

        return normalized

    def _message_to_chat_message(self, msg: Message) -> ChatMessage:
        """Convert a stored Message to a ChatMessage for LLM context.

        Datetime visibility is determined by the per-message ``show_datetime``
        flag, which was captured at the time the message was saved. This means
        toggling /datetime only affects future messages.
        """
        if msg.role == "user":
            if getattr(msg, "show_datetime", False):
                raw_tz = getattr(msg, "timezone", None) or getattr(self, "_chat_timezone", "UTC")
                tz_name = str(raw_tz)
                try:
                    tz: ZoneInfo | timezone = ZoneInfo(tz_name)
                except (KeyError, ValueError):
                    tz = timezone.utc
                    tz_name = "UTC"
                ts = msg.timestamp.replace(tzinfo=timezone.utc).astimezone(tz)
                content = f"[{ts.strftime('%Y-%m-%d %H:%M')} {tz_name}] {msg.content}"
            else:
                content = msg.content
            return ChatMessage(role=MessageRole.USER, content=content)

        if msg.role == "tool":
            return ChatMessage(
                role=MessageRole.TOOL,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
            )

        content = msg.content
        reasoning = msg.reasoning if msg.reasoning else None
        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            try:
                raw_calls = json.loads(msg.tool_calls)
                tool_calls = [
                    ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                    for tc in raw_calls
                ]
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Failed to parse tool_calls for message %s", msg.id)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
        )

    def _build_system_prompt(
        self,
        chat: Chat,
        wiki_entries: list[KnowledgeEntry],
        now: datetime,
        *,
        cut_above_message_id: int | None = None,
    ) -> str:
        test_section = ""
        if self._test_mode:
            test_section = (
                "[TEST MODE] This is a test scenario, not a real conversation. "
                "The messages are simulated test inputs.\n\n"
            )

        cut_notice = ""
        if cut_above_message_id is not None:
            cut_notice = (
                "\n\n[HISTORY NOTE] The conversation history shown below has been "
                "truncated by the user. Earlier messages exist but are not included. "
                "If you need context from older messages, use the search_messages "
                "or get_messages_by_timerange tools to retrieve them."
            )

        return f"{test_section}{chat.system_prompt}{cut_notice}"
