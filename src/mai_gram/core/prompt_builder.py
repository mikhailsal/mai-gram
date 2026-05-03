"""Builds LLM context from system prompt + wiki + message history.

The system prompt comes directly from the chat's stored configuration.
Wiki entries and raw message history are appended without summarization.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from mai_gram.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_gram.memory.messages import decode_persisted_message

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
        short_term_limit: int = 500,
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

        wiki_entries = await self._load_wiki_entries(chat.id)
        llm_history = await self._load_llm_history(chat.id, cut_above_message_id)
        loaded_count = len(llm_history)

        system_prompt = self._build_system_prompt(
            chat,
            wiki_entries,
            now,
            cut_above_message_id=cut_above_message_id,
        )
        llm_history, token_count = await self._truncate_history_for_budget(
            chat.id,
            system_prompt,
            llm_history,
        )
        context = self._build_context_messages(system_prompt, llm_history)

        final_count = len(llm_history)
        logger.info(
            "Context for %s: loaded %d msgs (limit=%d), after truncation %d msgs, ~%d tokens",
            chat.id,
            loaded_count,
            self._short_term_limit,
            final_count,
            token_count,
        )

        return context

    async def _load_wiki_entries(self, chat_id: str) -> list[KnowledgeEntry]:
        wiki_entries, _ = await self._wiki_store.list_entries_sorted(
            chat_id,
            sort_by="importance",
            limit=self._wiki_context_limit,
        )
        return wiki_entries

    async def _load_llm_history(
        self,
        chat_id: str,
        cut_above_message_id: int | None,
    ) -> list[ChatMessage]:
        recent_messages = await self._message_store.get_recent(
            chat_id,
            limit=self._short_term_limit,
            after_message_id=cut_above_message_id,
        )
        llm_history = [
            self._message_to_chat_message(msg)
            for msg in sorted(recent_messages, key=lambda message: message.id)
        ]
        return self._normalize_conversation(llm_history)

    async def _truncate_history_for_budget(
        self,
        chat_id: str,
        system_prompt: str,
        llm_history: list[ChatMessage],
    ) -> tuple[list[ChatMessage], int]:
        context = self._build_context_messages(system_prompt, llm_history)
        token_count = await self._llm.count_tokens(context)
        if token_count <= self._max_context_tokens or len(llm_history) <= 2:
            return llm_history, token_count

        logger.warning(
            "Context exceeds token budget for chat %s (%d > %d), truncating oldest messages",
            chat_id,
            token_count,
            self._max_context_tokens,
        )
        while token_count > self._max_context_tokens and len(llm_history) > 2:
            llm_history.pop(0)
            context = self._build_context_messages(system_prompt, llm_history)
            token_count = await self._llm.count_tokens(context)
        return llm_history, token_count

    @staticmethod
    def _build_context_messages(
        system_prompt: str,
        llm_history: list[ChatMessage],
    ) -> list[ChatMessage]:
        return [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt), *llm_history]

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
        persisted = decode_persisted_message(msg)

        if persisted.role == MessageRole.USER:
            if getattr(msg, "show_datetime", False):
                raw_tz = getattr(msg, "timezone", None) or getattr(self, "_chat_timezone", "UTC")
                tz_name = str(raw_tz)
                try:
                    tz: ZoneInfo | timezone = ZoneInfo(tz_name)
                except (KeyError, ValueError):
                    tz = timezone.utc
                    tz_name = "UTC"
                ts = msg.timestamp.replace(tzinfo=timezone.utc).astimezone(tz)
                content = f"[{ts.strftime('%Y-%m-%d %H:%M')} {tz_name}] {persisted.content}"
            else:
                content = persisted.content
            return ChatMessage(role=MessageRole.USER, content=content)

        if persisted.role == MessageRole.TOOL:
            return ChatMessage(
                role=MessageRole.TOOL,
                content=persisted.content,
                tool_call_id=persisted.tool_call_id,
            )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=persisted.content,
            tool_calls=persisted.tool_calls,
            reasoning=persisted.reasoning,
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

        template_section = self._build_template_section(chat)

        return f"{test_section}{chat.system_prompt}{cut_notice}{template_section}"

    @staticmethod
    def _build_template_section(chat: Chat) -> str:
        """Append response format instructions from the chat's template."""
        import json as _json

        from mai_gram.response_templates.registry import get_template

        raw_params = getattr(chat, "template_params", None)
        params: dict[str, object] | None = None
        if raw_params:
            try:
                params = _json.loads(raw_params)
            except (ValueError, TypeError):
                params = None

        template = get_template(getattr(chat, "response_template", None), params)
        instruction = template.format_instruction()
        if not instruction:
            return ""

        examples = template.examples()
        if not examples:
            return instruction

        parts = [instruction, "\n\nExamples:"]
        for ex in examples:
            label = "CORRECT" if ex.is_positive else "INCORRECT"
            parts.append(f"\n[{label}]\n{ex.text}")
        return "".join(parts)
