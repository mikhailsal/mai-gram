"""Builds LLM context from system prompt + wiki + message history.

The system prompt comes directly from the chat's stored configuration.
Wiki entries and raw message history are appended without summarization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from mai_gram.db.models import Chat, Message
from mai_gram.llm.provider import ChatMessage, LLMProvider, MessageRole, ToolCall
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
    ) -> list[ChatMessage]:
        """Build full model context: system message + message history."""
        now = current_time or datetime.now(timezone.utc)

        wiki_entries = await self._wiki_store.get_top_entries(
            chat.id, limit=self._wiki_context_limit
        )

        recent_messages = await self._message_store.get_recent(
            chat.id,
            limit=self._short_term_limit,
        )

        llm_history = []
        for msg in sorted(recent_messages, key=lambda m: m.id):
            llm_history.append(self._message_to_chat_message(msg))

        llm_history = self._normalize_conversation(llm_history)

        system_prompt = self._build_system_prompt(chat, wiki_entries, now)
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

    def _normalize_conversation(
        self, messages: list[ChatMessage]
    ) -> list[ChatMessage]:
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
        """Convert a stored Message to a ChatMessage for LLM context."""
        if msg.role == "user":
            content = f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {msg.content}"
            return ChatMessage(role=MessageRole.USER, content=content)

        if msg.role == "tool":
            return ChatMessage(
                role=MessageRole.TOOL,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
            )

        content = msg.content
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

        return ChatMessage(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    def _build_system_prompt(
        self,
        chat: Chat,
        wiki_entries: list,
        now: datetime,
    ) -> str:
        prompt_now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        prompt_now = prompt_now.astimezone(timezone.utc)

        time_section = (
            f"Current date and time: "
            f"{prompt_now.strftime('%A, %B')} {prompt_now.day}, "
            f"{prompt_now.year}, {prompt_now.strftime('%H:%M')} UTC."
        )

        test_section = ""
        if self._test_mode:
            test_section = (
                "[TEST MODE] This is a test scenario, not a real conversation. "
                "The messages are simulated test inputs.\n\n"
            )

        wiki_lines = []
        for entry in wiki_entries:
            wiki_lines.append(f"- ({int(entry.importance)}) {entry.key}: {entry.value}")
        wiki_section = (
            "Things you know:\n"
            + ("\n".join(wiki_lines) if wiki_lines else "- No saved knowledge yet.")
        )

        tool_instructions = (
            "You have tools to save and recall information. "
            "Use wiki_create to remember important facts. "
            "Use search_messages to find past conversations."
        )

        return (
            f"{test_section}{chat.system_prompt}\n\n"
            f"{time_section}\n\n"
            f"{wiki_section}\n\n"
            f"{tool_instructions}"
        )
