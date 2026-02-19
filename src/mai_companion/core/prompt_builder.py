"""Builds model context from personality prompt + memory layers."""

from __future__ import annotations

import logging

from mai_companion.db.models import Companion
from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import StoredSummary, SummaryStore
from mai_companion.personality.mood import MoodSnapshot, mood_to_prompt_section

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Assembles chat context with memory-aware token budgeting."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        message_store: MessageStore,
        wiki_store: WikiStore,
        summary_store: SummaryStore,
        *,
        wiki_context_limit: int = 20,
        short_term_limit: int = 30,
        max_context_tokens: int = 120000,
    ) -> None:
        self._llm = llm_provider
        self._message_store = message_store
        self._wiki_store = wiki_store
        self._summary_store = summary_store
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._max_context_tokens = max_context_tokens

    async def build_context(self, companion: Companion, mood: MoodSnapshot) -> list[ChatMessage]:
        """Build full model context: system message + short-term conversation."""
        wiki_entries = await self._wiki_store.get_top_entries(
            companion.id, limit=self._wiki_context_limit
        )
        summaries = self._summary_store.get_all_summaries(companion.id)
        recent_messages = await self._message_store.get_short_term(
            companion.id, limit=self._short_term_limit
        )

        monthly = [summary for summary in summaries if summary.summary_type == "monthly"]
        weekly = [summary for summary in summaries if summary.summary_type == "weekly"]
        daily = [summary for summary in summaries if summary.summary_type == "daily"]

        llm_history = [
            ChatMessage(
                role=MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT,
                content=msg.content,
            )
            for msg in sorted(recent_messages, key=lambda item: item.id)
        ]

        warned = False
        while True:
            ordered_summaries = [*monthly, *weekly, *daily]
            system_prompt = self._build_system_prompt(companion, mood, wiki_entries, ordered_summaries)
            context = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt), *llm_history]
            token_count = await self._llm.count_tokens(context)

            if token_count > int(self._max_context_tokens * 0.9):
                logger.warning(
                    "Prompt token usage near budget for companion %s: %s/%s",
                    companion.id,
                    token_count,
                    self._max_context_tokens,
                )

            if token_count <= self._max_context_tokens:
                return context

            if not warned:
                logger.warning(
                    "Prompt exceeds token budget for companion %s: truncating summaries.",
                    companion.id,
                )
                warned = True

            if monthly:
                monthly.pop(0)
                continue
            if weekly:
                weekly.pop(0)
                continue
            if daily:
                daily.pop(0)
                continue
            return context

    def _build_system_prompt(
        self,
        companion: Companion,
        mood: MoodSnapshot,
        wiki_entries: list,
        summaries: list[StoredSummary],
    ) -> str:
        base_prompt = companion.system_prompt
        mood_section = mood_to_prompt_section(mood)
        relationship_section = (
            "## Relationship stage\n"
            f"You are in the '{companion.relationship_stage}' stage of your relationship. "
            "This is still early -- you're getting to know each other."
        )
        full_prompt = (
            base_prompt.replace("{mood_section}", mood_section).replace(
                "{relationship_section}", relationship_section
            )
        )

        wiki_lines = []
        for entry in wiki_entries:
            wiki_lines.append(f"- ({int(entry.importance)}) {entry.key}: {entry.value}")
        wiki_section = (
            "## Things you know\n"
            + ("\n".join(wiki_lines) if wiki_lines else "- No persistent knowledge yet.")
        )

        summary_lines = []
        for item in summaries:
            summary_lines.append(f"- [{item.summary_type}:{item.period}] {item.content.strip()}")
        memories_section = (
            "## Your memories\n"
            + ("\n".join(summary_lines) if summary_lines else "- No memory summaries yet.")
        )

        return f"{full_prompt}\n\n{wiki_section}\n\n{memories_section}"
