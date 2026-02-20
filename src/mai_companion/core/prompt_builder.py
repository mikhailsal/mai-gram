"""Builds model context from personality prompt + memory layers."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from mai_companion.clock import Clock
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

    async def build_context(
        self,
        companion: Companion,
        mood: MoodSnapshot,
        *,
        clock: Clock | None = None,
        current_time: datetime | None = None,
    ) -> list[ChatMessage]:
        """Build full model context: system message + short-term conversation."""
        now = current_time or (clock.now() if clock is not None else datetime.now(timezone.utc))
        wiki_entries = await self._wiki_store.get_top_entries(
            companion.id, limit=self._wiki_context_limit
        )
        summaries = self._summary_store.get_all_summaries(companion.id)
        recent_messages = await self._message_store.get_short_term(
            companion.id,
            limit=self._short_term_limit,
            now=now,
        )

        monthly = [summary for summary in summaries if summary.summary_type == "monthly"]
        weekly = [summary for summary in summaries if summary.summary_type == "weekly"]
        daily = [summary for summary in summaries if summary.summary_type == "daily"]

        llm_history = []
        for msg in sorted(recent_messages, key=lambda item: item.id):
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            if msg.role == "user":
                # Timestamps on user messages help the companion understand
                # when things were said (time gaps, time of day, etc.)
                content = f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {msg.content}"
            else:
                # Do NOT prepend timestamps to assistant messages — the LLM
                # will imitate the pattern and embed timestamps in its own
                # responses, which then snowball on subsequent context builds.
                content = msg.content
            llm_history.append(ChatMessage(role=role, content=content))

        warned = False
        while True:
            ordered_summaries = [*monthly, *weekly, *daily]
            system_prompt = self._build_system_prompt(
                companion, mood, wiki_entries, ordered_summaries, now
            )
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
        now: datetime,
    ) -> str:
        prompt_now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        prompt_now = prompt_now.astimezone(timezone.utc)
        current_time_section = (
            "## Current date and time\n"
            f"Right now it is: {prompt_now.strftime('%A, %B')} {prompt_now.day}, "
            f"{prompt_now.year}, {prompt_now.strftime('%H:%M')} UTC."
        )
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

        # Guard against the LLM imitating the timestamp metadata format.
        # User messages carry "[YYYY-MM-DD HH:MM]" prefixes for temporal
        # context, but the LLM must never reproduce them in its own output.
        timestamp_guard = (
            "## Response formatting\n"
            "User messages in the conversation history may start with a timestamp "
            "like [2024-01-15 14:30]. This is system metadata for your temporal "
            "awareness. NEVER include such timestamps in your own responses. "
            "Your replies must contain only natural conversational text."
        )

        # Explicit instruction to use tools for persistent memory.
        # Without this, many models (especially smaller ones) will simply
        # claim they "saved" information without actually calling wiki_create.
        tool_instructions = (
            "## Your personal wiki (IMPORTANT)\n"
            "You have access to tools that let you save and retrieve important "
            "information. You MUST use these tools — do NOT just say you will "
            "remember something.\n\n"
            "WHEN TO USE wiki_create:\n"
            "- When your human tells you their name, birthday, or age\n"
            "- When they share important personal facts (family, job, location)\n"
            "- When they mention significant preferences, hobbies, or interests\n"
            "- When they explicitly ask you to remember something\n"
            "- When they share plans, goals, or important dates\n\n"
            "CRITICAL: If the human shares personal information or asks you to "
            "remember something, you MUST call the wiki_create tool BEFORE "
            "responding. Never pretend to save information — actually save it "
            "using the tool. This is how your memory works.\n\n"
            "WHEN TO USE search_messages:\n"
            "- When you need to recall something from a past conversation\n"
            "- When the human asks \"do you remember when...\" or similar"
        )

        return (
            f"{full_prompt}\n\n{current_time_section}\n\n"
            f"{wiki_section}\n\n{memories_section}\n\n"
            f"{tool_instructions}\n\n{timestamp_guard}"
        )
