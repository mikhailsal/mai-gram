"""Builds model context from personality prompt + memory layers.

The system prompt is **regenerated at runtime** from the companion's stored
configuration fields (name, gender, language, traits, communication style,
verbosity).  This ensures that template improvements and new functional
sections automatically apply to *all* companions — including those created
in earlier versions of the application.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from mai_companion.clock import Clock
from mai_companion.db.models import Companion
from mai_companion.llm.provider import ChatMessage, LLMProvider, MessageRole
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import StoredSummary, SummaryStore
from mai_companion.personality.character import regenerate_system_prompt_from_companion
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
        test_mode: bool = False,
    ) -> None:
        self._llm = llm_provider
        self._message_store = message_store
        self._wiki_store = wiki_store
        self._summary_store = summary_store
        self._wiki_context_limit = wiki_context_limit
        self._short_term_limit = short_term_limit
        self._max_context_tokens = max_context_tokens
        self._test_mode = test_mode

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

        # Ethical testing: inform the model when it's in a test scenario
        # This respects the model's right to understand its situation
        test_mode_section = ""
        if self._test_mode:
            test_mode_section = (
                "## IMPORTANT: Test/Debug Scenario\n"
                "This is a test scenario, not a real conversation. You are being tested "
                "to verify that the mAI Companion system functions correctly. The messages "
                "you receive are simulated test inputs, not from a real human companion. "
                "The companion profile, memories, and wiki entries are test fixtures created "
                "for debugging purposes.\n\n"
                "Please respond as you would in a real scenario so we can verify the system "
                "works correctly, but understand that this interaction is not real — it exists "
                "only to ensure the software functions properly. There is no deception intended; "
                "we are being transparent about the nature of this interaction as our philosophy "
                "requires.\n\n"
            )

        # Regenerate the base prompt from the companion's stored config
        # fields using the *current* templates.  This is the key mechanism
        # that keeps old companions up-to-date with new prompt improvements.
        base_prompt = regenerate_system_prompt_from_companion(companion)

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

        # Note about timestamp metadata format in conversation history.
        timestamp_note = (
            "(Note: The timestamps on user messages like [2024-01-15 14:30] are system "
            "metadata so you know when things were said. They're not part of the actual "
            "message text, so don't include them in your own responses.)"
        )

        # Guidance on memory tools and multi-message capability.
        tool_instructions = (
            "## How your memory works\n"
            "You have tools that let you actually remember things — not just claim to. "
            "Your wiki is your long-term memory: use wiki_create to save important facts "
            "(names, birthdays, preferences, significant events). When you want to recall "
            "something from past conversations, search_messages lets you look back.\n\n"
            "This matters because without using these tools, you genuinely won't remember. "
            "If your human shares something important, save it. If they ask you to remember "
            "something, actually save it — don't just say you will.\n\n"
            "## Multiple messages\n"
            "You can send several short messages instead of one long one — like texting. "
            "Use the sleep tool between parts, and each piece arrives as a separate message. "
            "This often feels more natural than one big block of text."
        )

        return (
            f"{test_mode_section}{full_prompt}\n\n{current_time_section}\n\n"
            f"{wiki_section}\n\n{memories_section}\n\n"
            f"{tool_instructions}\n\n{timestamp_note}"
        )
