"""Tests for MessageStore."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import Companion
from mai_companion.memory.messages import MessageStore


async def _create_companion(session: AsyncSession, companion_id: str = "comp-1") -> str:
    companion = Companion(id=companion_id, name="Test Companion")
    session.add(companion)
    await session.flush()
    return companion_id


class TestMessageStore:
    """MessageStore behavior."""

    async def test_save_and_retrieve_message(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        await store.save_message(companion_id, "user", "Hello memory")
        messages = await store.get_short_term(companion_id)

        assert len(messages) == 1
        assert messages[0].content == "Hello memory"
        assert messages[0].role == "user"
        assert isinstance(messages[0].timestamp, datetime)

    async def test_short_term_returns_only_today(self, session: AsyncSession) -> None:
        """get_short_term only returns messages from today.

        Past days should have daily summaries (via backfill mechanism).
        Raw messages from past days are not included to avoid duplication.
        """
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        today = datetime(2026, 2, 14, 9, 0, 0)
        yesterday = today - timedelta(days=1)

        # Create messages from yesterday
        for i in range(5):
            await store.save_message(
                companion_id, "assistant", f"y-{i}", timestamp=yesterday + timedelta(minutes=i)
            )
        # Create messages from today
        for i in range(35):
            await store.save_message(
                companion_id, "user", f"t-{i}", timestamp=today + timedelta(minutes=i)
            )

        messages = await store.get_short_term(companion_id, now=today)

        # Only today's messages should be returned
        assert len(messages) == 35
        assert all(m.content.startswith("t-") for m in messages)

    async def test_short_term_all_today_messages(self, session: AsyncSession) -> None:
        """All of today's messages are included with no limit.

        Even if there are many messages, all are returned since the AI
        needs full visibility of the current conversation.
        """
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        now = datetime(2026, 2, 14, 9, 0, 0)

        # Create many messages for today
        for i in range(100):
            await store.save_message(
                companion_id,
                "user",
                f"msg-{i}",
                timestamp=now + timedelta(minutes=i),
            )

        messages = await store.get_short_term(companion_id, now=now)

        # All 100 messages should be returned (no limit)
        assert len(messages) == 100

    async def test_short_term_deduplication(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        now = datetime(2026, 2, 14, 9, 0, 0)

        for i in range(35):
            await store.save_message(
                companion_id,
                "user",
                f"today-{i}",
                timestamp=now + timedelta(minutes=i),
            )

        messages = await store.get_short_term(companion_id, limit=30, now=now)
        ids = [message.id for message in messages]
        assert len(ids) == 35
        assert len(set(ids)) == 35

    async def test_search_basic(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "I love Paris")
        await store.save_message(companion_id, "assistant", "Tokyo is amazing")

        results = await store.search(companion_id, "Paris")
        assert len(results) == 1
        assert results[0].content == "I love Paris"

    async def test_search_case_insensitive(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "I went to Paris")

        results = await store.search(companion_id, "paris")
        assert len(results) == 1
        assert results[0].content == "I went to Paris"

    async def test_search_cjk(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "今日は天気がいい")

        results = await store.search(companion_id, "天気")
        assert len(results) == 1
        assert results[0].content == "今日は天気がいい"

    async def test_search_cyrillic(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "Привет мир")

        results = await store.search(companion_id, "мир")
        assert len(results) == 1
        assert results[0].content == "Привет мир"

    async def test_search_limit(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        for i in range(30):
            await store.save_message(companion_id, "assistant", f"keyword-{i}")

        results = await store.search(companion_id, "keyword", limit=10)
        assert len(results) == 10

    async def test_search_order_newest_first(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "keyword-old")
        await store.save_message(companion_id, "user", "keyword-new")

        results = await store.search(companion_id, "keyword")
        assert results[0].content == "keyword-new"
        assert results[1].content == "keyword-old"

    async def test_get_messages_for_date(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        await store.save_message(companion_id, "user", "day1", timestamp=datetime(2026, 2, 1, 10, 0))
        await store.save_message(companion_id, "user", "day2", timestamp=datetime(2026, 2, 2, 10, 0))

        results = await store.get_messages_for_date(companion_id, date(2026, 2, 1))
        assert [message.content for message in results] == ["day1"]

    async def test_get_messages_in_range(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        for i in range(5):
            await store.save_message(
                companion_id,
                "user",
                f"d{i}",
                timestamp=datetime(2026, 2, 10 + i, 10, 0),
            )

        results = await store.get_messages_in_range(
            companion_id, start_date=date(2026, 2, 11), end_date=date(2026, 2, 13)
        )
        assert [message.content for message in results] == ["d1", "d2", "d3"]

    async def test_search_escapes_like_wildcards(self, session: AsyncSession) -> None:
        """Verify that SQL LIKE wildcards (%, _) in queries are treated as literals."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        # Messages with literal % and _ characters
        await store.save_message(companion_id, "user", "I got 100% on the test")
        await store.save_message(companion_id, "user", "The file is named data_file.txt")
        # Messages that would match if wildcards weren't escaped
        await store.save_message(companion_id, "user", "I got 1000 points")
        await store.save_message(companion_id, "user", "The file is named dataXfile.txt")

        # Search for literal "100%" - should only match the first message
        results = await store.search(companion_id, "100%")
        assert len(results) == 1
        assert results[0].content == "I got 100% on the test"

        # Search for literal "data_file" - should only match the second message
        results = await store.search(companion_id, "data_file")
        assert len(results) == 1
        assert results[0].content == "The file is named data_file.txt"

    async def test_save_message_rejects_non_monotonic_timestamp(
        self, session: AsyncSession
    ) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        timestamp = datetime(2026, 2, 1, 10, 0, 0)
        await store.save_message(companion_id, "user", "first", timestamp=timestamp)

        with pytest.raises(ValueError, match="chronologically ordered"):
            await store.save_message(companion_id, "assistant", "second", timestamp=timestamp)

    async def test_save_message_allows_strictly_newer_timestamp(
        self, session: AsyncSession
    ) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)
        await store.save_message(companion_id, "user", "first", timestamp=base)
        newer = await store.save_message(
            companion_id, "assistant", "second", timestamp=base + timedelta(seconds=1)
        )

        assert newer.content == "second"

    async def test_search_oldest_first(self, session: AsyncSession) -> None:
        """Verify oldest_first parameter returns messages in chronological order."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)
        await store.save_message(
            companion_id, "user", "keyword-first", timestamp=base
        )
        await store.save_message(
            companion_id, "user", "keyword-second", timestamp=base + timedelta(hours=1)
        )
        await store.save_message(
            companion_id, "user", "keyword-third", timestamp=base + timedelta(hours=2)
        )

        # Default (newest first)
        results_newest = await store.search(companion_id, "keyword")
        assert results_newest[0].content == "keyword-third"
        assert results_newest[-1].content == "keyword-first"

        # Oldest first
        results_oldest = await store.search(companion_id, "keyword", oldest_first=True)
        assert results_oldest[0].content == "keyword-first"
        assert results_oldest[-1].content == "keyword-third"

    async def test_get_message_by_id(self, session: AsyncSession) -> None:
        """Verify get_message_by_id returns correct message or None."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        msg = await store.save_message(companion_id, "user", "find me")

        found = await store.get_message_by_id(companion_id, msg.id)
        assert found is not None
        assert found.content == "find me"

        not_found = await store.get_message_by_id(companion_id, 99999)
        assert not_found is None

    async def test_get_message_by_id_wrong_companion(self, session: AsyncSession) -> None:
        """Verify get_message_by_id doesn't return messages from other companions."""
        companion_id = await _create_companion(session, "comp-a")
        other_id = await _create_companion(session, "comp-b")
        store = MessageStore(session)

        msg = await store.save_message(companion_id, "user", "comp-a message")

        # Should not find message when querying with wrong companion_id
        not_found = await store.get_message_by_id(other_id, msg.id)
        assert not_found is None

    async def test_get_message_context_basic(self, session: AsyncSession) -> None:
        """Verify get_message_context returns surrounding messages."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        # Create 10 messages
        messages = []
        for i in range(10):
            msg = await store.save_message(
                companion_id,
                "user" if i % 2 == 0 else "assistant",
                f"msg-{i}",
                timestamp=base + timedelta(minutes=i),
            )
            messages.append(msg)

        # Get context for message 5 (middle)
        before, target, after = await store.get_message_context(
            companion_id, messages[5].id, before=3, after=3
        )

        assert target is not None
        assert target.content == "msg-5"
        assert len(before) == 3
        assert [m.content for m in before] == ["msg-2", "msg-3", "msg-4"]
        assert len(after) == 3
        assert [m.content for m in after] == ["msg-6", "msg-7", "msg-8"]

    async def test_get_message_context_at_start(self, session: AsyncSession) -> None:
        """Verify get_message_context handles messages at conversation start."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        messages = []
        for i in range(5):
            msg = await store.save_message(
                companion_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )
            messages.append(msg)

        # Get context for first message
        before, target, after = await store.get_message_context(
            companion_id, messages[0].id, before=5, after=3
        )

        assert target is not None
        assert target.content == "msg-0"
        assert len(before) == 0  # No messages before
        assert len(after) == 3
        assert [m.content for m in after] == ["msg-1", "msg-2", "msg-3"]

    async def test_get_message_context_at_end(self, session: AsyncSession) -> None:
        """Verify get_message_context handles messages at conversation end."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        messages = []
        for i in range(5):
            msg = await store.save_message(
                companion_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )
            messages.append(msg)

        # Get context for last message
        before, target, after = await store.get_message_context(
            companion_id, messages[4].id, before=3, after=5
        )

        assert target is not None
        assert target.content == "msg-4"
        assert len(before) == 3
        assert [m.content for m in before] == ["msg-1", "msg-2", "msg-3"]
        assert len(after) == 0  # No messages after

    async def test_get_message_context_not_found(self, session: AsyncSession) -> None:
        """Verify get_message_context returns empty when message not found."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        before, target, after = await store.get_message_context(
            companion_id, 99999, before=5, after=5
        )

        assert target is None
        assert before == []
        assert after == []

    async def test_get_messages_paginated_basic(self, session: AsyncSession) -> None:
        """Verify get_messages_paginated returns correct page of messages."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        for i in range(25):
            await store.save_message(
                companion_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        # First page
        messages, total = await store.get_messages_paginated(
            companion_id, limit=10, offset=0
        )
        assert total == 25
        assert len(messages) == 10
        assert messages[0].content == "msg-0"  # oldest first by default
        assert messages[9].content == "msg-9"

        # Second page
        messages, total = await store.get_messages_paginated(
            companion_id, limit=10, offset=10
        )
        assert total == 25
        assert len(messages) == 10
        assert messages[0].content == "msg-10"

        # Last page (partial)
        messages, total = await store.get_messages_paginated(
            companion_id, limit=10, offset=20
        )
        assert total == 25
        assert len(messages) == 5
        assert messages[0].content == "msg-20"

    async def test_get_messages_paginated_newest_first(self, session: AsyncSession) -> None:
        """Verify get_messages_paginated respects oldest_first=False."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        for i in range(10):
            await store.save_message(
                companion_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        messages, _ = await store.get_messages_paginated(
            companion_id, limit=5, oldest_first=False
        )
        assert messages[0].content == "msg-9"  # newest first
        assert messages[4].content == "msg-5"

    async def test_get_messages_paginated_with_date_filter(
        self, session: AsyncSession
    ) -> None:
        """Verify get_messages_paginated filters by date range."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        # Messages across multiple days
        await store.save_message(
            companion_id, "user", "day1", timestamp=datetime(2026, 2, 1, 10, 0)
        )
        await store.save_message(
            companion_id, "user", "day2", timestamp=datetime(2026, 2, 2, 10, 0)
        )
        await store.save_message(
            companion_id, "user", "day3", timestamp=datetime(2026, 2, 3, 10, 0)
        )
        await store.save_message(
            companion_id, "user", "day4", timestamp=datetime(2026, 2, 4, 10, 0)
        )

        # Filter to days 2-3
        messages, total = await store.get_messages_paginated(
            companion_id,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 3),
        )
        assert total == 2
        assert [m.content for m in messages] == ["day2", "day3"]

    async def test_get_messages_paginated_limit_clamped(
        self, session: AsyncSession
    ) -> None:
        """Verify limit is clamped to max 50."""
        companion_id = await _create_companion(session)
        store = MessageStore(session)
        base = datetime(2026, 2, 1, 10, 0, 0)

        for i in range(60):
            await store.save_message(
                companion_id, "user", f"msg-{i}", timestamp=base + timedelta(minutes=i)
            )

        messages, _ = await store.get_messages_paginated(companion_id, limit=100)
        assert len(messages) == 50  # Clamped to max
