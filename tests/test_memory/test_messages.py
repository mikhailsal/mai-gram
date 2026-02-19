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

    async def test_short_term_limit(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        base = datetime(2026, 1, 1, 12, 0, 0)
        for i in range(40):
            await store.save_message(
                companion_id,
                "user",
                f"msg-{i}",
                timestamp=base + timedelta(minutes=i),
            )

        messages = await store.get_short_term(companion_id, limit=30, now=datetime(2026, 1, 2, 8, 0))
        assert len(messages) == 30
        assert messages[0].content == "msg-39"
        assert messages[-1].content == "msg-10"

    async def test_short_term_includes_all_today(self, session: AsyncSession) -> None:
        companion_id = await _create_companion(session)
        store = MessageStore(session)

        today = datetime(2026, 2, 14, 9, 0, 0)
        yesterday = today - timedelta(days=1)

        for i in range(5):
            await store.save_message(
                companion_id, "assistant", f"y-{i}", timestamp=yesterday + timedelta(minutes=i)
            )
        for i in range(35):
            await store.save_message(
                companion_id, "user", f"t-{i}", timestamp=today + timedelta(minutes=i)
            )

        messages = await store.get_short_term(companion_id, limit=30, now=today)
        assert len(messages) == 35
        assert all(message.content.startswith("t-") for message in messages)

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
