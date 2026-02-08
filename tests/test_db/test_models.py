"""Tests for SQLAlchemy ORM models.

Verifies that all models can be created, queried, and that relationships
and cascades work correctly.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mai_companion.db.models import (
    Companion,
    DailySummary,
    KnowledgeEntry,
    Message,
    MoodState,
    RelationshipEvent,
    SchemaVersion,
    SharedActivity,
)


class TestCompanionModel:
    """Tests for the Companion model."""

    async def test_create_companion_with_defaults(self, session: AsyncSession) -> None:
        """A companion can be created with just a name, getting sensible defaults."""
        companion = Companion(name="Aria")
        session.add(companion)
        await session.flush()

        assert companion.id is not None
        assert len(companion.id) == 36  # UUID format
        assert companion.name == "Aria"
        assert companion.mood_volatility == 0.5
        assert companion.temperature == 0.7
        assert companion.relationship_stage == "getting_to_know"
        assert companion.personality_traits == "{}"
        assert companion.system_prompt == ""
        assert companion.avatar_path is None

    async def test_create_companion_with_all_fields(self, session: AsyncSession) -> None:
        """A companion can be created with all fields specified."""
        companion = Companion(
            id="test-companion-001",
            name="Luna",
            personality_traits='{"warmth": 0.8, "humor": 0.6}',
            mood_volatility=0.3,
            temperature=0.9,
            avatar_path="/data/avatars/luna.png",
            system_prompt="You are Luna, a warm and humorous companion.",
            relationship_stage="building_trust",
        )
        session.add(companion)
        await session.flush()

        result = await session.execute(
            select(Companion).where(Companion.id == "test-companion-001")
        )
        loaded = result.scalar_one()
        assert loaded.name == "Luna"
        assert loaded.mood_volatility == 0.3
        assert loaded.temperature == 0.9
        assert loaded.avatar_path == "/data/avatars/luna.png"
        assert loaded.relationship_stage == "building_trust"

    async def test_companion_repr(self, session: AsyncSession) -> None:
        """The repr is readable and useful for debugging."""
        companion = Companion(id="abc", name="Aria", relationship_stage="deep_bond")
        assert "Aria" in repr(companion)
        assert "deep_bond" in repr(companion)


class TestMoodStateModel:
    """Tests for the MoodState model."""

    async def test_create_mood_state(self, session: AsyncSession) -> None:
        companion = Companion(id="c1", name="Test")
        session.add(companion)
        await session.flush()

        mood = MoodState(
            valence=0.7,
            arousal=0.5,
            label="excited",
            cause="fun conversation",
            companion_id="c1",
        )
        session.add(mood)
        await session.flush()

        assert mood.id is not None
        assert mood.valence == 0.7
        assert mood.arousal == 0.5
        assert mood.label == "excited"
        assert mood.cause == "fun conversation"

    async def test_mood_state_without_cause(self, session: AsyncSession) -> None:
        """Cause is optional (spontaneous mood shifts have no specific cause)."""
        companion = Companion(id="c2", name="Test")
        session.add(companion)
        await session.flush()

        mood = MoodState(valence=-0.3, arousal=-0.5, label="melancholic", companion_id="c2")
        session.add(mood)
        await session.flush()

        assert mood.cause is None

    async def test_mood_state_repr(self) -> None:
        mood = MoodState(valence=0.5, arousal=-0.2, label="serene", companion_id="x")
        assert "serene" in repr(mood)


class TestRelationshipEventModel:
    """Tests for the RelationshipEvent model."""

    async def test_create_relationship_event(self, session: AsyncSession) -> None:
        companion = Companion(id="c3", name="Test")
        session.add(companion)
        await session.flush()

        event = RelationshipEvent(
            stage="building_trust",
            trigger_reason="Reached 50 meaningful exchanges",
            companion_id="c3",
        )
        session.add(event)
        await session.flush()

        assert event.id is not None
        assert event.stage == "building_trust"
        assert event.trigger_reason == "Reached 50 meaningful exchanges"


class TestMessageModel:
    """Tests for the Message model."""

    async def test_create_user_message(self, session: AsyncSession) -> None:
        companion = Companion(id="c4", name="Test")
        session.add(companion)
        await session.flush()

        msg = Message(
            role="user",
            content="Hello, how are you?",
            companion_id="c4",
        )
        session.add(msg)
        await session.flush()

        assert msg.id is not None
        assert msg.role == "user"
        assert msg.is_proactive is False

    async def test_create_proactive_assistant_message(self, session: AsyncSession) -> None:
        companion = Companion(id="c5", name="Test")
        session.add(companion)
        await session.flush()

        msg = Message(
            role="assistant",
            content="I was thinking about what you said earlier...",
            companion_id="c5",
            is_proactive=True,
        )
        session.add(msg)
        await session.flush()

        assert msg.is_proactive is True

    async def test_message_repr_truncates_long_content(self) -> None:
        msg = Message(
            role="user",
            content="A" * 100,
            companion_id="x",
        )
        r = repr(msg)
        assert "..." in r
        assert len(r) < 120


class TestDailySummaryModel:
    """Tests for the DailySummary model."""

    async def test_create_daily_summary(self, session: AsyncSession) -> None:
        companion = Companion(id="c6", name="Test")
        session.add(companion)
        await session.flush()

        summary = DailySummary(
            summary_date=date(2026, 2, 8),
            summary_text="User and companion discussed weekend plans and shared a recipe.",
            companion_id="c6",
        )
        session.add(summary)
        await session.flush()

        assert summary.id is not None
        assert summary.summary_date == date(2026, 2, 8)


class TestKnowledgeEntryModel:
    """Tests for the KnowledgeEntry model."""

    async def test_create_knowledge_entry(self, session: AsyncSession) -> None:
        companion = Companion(id="c7", name="Test")
        session.add(companion)
        await session.flush()

        entry = KnowledgeEntry(
            category="user_info",
            key="name",
            value="Alice",
            importance=1.0,
            companion_id="c7",
        )
        session.add(entry)
        await session.flush()

        assert entry.id is not None
        assert entry.importance == 1.0
        assert entry.is_pinned is False

    async def test_pinned_knowledge_entry(self, session: AsyncSession) -> None:
        companion = Companion(id="c8", name="Test")
        session.add(companion)
        await session.flush()

        entry = KnowledgeEntry(
            category="life_event",
            key="birthday",
            value="March 15",
            importance=1.0,
            is_pinned=True,
            companion_id="c8",
        )
        session.add(entry)
        await session.flush()

        assert entry.is_pinned is True


class TestSharedActivityModel:
    """Tests for the SharedActivity model."""

    async def test_create_shared_activity(self, session: AsyncSession) -> None:
        companion = Companion(id="c9", name="Test")
        session.add(companion)
        await session.flush()

        activity = SharedActivity(
            activity_type="watch",
            reference_url="https://youtube.com/watch?v=example",
            companion_notes="A fascinating video about space exploration.",
            companion_id="c9",
        )
        session.add(activity)
        await session.flush()

        assert activity.id is not None
        assert activity.activity_type == "watch"

    async def test_shared_activity_without_url(self, session: AsyncSession) -> None:
        """Activities like games don't need a URL."""
        companion = Companion(id="c10", name="Test")
        session.add(companion)
        await session.flush()

        activity = SharedActivity(
            activity_type="play",
            companion_id="c10",
        )
        session.add(activity)
        await session.flush()

        assert activity.reference_url is None
        assert activity.companion_notes is None


class TestSchemaVersionModel:
    """Tests for the SchemaVersion model."""

    async def test_create_schema_version(self, session: AsyncSession) -> None:
        version = SchemaVersion(
            version=1,
            description="Initial schema creation",
        )
        session.add(version)
        await session.flush()

        assert version.id is not None
        assert version.version == 1


class TestRelationships:
    """Tests for model relationships and cascading."""

    async def test_companion_has_mood_states(self, session: AsyncSession) -> None:
        """Mood states can be accessed via the companion relationship."""
        companion = Companion(id="r1", name="Test")
        session.add(companion)
        await session.flush()

        mood1 = MoodState(valence=0.5, arousal=0.3, label="happy", companion_id="r1")
        mood2 = MoodState(valence=-0.2, arousal=-0.4, label="calm", companion_id="r1")
        session.add_all([mood1, mood2])
        await session.flush()

        result = await session.execute(
            select(Companion).where(Companion.id == "r1")
        )
        loaded = result.scalar_one()
        await session.refresh(loaded, ["mood_states"])
        assert len(loaded.mood_states) == 2

    async def test_companion_has_messages(self, session: AsyncSession) -> None:
        companion = Companion(id="r2", name="Test")
        session.add(companion)
        await session.flush()

        msg1 = Message(role="user", content="Hello", companion_id="r2")
        msg2 = Message(role="assistant", content="Hi there!", companion_id="r2")
        session.add_all([msg1, msg2])
        await session.flush()

        result = await session.execute(
            select(Companion).where(Companion.id == "r2")
        )
        loaded = result.scalar_one()
        await session.refresh(loaded, ["messages"])
        assert len(loaded.messages) == 2
        roles = {m.role for m in loaded.messages}
        assert roles == {"user", "assistant"}

    async def test_cascade_delete_removes_children(self, session: AsyncSession) -> None:
        """Deleting a companion cascades to all related entities."""
        companion = Companion(id="r3", name="TestCascade")
        session.add(companion)
        await session.flush()

        # Add various child entities
        session.add_all([
            MoodState(valence=0.5, arousal=0.3, label="happy", companion_id="r3"),
            Message(role="user", content="Hello", companion_id="r3"),
            KnowledgeEntry(
                category="user_info", key="name", value="Bob", companion_id="r3"
            ),
            DailySummary(
                summary_date=date(2026, 1, 1),
                summary_text="Test summary",
                companion_id="r3",
            ),
            RelationshipEvent(
                stage="building_trust",
                trigger_reason="test",
                companion_id="r3",
            ),
            SharedActivity(activity_type="play", companion_id="r3"),
        ])
        await session.flush()

        # Delete the companion
        await session.delete(companion)
        await session.flush()

        # Verify all children are gone
        for model in [MoodState, Message, KnowledgeEntry, DailySummary,
                       RelationshipEvent, SharedActivity]:
            result = await session.execute(select(model))
            assert result.scalars().all() == [], f"{model.__name__} not cascade-deleted"
