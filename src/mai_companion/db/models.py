"""SQLAlchemy ORM models for mAI Companion.

Defines all database tables matching the data model:
- Companion: core companion configuration and personality
- MoodState: emotional state history (valence/arousal model)
- RelationshipEvent: relationship stage transitions
- Message: conversation history
- DailySummary: compressed daily conversation summaries
- KnowledgeEntry: wiki-like structured facts about user and companion
- SharedActivity: logged shared activities (watch/read/learn/play)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Companion(Base):
    """Core companion entity with personality configuration."""

    __tablename__ = "companions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="neutral",
        doc="Companion's gender identity: 'male', 'female', or 'neutral'",
    )
    human_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="English",
        doc="Human companion's preferred language for communication",
    )
    language_style: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        default=None,
        doc="Language style/variant specification (e.g., 'pre-revolutionary orthography', 'like a 10-year-old child')",
    )
    personality_traits: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        doc="JSON-encoded personality trait values",
    )
    mood_volatility: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        doc="How dramatically and frequently mood shifts (0.0=steady, 1.0=volatile)",
    )
    temperature: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.7,
        doc="LLM temperature derived from personality traits",
    )
    avatar_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        doc="File path to the generated avatar image",
    )
    communication_style: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="casual",
        doc="Communication style: 'casual', 'balanced', or 'formal'",
    )
    verbosity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="concise",
        doc="Response verbosity: 'concise', 'normal', or 'detailed'",
    )
    system_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Legacy base system prompt (kept for reference; prompt is now regenerated at runtime)",
    )
    relationship_stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="getting_to_know",
        doc="Current relationship stage identifier",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    mood_states: Mapped[list[MoodState]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
        order_by="MoodState.timestamp.desc()",
    )
    relationship_events: Mapped[list[RelationshipEvent]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
        order_by="RelationshipEvent.transitioned_at.desc()",
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
        order_by="Message.timestamp.desc()",
    )
    daily_summaries: Mapped[list[DailySummary]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
        order_by="DailySummary.summary_date.desc()",
    )
    knowledge_entries: Mapped[list[KnowledgeEntry]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
    )
    shared_activities: Mapped[list[SharedActivity]] = relationship(
        back_populates="companion",
        cascade="all, delete-orphan",
        order_by="SharedActivity.created_at.desc()",
    )

    def __repr__(self) -> str:
        return f"<Companion(id={self.id!r}, name={self.name!r}, stage={self.relationship_stage!r})>"


class MoodState(Base):
    """Emotional state snapshot using the valence/arousal model.

    Valence: positive ↔ negative (-1.0 to 1.0)
    Arousal: energetic ↔ calm (-1.0 to 1.0)
    Together they produce mood labels like "excited", "melancholic", "irritated", "serene".
    """

    __tablename__ = "mood_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    valence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Positive ↔ negative axis (-1.0 to 1.0)",
    )
    arousal: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Energetic ↔ calm axis (-1.0 to 1.0)",
    )
    label: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Human-readable mood label (e.g. 'excited', 'melancholic')",
    )
    cause: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        doc="What triggered this mood shift (conversation event, spontaneous, etc.)",
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )

    companion: Mapped[Companion] = relationship(back_populates="mood_states")

    def __repr__(self) -> str:
        return (
            f"<MoodState(label={self.label!r}, "
            f"valence={self.valence:.2f}, arousal={self.arousal:.2f})>"
        )


class RelationshipEvent(Base):
    """Records relationship stage transitions and their triggers."""

    __tablename__ = "relationship_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="The relationship stage transitioned TO",
    )
    trigger_reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Why this transition occurred",
    )
    transitioned_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )

    companion: Mapped[Companion] = relationship(back_populates="relationship_events")

    def __repr__(self) -> str:
        return f"<RelationshipEvent(stage={self.stage!r}, at={self.transitioned_at!r})>"


class Message(Base):
    """Individual conversation message (user or companion)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Message role: 'user', 'assistant', or 'system'",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )
    is_proactive: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether this message was self-initiated by the companion",
    )

    companion: Mapped[Companion] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"<Message(role={self.role!r}, content={preview!r})>"


class DailySummary(Base):
    """Compressed summary of a day's conversations."""

    __tablename__ = "daily_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    summary_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        doc="The date this summary covers",
    )
    summary_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="LLM-generated summary of the day's conversations",
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )

    companion: Mapped[Companion] = relationship(back_populates="daily_summaries")

    def __repr__(self) -> str:
        return f"<DailySummary(date={self.summary_date!r})>"


class KnowledgeEntry(Base):
    """Wiki-like structured fact about the user or companion.

    Used for persistent knowledge that should always be available in context,
    e.g. user's name, preferences, important dates, life events.
    """

    __tablename__ = "knowledge_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Fact category: 'user_info', 'preference', 'life_event', 'opinion', etc.",
    )
    key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Fact identifier within the category (e.g. 'name', 'favorite_food')",
    )
    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The fact content",
    )
    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        doc="Importance score 0.0-1.0. High = always in context, low = may be forgotten",
    )
    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="User explicitly pinned this memory (never forgotten)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )

    companion: Mapped[Companion] = relationship(back_populates="knowledge_entries")

    def __repr__(self) -> str:
        return (
            f"<KnowledgeEntry(category={self.category!r}, "
            f"key={self.key!r}, importance={self.importance:.2f})>"
        )


class SharedActivity(Base):
    """Logged shared activity between user and companion."""

    __tablename__ = "shared_activities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    activity_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        doc="Activity type: 'watch', 'read', 'learn', 'play'",
    )
    reference_url: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
        doc="URL of the shared content (YouTube link, article, etc.)",
    )
    companion_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Companion's notes/summary about the activity",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    companion_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("companions.id", ondelete="CASCADE"),
        nullable=False,
    )

    companion: Mapped[Companion] = relationship(back_populates="shared_activities")

    def __repr__(self) -> str:
        return f"<SharedActivity(type={self.activity_type!r}, at={self.created_at!r})>"


class SchemaVersion(Base):
    """Tracks database schema version for simple migration support."""

    __tablename__ = "schema_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Schema version number",
    )
    description: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="Brief description of what changed in this version",
    )
    applied_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<SchemaVersion(version={self.version}, desc={self.description!r})>"
