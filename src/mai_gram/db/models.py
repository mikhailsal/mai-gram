"""SQLAlchemy ORM models for mai-gram.

Defines all database tables:
- Chat: per-user chat configuration (model + system prompt)
- Message: conversation history
- KnowledgeEntry: wiki-like structured facts
- SchemaVersion: migration tracking
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — needed at runtime for SQLAlchemy Mapped[datetime]

from sqlalchemy import (
    Boolean,
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


class Chat(Base):
    """Per-user chat configuration binding a Telegram user to a model and system prompt."""

    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
        doc="Composite key: user_id@bot_id",
    )
    user_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Telegram user ID",
    )
    bot_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Telegram bot username",
    )
    llm_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="openai/gpt-4o-mini",
        doc="OpenRouter model identifier chosen by the user",
    )
    system_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="System prompt text (from template or user-provided)",
    )
    prompt_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        default=None,
        doc=(
            "Name of the selected prompt template (e.g. 'default', 'coder'). "
            "Used to load per-prompt config at runtime. NULL for custom prompts."
        ),
    )
    display_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        default=None,
        doc="Optional display name for this chat",
    )
    show_reasoning: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether to display LLM reasoning in Telegram messages",
    )
    show_tool_calls: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether to display tool call details in Telegram messages",
    )
    send_datetime: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether to prepend date/time to user messages sent to the LLM",
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="UTC",
        doc="IANA timezone for this chat (e.g. Europe/Moscow). Affects timestamps shown to LLM.",
    )
    cut_above_message_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
        doc=(
            "DB message ID at which to cut history. Messages before this ID "
            "are excluded from LLM context but remain searchable."
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.timestamp.desc()",
    )
    knowledge_entries: Mapped[list[KnowledgeEntry]] = relationship(
        back_populates="chat",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Chat(id={self.id!r}, model={self.llm_model!r})>"


class Message(Base):
    """Individual conversation message.

    Messages can represent:
    - User messages (role='user')
    - Assistant messages (role='assistant'), optionally with tool_calls
    - Tool result messages (role='tool') with tool_call_id
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="Message role: 'user', 'assistant', 'tool', or 'system'",
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
    chat_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_calls: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc=(
            "JSON-serialized list of tool calls made by the assistant. "
            "Format: [{id, name, arguments}, ...]. Only set for role='assistant'."
        ),
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        default=None,
        doc="ID of the tool call this message is responding to. Only set for role='tool'.",
    )
    reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        doc="LLM reasoning/thinking content. Only set for role='assistant'.",
    )
    timezone: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="UTC",
        doc="IANA timezone active when this message was created.",
    )
    show_datetime: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc=(
            "Whether this message's timestamp should be visible to the LLM. "
            "Captured at save time from the chat's send_datetime setting, "
            "so toggling only affects future messages."
        ),
    )

    chat: Mapped[Chat] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"<Message(role={self.role!r}, content={preview!r})>"


class KnowledgeEntry(Base):
    """Wiki-like structured fact about the user or AI.

    Used for persistent knowledge that should always be available in context.
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
        doc="Fact identifier within the category",
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
        doc="Importance score 0.0-1.0. High = always in context",
    )
    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Pinned entries are never forgotten",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    chat_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
    )

    chat: Mapped[Chat] = relationship(back_populates="knowledge_entries")

    def __repr__(self) -> str:
        return (
            f"<KnowledgeEntry(category={self.category!r}, "
            f"key={self.key!r}, importance={self.importance:.2f})>"
        )


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
