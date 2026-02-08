"""Database models and session management.

Public API:
    - Base: SQLAlchemy declarative base
    - Companion, MoodState, RelationshipEvent, Message, DailySummary,
      KnowledgeEntry, SharedActivity, SchemaVersion: ORM models
    - init_db, close_db, get_session: database lifecycle and session management
    - run_migrations: schema versioning
"""

from mai_companion.db.database import close_db, get_session, init_db, reset_db_state
from mai_companion.db.migrations import run_migrations
from mai_companion.db.models import (
    Base,
    Companion,
    DailySummary,
    KnowledgeEntry,
    Message,
    MoodState,
    RelationshipEvent,
    SchemaVersion,
    SharedActivity,
)

__all__ = [
    "Base",
    "Companion",
    "DailySummary",
    "KnowledgeEntry",
    "Message",
    "MoodState",
    "RelationshipEvent",
    "SchemaVersion",
    "SharedActivity",
    "close_db",
    "get_session",
    "init_db",
    "reset_db_state",
    "run_migrations",
]
