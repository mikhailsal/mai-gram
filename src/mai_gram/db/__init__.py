"""Database models and session management.

Public API:
    - Base: SQLAlchemy declarative base
    - Chat, Message, KnowledgeEntry, SchemaVersion: ORM models
    - init_db, close_db, get_session: database lifecycle and session management
    - run_migrations: schema versioning
"""

from mai_gram.db.database import close_db, get_session, init_db, reset_db_state
from mai_gram.db.migrations import run_migrations
from mai_gram.db.models import (
    Base,
    Chat,
    KnowledgeEntry,
    Message,
    SchemaVersion,
)

__all__ = [
    "Base",
    "Chat",
    "KnowledgeEntry",
    "Message",
    "SchemaVersion",
    "close_db",
    "get_session",
    "init_db",
    "reset_db_state",
    "run_migrations",
]
