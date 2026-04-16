"""Simple schema versioning for mai-gram.

Provides a lightweight migration system that tracks schema versions in the
database. Each migration is a named async function that receives an
AsyncConnection and performs schema changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from mai_gram.db.models import SchemaVersion

logger = logging.getLogger(__name__)

MigrationFunc = Callable[[AsyncConnection], Coroutine[None, None, None]]


@dataclass
class Migration:
    """A single schema migration step."""

    version: int
    description: str
    migrate: MigrationFunc


_MIGRATIONS: list[Migration] = []

CURRENT_SCHEMA_VERSION = 6


def register_migration(version: int, description: str) -> Callable[[MigrationFunc], MigrationFunc]:
    """Decorator to register a migration function."""

    def decorator(func: MigrationFunc) -> MigrationFunc:
        _MIGRATIONS.append(Migration(version=version, description=description, migrate=func))
        _MIGRATIONS.sort(key=lambda m: m.version)
        return func

    return decorator


@register_migration(1, "Initial mai-gram schema")
async def _migrate_v1(conn: AsyncConnection) -> None:
    """Version 1: initial schema. Tables created by Base.metadata.create_all."""
    logger.info("Schema v1: initial tables verified")


@register_migration(2, "Add reasoning column and chat toggle columns")
async def _migrate_v2(conn: AsyncConnection) -> None:
    """Version 2: add Message.reasoning, Chat.show_reasoning/show_tool_calls/send_datetime."""
    existing_tables: dict[str, list[str]] = {}
    for table_name in ("messages", "chats"):
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        existing_tables[table_name] = [row[1] for row in result.fetchall()]

    if "reasoning" not in existing_tables["messages"]:
        await conn.execute(text("ALTER TABLE messages ADD COLUMN reasoning TEXT"))
        logger.info("Added messages.reasoning column")

    if "show_reasoning" not in existing_tables["chats"]:
        await conn.execute(
            text("ALTER TABLE chats ADD COLUMN show_reasoning BOOLEAN NOT NULL DEFAULT 0")
        )
        logger.info("Added chats.show_reasoning column")

    if "show_tool_calls" not in existing_tables["chats"]:
        await conn.execute(
            text("ALTER TABLE chats ADD COLUMN show_tool_calls BOOLEAN NOT NULL DEFAULT 0")
        )
        logger.info("Added chats.show_tool_calls column")

    if "send_datetime" not in existing_tables["chats"]:
        await conn.execute(
            text("ALTER TABLE chats ADD COLUMN send_datetime BOOLEAN NOT NULL DEFAULT 1")
        )
        logger.info("Added chats.send_datetime column")


@register_migration(3, "Add timezone and cut_above_message_id to chats")
async def _migrate_v3(conn: AsyncConnection) -> None:
    """Version 3: add Chat.timezone and Chat.cut_above_message_id."""
    result = await conn.execute(text("PRAGMA table_info(chats)"))
    existing_cols = [row[1] for row in result.fetchall()]

    if "timezone" not in existing_cols:
        await conn.execute(
            text("ALTER TABLE chats ADD COLUMN timezone VARCHAR(50) NOT NULL DEFAULT 'UTC'")
        )
        logger.info("Added chats.timezone column")

    if "cut_above_message_id" not in existing_cols:
        await conn.execute(text("ALTER TABLE chats ADD COLUMN cut_above_message_id INTEGER"))
        logger.info("Added chats.cut_above_message_id column")


@register_migration(4, "Add timezone column to messages")
async def _migrate_v4(conn: AsyncConnection) -> None:
    """Version 4: add Message.timezone to track timezone per-message."""
    result = await conn.execute(text("PRAGMA table_info(messages)"))
    existing_cols = [row[1] for row in result.fetchall()]

    if "timezone" not in existing_cols:
        await conn.execute(
            text("ALTER TABLE messages ADD COLUMN timezone VARCHAR(50) NOT NULL DEFAULT 'UTC'")
        )
        logger.info("Added messages.timezone column")


@register_migration(5, "Add prompt_name column to chats")
async def _migrate_v5(conn: AsyncConnection) -> None:
    """Version 5: add Chat.prompt_name for per-prompt config lookup at runtime."""
    result = await conn.execute(text("PRAGMA table_info(chats)"))
    existing_cols = [row[1] for row in result.fetchall()]

    if "prompt_name" not in existing_cols:
        await conn.execute(text("ALTER TABLE chats ADD COLUMN prompt_name VARCHAR(100)"))
        logger.info("Added chats.prompt_name column")


@register_migration(6, "Add per-message show_datetime column with backfill from chat settings")
async def _migrate_v6(conn: AsyncConnection) -> None:
    """Version 6: add Message.show_datetime to track datetime visibility per-message.

    Previously, datetime visibility was a chat-level toggle that retroactively
    affected all messages. Now each message records whether its timestamp should
    be shown to the LLM. Existing messages are backfilled from their chat's
    current send_datetime setting.
    """
    result = await conn.execute(text("PRAGMA table_info(messages)"))
    existing_cols = [row[1] for row in result.fetchall()]

    if "show_datetime" not in existing_cols:
        await conn.execute(
            text("ALTER TABLE messages ADD COLUMN show_datetime BOOLEAN NOT NULL DEFAULT 1")
        )
        logger.info("Added messages.show_datetime column")

        # Backfill: set show_datetime from each chat's current send_datetime setting
        await conn.execute(
            text(
                "UPDATE messages SET show_datetime = ("
                "  SELECT c.send_datetime FROM chats c WHERE c.id = messages.chat_id"
                ")"
            )
        )
        logger.info("Backfilled messages.show_datetime from chat send_datetime settings")


async def get_current_version(engine: AsyncEngine) -> int:
    """Get the current schema version from the database."""
    async with engine.begin() as conn:
        try:
            result = await conn.execute(
                select(SchemaVersion.version).order_by(SchemaVersion.version.desc()).limit(1)
            )
            row = result.scalar_one_or_none()
            return row if row is not None else 0
        except Exception:
            return 0


async def run_migrations(engine: AsyncEngine) -> None:
    """Run all pending migrations."""
    current_version = await get_current_version(engine)
    logger.info("Current schema version: %d", current_version)

    pending = [m for m in _MIGRATIONS if m.version > current_version]
    if not pending:
        logger.info("Database schema is up to date (v%d)", current_version)
        return

    for migration in pending:
        logger.info(
            "Applying migration v%d: %s",
            migration.version,
            migration.description,
        )
        async with engine.begin() as conn:
            await migration.migrate(conn)
            await conn.execute(
                text(
                    "INSERT INTO schema_versions (version, description) "
                    "VALUES (:version, :description)"
                ),
                {"version": migration.version, "description": migration.description},
            )
        logger.info("Migration v%d applied successfully", migration.version)

    logger.info("All migrations applied. Schema is now at v%d", pending[-1].version)
