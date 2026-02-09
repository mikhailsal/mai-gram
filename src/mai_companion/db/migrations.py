"""Simple schema versioning for mAI Companion.

Provides a lightweight migration system that tracks schema versions in the
database. Each migration is a named async function that receives an
AsyncConnection and performs schema changes.

This is intentionally simple -- for a single-user self-hosted app, we don't
need Alembic's full power. If the project grows significantly, consider
migrating to Alembic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Coroutine

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from mai_companion.db.models import SchemaVersion

logger = logging.getLogger(__name__)

# Type alias for a migration function
MigrationFunc = Callable[[AsyncConnection], Coroutine[None, None, None]]


@dataclass
class Migration:
    """A single schema migration step."""

    version: int
    description: str
    migrate: MigrationFunc


# Registry of all migrations in order
_MIGRATIONS: list[Migration] = []

# Current schema version (matches the highest migration version)
CURRENT_SCHEMA_VERSION = 4


def register_migration(
    version: int, description: str
) -> Callable[[MigrationFunc], MigrationFunc]:
    """Decorator to register a migration function.

    Usage::

        @register_migration(2, "Add mood_label index")
        async def migrate_v2(conn: AsyncConnection) -> None:
            await conn.execute(text("CREATE INDEX ..."))
    """

    def decorator(func: MigrationFunc) -> MigrationFunc:
        _MIGRATIONS.append(Migration(version=version, description=description, migrate=func))
        # Keep sorted by version
        _MIGRATIONS.sort(key=lambda m: m.version)
        return func

    return decorator


# -- Built-in migrations --


@register_migration(1, "Initial schema creation")
async def _migrate_v1(conn: AsyncConnection) -> None:
    """Version 1: initial schema.

    Tables are already created by Base.metadata.create_all in init_db(),
    so this migration just records the version.
    """
    # Nothing to do -- tables created by create_all
    logger.info("Schema v1: initial tables verified")


@register_migration(2, "Add human_language to companions")
async def _migrate_v2(conn: AsyncConnection) -> None:
    """Version 2: add human_language column to companions table.

    Part of Phase 3 -- personality system needs to know the human's
    preferred language for multilingual onboarding and response generation.
    """
    # Check if column already exists (create_all may have added it)
    result = await conn.execute(text("PRAGMA table_info(companions)"))
    columns = {row[1] for row in result.fetchall()}
    if "human_language" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN human_language "
                "VARCHAR(50) NOT NULL DEFAULT 'English'"
            )
        )
        logger.info("Schema v2: added human_language column to companions")
    else:
        logger.info("Schema v2: human_language column already exists, skipping")


@register_migration(3, "Add gender to companions")
async def _migrate_v3(conn: AsyncConnection) -> None:
    """Version 3: add gender column to companions table.

    Part of Phase 4 improvements -- companions now have a gender identity
    inferred from their name, which affects how they use grammatical gender
    in gendered languages like Russian, Spanish, French.
    """
    # Check if column already exists (create_all may have added it)
    result = await conn.execute(text("PRAGMA table_info(companions)"))
    columns = {row[1] for row in result.fetchall()}
    if "gender" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN gender "
                "VARCHAR(20) NOT NULL DEFAULT 'neutral'"
            )
        )
        logger.info("Schema v3: added gender column to companions")
    else:
        logger.info("Schema v3: gender column already exists, skipping")


@register_migration(4, "Add language_style to companions")
async def _migrate_v4(conn: AsyncConnection) -> None:
    """Version 4: add language_style column to companions table.

    Supports language style/variant specifications like 'pre-revolutionary orthography',
    'like a 10-year-old child', 'British variant', etc. These affect how the AI
    translates and responds in the human's language.
    """
    # Check if column already exists (create_all may have added it)
    result = await conn.execute(text("PRAGMA table_info(companions)"))
    columns = {row[1] for row in result.fetchall()}
    if "language_style" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN language_style "
                "VARCHAR(200) DEFAULT NULL"
            )
        )
        logger.info("Schema v4: added language_style column to companions")
    else:
        logger.info("Schema v4: language_style column already exists, skipping")


# -- Migration runner --


async def get_current_version(engine: AsyncEngine) -> int:
    """Get the current schema version from the database.

    Returns 0 if no version has been recorded yet.
    """
    async with engine.begin() as conn:
        # Check if the schema_versions table exists
        try:
            result = await conn.execute(
                select(SchemaVersion.version)
                .order_by(SchemaVersion.version.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return row if row is not None else 0
        except Exception:
            # Table doesn't exist yet
            return 0


async def run_migrations(engine: AsyncEngine) -> None:
    """Run all pending migrations.

    Compares the current DB version against registered migrations and
    applies any that haven't been run yet.
    """
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
            # Record the migration
            await conn.execute(
                text(
                    "INSERT INTO schema_versions (version, description) "
                    "VALUES (:version, :description)"
                ),
                {"version": migration.version, "description": migration.description},
            )
        logger.info("Migration v%d applied successfully", migration.version)

    logger.info("All migrations applied. Schema is now at v%d", pending[-1].version)
