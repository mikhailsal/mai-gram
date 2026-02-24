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
CURRENT_SCHEMA_VERSION = 7


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


@register_migration(5, "Add communication_style and verbosity to companions")
async def _migrate_v5(conn: AsyncConnection) -> None:
    """Version 5: add communication_style and verbosity columns to companions.

    These fields were previously baked into the system_prompt at creation time
    and lost.  Storing them separately allows the system prompt to be
    regenerated at runtime so that all companions (old and new) benefit from
    template improvements.

    Defaults match the onboarding flow (casual / concise).
    """
    result = await conn.execute(text("PRAGMA table_info(companions)"))
    columns = {row[1] for row in result.fetchall()}

    if "communication_style" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN communication_style "
                "VARCHAR(20) NOT NULL DEFAULT 'casual'"
            )
        )
        logger.info("Schema v5: added communication_style column to companions")
    else:
        logger.info("Schema v5: communication_style column already exists, skipping")

    if "verbosity" not in columns:
        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN verbosity "
                "VARCHAR(20) NOT NULL DEFAULT 'concise'"
            )
        )
        logger.info("Schema v5: added verbosity column to companions")
    else:
        logger.info("Schema v5: verbosity column already exists, skipping")


@register_migration(6, "Add llm_model to companions")
async def _migrate_v6(conn: AsyncConnection) -> None:
    """Version 6: add llm_model column to companions table.

    The LLM model is now stored per-companion rather than as a global setting.
    This protects companion identity -- changing the model in .env will only
    affect new companions, not existing ones.

    The model is the companion's "soul" -- the fundamental substrate that
    processes their memories and personality. Changing it would create a
    different entity wearing the same memories.

    For existing companions, we use the CURRENT configured model from settings,
    since that's what they've been running on. This is the best approximation
    we have -- we don't have historical data about what model was configured
    when each companion was created.
    """
    from mai_companion.config import get_settings

    result = await conn.execute(text("PRAGMA table_info(companions)"))
    columns = {row[1] for row in result.fetchall()}

    if "llm_model" not in columns:
        # Get the current model from settings -- this is what existing
        # companions have been using, so it's the correct value for them.
        settings = get_settings()
        current_model = settings.llm_model
        logger.info(
            "Schema v6: migrating existing companions to model '%s' (from current settings)",
            current_model,
        )

        await conn.execute(
            text(
                "ALTER TABLE companions ADD COLUMN llm_model "
                f"VARCHAR(100) NOT NULL DEFAULT '{current_model}'"
            )
        )
        logger.info("Schema v6: added llm_model column to companions")
    else:
        logger.info("Schema v6: llm_model column already exists, skipping")


@register_migration(7, "Add tool_calls and tool_call_id to messages")
async def _migrate_v7(conn: AsyncConnection) -> None:
    """Version 7: add tool_calls and tool_call_id columns to messages table.

    This enables storing the full conversation flow including tool calls,
    so the AI can "remember" that it used tools in past conversations.
    This prevents behavioral drift where the AI stops using tools because
    it has no memory of ever having done so.

    - tool_calls: JSON array of tool calls for assistant messages
    - tool_call_id: ID linking tool result messages to their calls
    """
    result = await conn.execute(text("PRAGMA table_info(messages)"))
    columns = {row[1] for row in result.fetchall()}

    if "tool_calls" not in columns:
        await conn.execute(
            text("ALTER TABLE messages ADD COLUMN tool_calls TEXT DEFAULT NULL")
        )
        logger.info("Schema v7: added tool_calls column to messages")
    else:
        logger.info("Schema v7: tool_calls column already exists, skipping")

    if "tool_call_id" not in columns:
        await conn.execute(
            text("ALTER TABLE messages ADD COLUMN tool_call_id VARCHAR(100) DEFAULT NULL")
        )
        logger.info("Schema v7: added tool_call_id column to messages")
    else:
        logger.info("Schema v7: tool_call_id column already exists, skipping")


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
