"""Simple schema versioning for mai-gram.

Provides a lightweight migration system that tracks schema versions in the
database. Each migration is a named async function that receives an
AsyncConnection and performs schema changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Coroutine

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

CURRENT_SCHEMA_VERSION = 1


def register_migration(
    version: int, description: str
) -> Callable[[MigrationFunc], MigrationFunc]:
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


async def get_current_version(engine: AsyncEngine) -> int:
    """Get the current schema version from the database."""
    async with engine.begin() as conn:
        try:
            result = await conn.execute(
                select(SchemaVersion.version)
                .order_by(SchemaVersion.version.desc())
                .limit(1)
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
