"""Tests for the schema migration system."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mai_companion.db.database import reset_db_state
from mai_companion.db.migrations import CURRENT_SCHEMA_VERSION, get_current_version, run_migrations
from mai_companion.db.models import Base, SchemaVersion

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def _clean_db_state() -> None:
    """Ensure module-level singletons are reset between tests."""
    reset_db_state()
    yield  # type: ignore[misc]
    reset_db_state()


@pytest.fixture
async def fresh_engine() -> AsyncEngine:
    """Create a fresh in-memory engine with all tables."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine  # type: ignore[misc]
    await engine.dispose()


class TestGetCurrentVersion:
    """Tests for get_current_version()."""

    async def test_returns_zero_for_fresh_db(self, fresh_engine: AsyncEngine) -> None:
        version = await get_current_version(fresh_engine)
        assert version == 0

    async def test_returns_version_after_migration(self, fresh_engine: AsyncEngine) -> None:
        await run_migrations(fresh_engine)
        version = await get_current_version(fresh_engine)
        assert version == CURRENT_SCHEMA_VERSION


class TestRunMigrations:
    """Tests for run_migrations()."""

    async def test_applies_all_migrations(self, fresh_engine: AsyncEngine) -> None:
        await run_migrations(fresh_engine)

        async with fresh_engine.begin() as conn:
            result = await conn.execute(
                select(SchemaVersion.version, SchemaVersion.description).order_by(
                    SchemaVersion.version
                )
            )
            rows = result.all()

        assert len(rows) == CURRENT_SCHEMA_VERSION
        assert rows[0][0] == 1  # version
        assert "Initial" in rows[0][1]  # description

    async def test_idempotent_migration(self, fresh_engine: AsyncEngine) -> None:
        """Running migrations twice doesn't duplicate version entries."""
        await run_migrations(fresh_engine)
        await run_migrations(fresh_engine)

        async with fresh_engine.begin() as conn:
            result = await conn.execute(
                select(SchemaVersion).order_by(SchemaVersion.version)
            )
            versions = result.scalars().all()

        assert len(versions) == CURRENT_SCHEMA_VERSION
