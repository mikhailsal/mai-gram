"""Tests for database engine, session management, and initialization."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from mai_gram.db.database import (
    close_db,
    get_engine,
    get_session,
    get_session_factory,
    init_db,
    reset_db_state,
)
from mai_gram.db.models import Base, Chat

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def _clean_db_state() -> None:
    """Ensure module-level singletons are reset between tests."""
    reset_db_state()
    yield  # type: ignore[misc]
    reset_db_state()


class TestGetEngine:

    async def test_creates_engine(self) -> None:
        engine = get_engine(TEST_DATABASE_URL)
        assert engine is not None
        assert isinstance(engine, AsyncEngine)
        await engine.dispose()

    async def test_returns_same_engine_on_second_call(self) -> None:
        engine1 = get_engine(TEST_DATABASE_URL)
        engine2 = get_engine(TEST_DATABASE_URL)
        assert engine1 is engine2
        await engine1.dispose()


class TestGetSessionFactory:

    async def test_raises_without_engine(self) -> None:
        with pytest.raises(RuntimeError, match="No database engine"):
            get_session_factory()

    async def test_creates_factory_with_engine(self) -> None:
        engine = get_engine(TEST_DATABASE_URL)
        factory = get_session_factory(engine)
        assert factory is not None
        await engine.dispose()


class TestGetSession:

    async def test_session_commits_on_success(self) -> None:
        engine = await init_db(TEST_DATABASE_URL)

        async with get_session() as session:
            chat = Chat(
                id="test-1@bot",
                user_id="test-1",
                bot_id="bot",
                llm_model="test/model",
                system_prompt="test",
            )
            session.add(chat)

        async with get_session() as session:
            result = await session.execute(
                select(Chat).where(Chat.id == "test-1@bot")
            )
            loaded = result.scalar_one()
            assert loaded.llm_model == "test/model"

        await engine.dispose()

    async def test_session_rolls_back_on_exception(self) -> None:
        engine = await init_db(TEST_DATABASE_URL)

        with pytest.raises(ValueError, match="intentional"):
            async with get_session() as session:
                chat = Chat(
                    id="test-2@bot",
                    user_id="test-2",
                    bot_id="bot",
                    llm_model="test/model",
                    system_prompt="test",
                )
                session.add(chat)
                raise ValueError("intentional error")

        async with get_session() as session:
            result = await session.execute(
                select(Chat).where(Chat.id == "test-2@bot")
            )
            assert result.scalar_one_or_none() is None

        await engine.dispose()


class TestInitDb:

    async def test_creates_all_tables(self) -> None:
        engine = await init_db(TEST_DATABASE_URL)

        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            table_names = {row[0] for row in result.fetchall()}

        expected_tables = {
            "chats",
            "messages",
            "knowledge_entries",
            "schema_versions",
        }
        assert expected_tables.issubset(table_names), (
            f"Missing tables: {expected_tables - table_names}"
        )
        await engine.dispose()

    async def test_init_is_idempotent(self) -> None:
        engine1 = await init_db(TEST_DATABASE_URL)
        reset_db_state()
        engine2 = await init_db(TEST_DATABASE_URL)

        async with engine2.begin() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = result.fetchall()
            assert len(tables) > 0

        await engine1.dispose()
        await engine2.dispose()


class TestCloseDb:

    async def test_close_resets_state(self) -> None:
        await init_db(TEST_DATABASE_URL)
        await close_db()

        with pytest.raises(RuntimeError, match="No database engine"):
            get_session_factory()
