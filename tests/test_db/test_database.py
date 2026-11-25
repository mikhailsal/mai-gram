"""Tests for database engine, session management, and initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from mai_gram.db.migrations import get_current_version, run_migrations
from mai_gram.db.models import Chat, Message

if TYPE_CHECKING:
    from pathlib import Path

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
            result = await session.execute(select(Chat).where(Chat.id == "test-1@bot"))
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
            result = await session.execute(select(Chat).where(Chat.id == "test-2@bot"))
            assert result.scalar_one_or_none() is None

        await engine.dispose()


class TestInitDb:
    async def test_creates_all_tables(self) -> None:
        engine = await init_db(TEST_DATABASE_URL)

        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
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
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = result.fetchall()
            assert len(tables) > 0

        await engine1.dispose()
        await engine2.dispose()


async def test_message_ids_are_not_recycled_when_upgrading_v9(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'history.db'}"
    engine = await init_db(url)
    async with engine.begin() as conn:
        schema = await conn.scalar(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'")
        )
        await conn.execute(text("DROP TABLE messages"))
        await conn.exec_driver_sql(schema.replace(" AUTOINCREMENT", ""))
        await conn.execute(
            text("INSERT INTO schema_versions (version, description) VALUES (9, 'v9')")
        )

    async with get_session() as session:
        chat = Chat(
            id="user@bot",
            user_id="user",
            bot_id="bot",
            llm_model="test/model",
            system_prompt="test",
        )
        session.add(chat)
        surviving = Message(
            chat_id=chat.id,
            role="assistant",
            content="Keep this",
            reasoning="Keep this too",
            timezone="Europe/Istanbul",
            show_datetime=False,
        )
        rolled_back = Message(chat_id=chat.id, role="assistant", content="Gone before migration")
        session.add_all([surviving, rolled_back])
        await session.flush()
        chat.cut_above_message_id = surviving.id
        await session.delete(rolled_back)

    await run_migrations(engine)
    assert await get_current_version(engine) == 11
    await close_db()
    engine = await init_db(url)
    await run_migrations(engine)
    async with get_session() as session:
        preserved = await session.get(Message, surviving.id)
        assert preserved is not None
        assert len(preserved.history_ref) == 32
        preserved_ref = preserved.history_ref
        assert preserved.content == surviving.content
        assert preserved.timestamp == surviving.timestamp
        assert preserved.reasoning == surviving.reasoning
        assert preserved.timezone == surviving.timezone
        assert preserved.show_datetime is False
        preserved_chat = await session.get(Chat, chat.id)
        assert preserved_chat is not None
        assert preserved_chat.cut_above_message_id == surviving.id
        replacement = Message(chat_id=chat.id, role="assistant", content="New reply")
        session.add(replacement)
        await session.flush()
        assert replacement.id > rolled_back.id
        assert replacement.history_ref != preserved_ref
        await session.delete(replacement)

    async with get_session() as session:
        later = Message(chat_id=chat.id, role="assistant", content="Later reply")
        session.add(later)
        await session.flush()
        assert later.id > replacement.id
    await close_db()


async def test_v10_upgrade_keeps_empty_table_sequence_and_gives_raw_inserts_tokens(
    tmp_path: Path,
) -> None:
    engine = await init_db(f"sqlite+aiosqlite:///{tmp_path / 'v10.db'}")
    async with engine.begin() as conn:
        await conn.execute(text("DROP INDEX ix_messages_history_ref"))
        await conn.execute(text("ALTER TABLE messages DROP COLUMN history_ref"))
        await conn.execute(
            text("INSERT INTO schema_versions (version, description) VALUES (10, 'v10')")
        )
        await conn.execute(
            text("INSERT INTO sqlite_sequence (name, seq) VALUES ('messages', 5000)")
        )
    await run_migrations(engine)
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "INSERT INTO messages (chat_id, role, content, timezone, show_datetime)"
                    " VALUES ('manual', 'assistant', 'Answer', 'UTC', 1)"
                    " RETURNING id, history_ref"
                )
            )
        ).one()
        assert row.id == 5001
        assert len(row.history_ref) == 32
    await close_db()


class TestCloseDb:
    async def test_close_resets_state(self) -> None:
        await init_db(TEST_DATABASE_URL)
        await close_db()

        with pytest.raises(RuntimeError, match="No database engine"):
            get_session_factory()
