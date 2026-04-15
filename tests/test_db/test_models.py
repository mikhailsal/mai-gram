"""Tests for the database models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.db.models import Chat, KnowledgeEntry, Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TestChat:
    async def test_create_chat(self, session: AsyncSession) -> None:
        chat = Chat(
            id="user1@testbot",
            user_id="user1",
            bot_id="testbot",
            llm_model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
        )
        session.add(chat)
        await session.flush()

        result = await session.execute(select(Chat).where(Chat.id == "user1@testbot"))
        fetched = result.scalar_one()
        assert fetched.user_id == "user1"
        assert fetched.bot_id == "testbot"
        assert fetched.llm_model == "openai/gpt-4o-mini"
        assert fetched.system_prompt == "You are a helpful assistant."

    async def test_chat_repr(self, session: AsyncSession) -> None:
        chat = Chat(
            id="u@b",
            user_id="u",
            bot_id="b",
            llm_model="test/model",
            system_prompt="test",
        )
        assert "test/model" in repr(chat)


class TestMessage:
    async def test_create_message(self, session: AsyncSession) -> None:
        chat = Chat(
            id="user1@testbot",
            user_id="user1",
            bot_id="testbot",
            llm_model="openai/gpt-4o-mini",
            system_prompt="test",
        )
        session.add(chat)
        await session.flush()

        msg = Message(chat_id=chat.id, role="user", content="Hello")
        session.add(msg)
        await session.flush()

        result = await session.execute(select(Message).where(Message.chat_id == chat.id))
        fetched = result.scalar_one()
        assert fetched.role == "user"
        assert fetched.content == "Hello"

    async def test_message_with_tool_calls(self, session: AsyncSession) -> None:
        chat = Chat(
            id="user1@testbot",
            user_id="user1",
            bot_id="testbot",
            llm_model="openai/gpt-4o-mini",
            system_prompt="test",
        )
        session.add(chat)
        await session.flush()

        msg = Message(
            chat_id=chat.id,
            role="assistant",
            content="Let me check...",
            tool_calls='[{"id":"tc1","name":"wiki_create","arguments":"{}"}]',
        )
        session.add(msg)
        await session.flush()
        assert msg.tool_calls is not None

    async def test_tool_message(self, session: AsyncSession) -> None:
        chat = Chat(
            id="user1@testbot",
            user_id="user1",
            bot_id="testbot",
            llm_model="openai/gpt-4o-mini",
            system_prompt="test",
        )
        session.add(chat)
        await session.flush()

        msg = Message(
            chat_id=chat.id,
            role="tool",
            content="Done.",
            tool_call_id="tc1",
        )
        session.add(msg)
        await session.flush()
        assert msg.tool_call_id == "tc1"


class TestKnowledgeEntry:
    async def test_create_entry(self, session: AsyncSession) -> None:
        chat = Chat(
            id="user1@testbot",
            user_id="user1",
            bot_id="testbot",
            llm_model="openai/gpt-4o-mini",
            system_prompt="test",
        )
        session.add(chat)
        await session.flush()

        entry = KnowledgeEntry(
            chat_id=chat.id,
            category="user_info",
            key="name",
            value="Alice",
            importance=9000.0,
        )
        session.add(entry)
        await session.flush()

        result = await session.execute(
            select(KnowledgeEntry).where(KnowledgeEntry.chat_id == chat.id)
        )
        fetched = result.scalar_one()
        assert fetched.key == "name"
        assert fetched.value == "Alice"
        assert fetched.importance == 9000.0
