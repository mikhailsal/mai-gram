"""Tests for the shared chat import service."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from mai_gram.core.import_chat_service import (
    ImportChatConflictError,
    create_chat_from_import,
    import_into_existing_chat,
    parse_import_payload,
)
from mai_gram.db.models import Chat, Message

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def test_parse_import_payload_preserves_system_prompt() -> None:
    payload = parse_import_payload(
        json.dumps(
            [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hi"},
            ]
        )
    )

    assert payload.system_prompt == "Be precise."
    assert len(payload.messages_data) == 2


def test_parse_import_payload_falls_back_to_default_prompt() -> None:
    payload = parse_import_payload(json.dumps([{"role": "user", "content": "Hi"}]))

    assert payload.system_prompt == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_import_into_existing_chat_requires_chat(session: AsyncSession) -> None:
    payload = parse_import_payload(json.dumps([{"role": "user", "content": "Hi"}]))

    with pytest.raises(LookupError):
        await import_into_existing_chat(session, chat_id="missing-chat", payload=payload)


@pytest.mark.asyncio
async def test_import_into_existing_chat_saves_messages_and_disables_datetime(
    session: AsyncSession,
) -> None:
    session.add(
        Chat(
            id="existing-chat",
            user_id="user-1",
            bot_id="",
            llm_model="openai/test-model",
            system_prompt="prompt",
            send_datetime=True,
        )
    )
    await session.commit()
    payload = parse_import_payload(
        json.dumps(
            [
                {"role": "system", "content": "Imported prompt"},
                {"role": "user", "content": "Imported hello"},
                {"role": "assistant", "content": "Imported reply"},
            ]
        )
    )

    result = await import_into_existing_chat(session, chat_id="existing-chat", payload=payload)
    await session.commit()

    chat = (await session.execute(select(Chat).where(Chat.id == "existing-chat"))).scalar_one()
    messages = list(
        (
            await session.execute(
                select(Message).where(Message.chat_id == "existing-chat").order_by(Message.id)
            )
        ).scalars()
    )

    assert result.imported_count == 2
    assert result.system_prompt == "Imported prompt"
    assert chat.send_datetime is False
    assert [message.role for message in messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_create_chat_from_import_rejects_existing_chat(session: AsyncSession) -> None:
    session.add(
        Chat(
            id="existing-chat",
            user_id="user-1",
            bot_id="",
            llm_model="openai/test-model",
            system_prompt="prompt",
        )
    )
    await session.commit()
    payload = parse_import_payload(json.dumps([{"role": "user", "content": "Hi"}]))

    with pytest.raises(ImportChatConflictError):
        await create_chat_from_import(
            session,
            chat_id="existing-chat",
            user_id="user-2",
            bot_id="",
            llm_model="openai/test-model",
            timezone="UTC",
            payload=payload,
        )


@pytest.mark.asyncio
async def test_create_chat_from_import_creates_chat_and_messages(session: AsyncSession) -> None:
    payload = parse_import_payload(
        json.dumps(
            [
                {"role": "system", "content": "Imported prompt"},
                {"role": "user", "content": "Imported hello"},
                {"role": "assistant", "content": "Imported reply"},
            ]
        )
    )

    result = await create_chat_from_import(
        session,
        chat_id="new-chat",
        user_id="user-1",
        bot_id="bot-1",
        llm_model="openai/test-model",
        timezone="UTC",
        payload=payload,
    )
    await session.commit()

    chat = (await session.execute(select(Chat).where(Chat.id == "new-chat"))).scalar_one()
    messages = list(
        (
            await session.execute(
                select(Message).where(Message.chat_id == "new-chat").order_by(Message.id)
            )
        ).scalars()
    )

    assert result.chat_id == "new-chat"
    assert result.imported_count == 2
    assert chat.system_prompt == "Imported prompt"
    assert chat.send_datetime is False
    assert [message.role for message in messages] == ["user", "assistant"]
