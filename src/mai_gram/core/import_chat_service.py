"""Shared chat import workflows for CLI and Telegram adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from mai_gram.core.importer import extract_system_prompt, parse_import_json, save_imported_messages
from mai_gram.db.models import Chat
from mai_gram.memory.messages import MessageStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from mai_gram.response_templates.base import ResponseTemplate


class ImportChatConflictError(Exception):
    """Raised when an import would collide with an existing chat."""


@dataclass(frozen=True, slots=True)
class ParsedImportPayload:
    """Normalized import payload plus the system prompt to preserve."""

    messages_data: list[dict[str, object]]
    system_prompt: str


@dataclass(frozen=True, slots=True)
class ImportedChatResult:
    """Persisted import result used by both CLI and Telegram flows."""

    chat_id: str
    imported_count: int
    system_prompt: str


def parse_import_payload(data: str | bytes) -> ParsedImportPayload:
    """Parse raw import JSON and resolve the effective system prompt."""
    messages_data = parse_import_json(data)
    system_prompt = extract_system_prompt(messages_data) or "You are a helpful assistant."
    return ParsedImportPayload(messages_data=messages_data, system_prompt=system_prompt)


async def import_into_existing_chat(
    session: AsyncSession,
    *,
    chat_id: str,
    payload: ParsedImportPayload,
    reasoning_template: ResponseTemplate | None = None,
) -> ImportedChatResult:
    """Import parsed messages into an already configured chat."""
    result = await session.execute(select(Chat).where(Chat.id == chat_id))
    chat = result.scalar_one_or_none()
    if chat is None:
        raise LookupError(chat_id)

    message_store = MessageStore(session)
    imported_count = await save_imported_messages(
        chat_id,
        payload.messages_data,
        message_store,
        reasoning_template=reasoning_template,
    )
    chat.send_datetime = False
    return ImportedChatResult(
        chat_id=chat_id,
        imported_count=imported_count,
        system_prompt=payload.system_prompt,
    )


async def create_chat_from_import(
    session: AsyncSession,
    *,
    chat_id: str,
    user_id: str,
    bot_id: str,
    llm_model: str,
    timezone: str,
    payload: ParsedImportPayload,
    reasoning_template: ResponseTemplate | None = None,
) -> ImportedChatResult:
    """Create a new chat from parsed import data and persist its messages."""
    result = await session.execute(select(Chat).where(Chat.id == chat_id))
    existing = result.scalar_one_or_none()
    if existing is not None:
        raise ImportChatConflictError(chat_id)

    chat = Chat(
        id=chat_id,
        user_id=user_id,
        bot_id=bot_id,
        llm_model=llm_model,
        system_prompt=payload.system_prompt,
        prompt_name=None,
        timezone=timezone,
        show_reasoning=True,
        show_tool_calls=True,
        send_datetime=False,
    )
    session.add(chat)
    await session.flush()

    message_store = MessageStore(session)
    imported_count = await save_imported_messages(
        chat_id,
        payload.messages_data,
        message_store,
        reasoning_template=reasoning_template,
    )
    return ImportedChatResult(
        chat_id=chat_id,
        imported_count=imported_count,
        system_prompt=payload.system_prompt,
    )
