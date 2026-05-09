"""Document validation, download, and parsing for the import workflow."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mai_gram.core.import_chat_service import (
    ImportChatConflictError,
    ImportedChatResult,
    ParsedImportPayload,
    create_chat_from_import,
    parse_import_payload,
)
from mai_gram.db.database import get_session
from mai_gram.messenger.base import OutgoingMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from mai_gram.messenger.base import IncomingMessage, Messenger
    from mai_gram.response_templates.base import ResponseTemplate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportDocument:
    """Validated document metadata for an import upload."""

    file_id: str
    file_name: str
    file_size: int


async def validate_import_document(
    message: IncomingMessage,
    messenger: Messenger,
) -> ImportDocument | None:
    file_name = message.document_file_name or ""
    file_size = message.document_file_size or 0
    if not file_name.lower().endswith(".json"):
        await _send(messenger, message.chat_id, "Please upload a .json file.")
        return None
    if file_size > 20 * 1024 * 1024:
        await _send(
            messenger,
            message.chat_id,
            "File is too large (max 20 MB). Please upload a smaller file.",
        )
        return None

    file_id = message.document_file_id
    if not file_id:
        await _send(
            messenger,
            message.chat_id,
            "Could not read the file. Please try again.",
        )
        return None

    return ImportDocument(file_id=file_id, file_name=file_name, file_size=file_size)


async def download_and_parse(
    message: IncomingMessage,
    document: ImportDocument,
    messenger: Messenger,
    clear_session: Callable[[str], None],
) -> ParsedImportPayload | None:
    await _send(
        messenger,
        message.chat_id,
        f"📄 Received: {document.file_name} ({document.file_size:,} bytes)\nParsing...",
    )

    try:
        file_data = await messenger.download_file(document.file_id)
    except (RuntimeError, OSError, asyncio.TimeoutError):
        logger.exception("Failed to download file %s", document.file_id)
        await _send(
            messenger,
            message.chat_id,
            "Failed to download the file from Telegram. Please try again.",
        )
        return None

    from mai_gram.core.importer import ImportDataError as ImportParseError

    try:
        parsed_payload = parse_import_payload(file_data)
    except ImportParseError as exc:
        clear_session(message.user_id)
        await _send(
            messenger,
            message.chat_id,
            f"❌ Import failed: {exc}\n\nUse /import to try again.",
        )
        return None

    if not parsed_payload.messages_data:
        clear_session(message.user_id)
        await _send(
            messenger,
            message.chat_id,
            "❌ The file contains no messages.\n\nUse /import to try again.",
        )
        return None

    return parsed_payload


async def save_imported_chat(
    *,
    message: IncomingMessage,
    chat_id: str,
    llm_model: str,
    default_timezone: str,
    parsed_document: ParsedImportPayload,
    reasoning_template: ResponseTemplate | None,
    reasoning_template_name: str | None,
    template_params_json: str | None,
    messenger: Messenger,
    clear_session: Callable[[str], None],
) -> ImportedChatResult | None:
    async with get_session() as db:
        try:
            saved_chat = await create_chat_from_import(
                db,
                chat_id=chat_id,
                user_id=message.user_id,
                bot_id=message.bot_id or "",
                llm_model=llm_model,
                timezone=default_timezone,
                payload=parsed_document,
                reasoning_template=reasoning_template,
                response_template_name=reasoning_template_name,
                template_params_json=template_params_json,
            )
        except ImportChatConflictError:
            clear_session(message.user_id)
            await _send(
                messenger,
                message.chat_id,
                "A chat was created while you were importing. "
                "Use /reset first, then /import again.",
            )
            return None
        await db.commit()

    return saved_chat


async def _send(messenger: Messenger, chat_id: str, text: str) -> None:
    await messenger.send_message(OutgoingMessage(text=text, chat_id=chat_id))
