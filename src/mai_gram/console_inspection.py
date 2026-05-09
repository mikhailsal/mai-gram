"""Console CLI inspection commands: chat list, history, wiki, prompt preview."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func as sql_func
from sqlalchemy import select

from mai_gram.db import get_session
from mai_gram.db.models import Chat, Message

if TYPE_CHECKING:
    from mai_gram.config import Settings
    from mai_gram.llm.provider import ChatMessage, LLMProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "---- -- --:--:--"
    if value.tzinfo is None:
        dt = value.replace(tzinfo=timezone.utc)
    else:
        dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def print_chat_list() -> None:
    async with get_session() as session:
        query = (
            select(
                Chat.id,
                Chat.llm_model,
                sql_func.count(Message.id).label("message_count"),
                sql_func.max(Message.timestamp).label("last_message"),
            )
            .outerjoin(Message, Chat.id == Message.chat_id)
            .group_by(Chat.id)
            .order_by(sql_func.max(Message.timestamp).desc().nulls_last())
        )
        result = await session.execute(query)
        rows = list(result.all())

    print("=== All Chats ===")
    if not rows:
        print("(no chats found)")
        return

    id_width = max(len("Chat ID"), max(len(row.id) for row in rows))
    model_width = max(len("Model"), max(len(row.llm_model or "") for row in rows))

    print(f"{'Chat ID':<{id_width}}  {'Model':<{model_width}}  {'Messages':>8}  Last Message")
    print("-" * (id_width + model_width + 35))

    for row in rows:
        last_msg = _format_timestamp(row.last_message) if row.last_message else "(no messages)"
        model = row.llm_model or ""
        print(f"{row.id:<{id_width}}  {model:<{model_width}}  {row.message_count:>8}  {last_msg}")


async def print_history(chat_id: str) -> None:
    from mai_gram.core.chat_inspection_service import ChatInspectionService

    async with get_session() as session:
        messages = await ChatInspectionService().list_history(session, chat_id=chat_id)
    print(f"=== History: {chat_id} ===")
    if not messages:
        print("(no messages)")
        return
    for item in messages:
        print(f"[{_format_timestamp(item.timestamp)}] {item.role.upper()}: {item.content}")


async def print_wiki(chat_id: str, data_dir: str) -> None:
    from mai_gram.core.chat_inspection_service import ChatInspectionService

    async with get_session() as session:
        inspection_service = ChatInspectionService(data_dir=data_dir)
        result = await inspection_service.list_wiki(session, chat_id=chat_id)
        if result.sync_report.total_changes > 0:
            print(f"[sync] {result.sync_report.summary()}")
            await session.commit()
    print(f"=== Wiki: {chat_id} ===")
    if not result.entries:
        print("(no wiki entries)")
        return
    for entry in result.entries:
        print(f"- ({int(entry.importance)}) {entry.key}: {entry.value}")


async def repair_wiki(chat_id: str, data_dir: str) -> None:
    from mai_gram.core.chat_inspection_service import ChatInspectionService

    async with get_session() as session:
        inspection_service = ChatInspectionService(data_dir=data_dir)
        report = await inspection_service.repair_wiki(session, chat_id=chat_id)
        await session.commit()
    print(f"=== Wiki Repair: {chat_id} ===")
    if report.total_changes == 0:
        print("Database is already in sync with disk files.")
        return
    print(f"Result: {report.summary()}")
    if report.created:
        for key in report.created:
            print(f"  + created: {key}")
    if report.updated:
        for key in report.updated:
            print(f"  ~ updated: {key}")
    if report.db_rows_deleted:
        for key in report.db_rows_deleted:
            print(f"  - removed orphan DB row: {key}")
    if report.skipped_files:
        for fname in report.skipped_files:
            print(f"  ? skipped unparseable file: {fname}")


async def print_prompt(
    chat_id: str,
    data_dir: str,
    llm: LLMProvider,
    settings: Settings,
    *,
    test_mode: bool = True,
    external_mcp_pool: ExternalMCPPool | None = None,
) -> None:
    from mai_gram.core.prompt_preview_service import PromptPreviewService

    async with get_session() as session:
        preview_service = PromptPreviewService(
            llm,
            settings,
            memory_data_dir=data_dir,
            test_mode=test_mode,
            external_mcp_pool=external_mcp_pool,
        )
        try:
            preview = await preview_service.build_preview(session, chat_id=chat_id)
        except LookupError as exc:
            raise SystemExit(f"Error: no chat found for '{chat_id}'. Run --start first.") from exc

    print("--- Prompt Preview ---")
    print(preview.context[0].content)
    print("")
    print("--- Available Tools ---")
    for tool in preview.tools:
        print(f"- {tool.name}: {tool.description}")
    print("")
    print("--- Message Context ---")
    for msg in preview.context[1:]:
        _print_context_message(msg)
    print("")
    print(f"Approx tokens: {preview.token_count}")


def _print_context_message(msg: ChatMessage) -> None:
    if msg.role.value == "tool":
        print(f"[tool result:{msg.tool_call_id}] {msg.content}")
    else:
        print(f"[{msg.role.value}] {msg.content}")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args_dict = json.loads(tc.arguments)
                args_text = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
            except (json.JSONDecodeError, TypeError):
                args_text = tc.arguments
            print(f"[tool call:{tc.id}] {tc.name}({args_text})")
