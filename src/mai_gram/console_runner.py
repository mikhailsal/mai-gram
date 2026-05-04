"""Console CLI for interacting with mai-gram chats."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import func as sql_func
from sqlalchemy import select

from mai_gram.config import Settings, get_settings
from mai_gram.console_cli import (
    ConsoleStateStore,
    build_parser,
    needs_live_llm,
    resolve_chat_id,
    resolve_user_id,
)
from mai_gram.console_output import print_debug_session_stats
from mai_gram.core.adapter_runtime import (
    build_bot_handler,
    build_external_mcp_pool,
    build_openrouter_provider,
)
from mai_gram.core.chat_inspection_service import ChatInspectionService
from mai_gram.core.import_chat_service import import_into_existing_chat, parse_import_payload
from mai_gram.core.prompt_preview_service import PromptPreviewService
from mai_gram.db import close_db, get_session, init_db, run_migrations
from mai_gram.db.models import Chat, Message
from mai_gram.debug import LLMLoggerProvider
from mai_gram.llm.provider import (
    ChatMessage,
    LLMAuthenticationError,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolDefinition,
)
from mai_gram.messenger.base import IncomingMessage, MessageType
from mai_gram.messenger.console import ConsoleMessenger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mai_gram.llm.openrouter import OpenRouterProvider
    from mai_gram.mcp_servers.external import ExternalMCPPool
    from mai_gram.response_templates.base import ResponseTemplate

logger = logging.getLogger(__name__)


class _OfflineCLIProvider(LLMProvider):
    """Minimal provider for command-only CLI flows that do not hit the network."""

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del messages, model, temperature, max_tokens, tools, tool_choice, extra_params
        raise LLMAuthenticationError("OpenRouter API key must not be empty")

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> Any:
        del messages, model, temperature, max_tokens, tools, tool_choice, extra_params
        raise LLMAuthenticationError("OpenRouter API key must not be empty")
        yield StreamChunk(content="")

    async def count_tokens(self, messages: list[ChatMessage], *, model: str | None = None) -> int:
        del model
        total_chars = 0
        for msg in messages:
            total_chars += 16
            total_chars += len(msg.content)
        return total_chars // 4

    async def close(self) -> None:
        return None


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "---- -- --:--:--"
    if value.tzinfo is None:
        dt = value.replace(tzinfo=timezone.utc)
    else:
        dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_command_text(raw_command: str) -> tuple[str, str | None]:
    command_text = raw_command.strip()
    if not command_text:
        raise SystemExit("Error: --command requires a command name.")

    if command_text.startswith("/"):
        command_text = command_text[1:]

    command, _, remainder = command_text.partition(" ")
    command = command.strip()
    command_args = remainder.strip() or None
    if not command:
        raise SystemExit("Error: --command requires a command name.")
    return command, command_args


# -- Display commands --


async def _print_chat_list() -> None:
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


async def _print_history(chat_id: str) -> None:
    async with get_session() as session:
        messages = await ChatInspectionService().list_history(session, chat_id=chat_id)
    print(f"=== History: {chat_id} ===")
    if not messages:
        print("(no messages)")
        return
    for item in messages:
        print(f"[{_format_timestamp(item.timestamp)}] {item.role.upper()}: {item.content}")


async def _print_wiki(chat_id: str, data_dir: str) -> None:
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


async def _repair_wiki(chat_id: str, data_dir: str) -> None:
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


async def _print_prompt(
    chat_id: str,
    data_dir: str,
    llm: LLMProvider,
    settings: Settings,
    *,
    test_mode: bool = True,
    external_mcp_pool: ExternalMCPPool | None = None,
) -> None:
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
    print("")
    print(f"Approx tokens: {preview.token_count}")


# -- Import command --


def _resolve_reasoning_template(
    template_name: str | None,
) -> ResponseTemplate | None:
    """Resolve a reasoning template by name, or return ``None``."""
    if not template_name:
        return None

    from mai_gram.response_templates.registry import get_template, list_template_names

    available = list_template_names()
    if template_name not in available:
        raise SystemExit(
            f"Error: unknown reasoning template '{template_name}'. "
            f"Available: {', '.join(available)}"
        )
    return get_template(template_name)


async def _import_json_dialogue(
    chat_id: str,
    json_path: str,
    *,
    reasoning_template_name: str | None = None,
) -> int:
    """Import a dialogue from a JSON file using the shared importer module."""
    from mai_gram.core.importer import ImportDataError as ImportParseError

    path = Path(json_path)
    if not path.exists():
        raise SystemExit(f"Error: file not found: {json_path}")

    try:
        file_data = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Error: cannot read file: {exc}") from exc

    try:
        payload = parse_import_payload(file_data)
    except ImportParseError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    reasoning_template = _resolve_reasoning_template(reasoning_template_name)

    async with get_session() as session:
        try:
            imported = await import_into_existing_chat(
                session,
                chat_id=chat_id,
                payload=payload,
                reasoning_template=reasoning_template,
            )
        except LookupError as exc:
            raise SystemExit(f"Error: no chat found for '{chat_id}'. Run --start first.") from exc
        if imported.imported_count == 0:
            raise SystemExit("Error: no messages could be imported from the JSON file.")
        await session.commit()

    return imported.imported_count


# -- Main --


def _incoming_command(
    chat_id: str,
    user_id: str,
    command: str,
    command_args: str | None = None,
) -> IncomingMessage:
    now = datetime.now(timezone.utc)
    text = f"/{command}"
    if command_args:
        text = f"{text} {command_args}"
    return IncomingMessage(
        platform="console",
        chat_id=chat_id,
        user_id=user_id,
        message_id=f"cmd-{int(now.timestamp())}",
        message_type=MessageType.COMMAND,
        text=text,
        command=command,
        command_args=command_args,
        timestamp=now,
    )


async def _run(args: Any) -> None:
    settings = get_settings()
    state_store = ConsoleStateStore()

    if args.list:
        engine = await init_db(settings.database_url, echo=settings.debug)
        await run_migrations(engine)
        try:
            await _print_chat_list()
        finally:
            await close_db()
        return

    user_id = resolve_user_id(args, settings)
    chat_id = resolve_chat_id(args, state_store)

    engine = await init_db(settings.database_url, echo=settings.debug)
    await run_migrations(engine)

    llm: LLMProvider | None = None
    external_mcp_pool: ExternalMCPPool | None = None
    try:
        if await _handle_console_inspection(args, chat_id, settings):
            return

        external_mcp_pool = build_external_mcp_pool(settings)
        llm, logger_provider = _build_cli_llm(args, chat_id, settings)

        if args.show_prompt:
            test_mode = not args.real
            await _print_prompt(
                chat_id,
                settings.memory_data_dir,
                llm,
                settings,
                test_mode=test_mode,
                external_mcp_pool=external_mcp_pool,
            )
            return

        await _dispatch_console_runtime(
            args,
            chat_id=chat_id,
            user_id=user_id,
            llm=llm,
            settings=settings,
            external_mcp_pool=external_mcp_pool,
        )

        if logger_provider is not None:
            print_debug_session_stats(logger_provider.get_session_stats())
    finally:
        if external_mcp_pool is not None:
            await external_mcp_pool.stop_all()
        if llm is not None:
            await llm.close()
        await close_db()


async def _handle_console_inspection(
    args: Any,
    chat_id: str,
    settings: Settings,
) -> bool:
    if args.history:
        await _print_history(chat_id)
        return True
    if args.repair_wiki:
        await _repair_wiki(chat_id, settings.memory_data_dir)
        return True
    if args.wiki:
        await _print_wiki(chat_id, settings.memory_data_dir)
        return True
    if args.import_json:
        count = await _import_json_dialogue(
            chat_id,
            args.import_json,
            reasoning_template_name=getattr(args, "reasoning_template", None),
        )
        tmpl_name = getattr(args, "reasoning_template", None)
        tmpl_info = f" (reasoning template: {tmpl_name})" if tmpl_name else ""
        print(f"Imported {count} messages into chat '{chat_id}'.{tmpl_info}")
        return True
    return False


def _build_cli_llm(
    args: Any,
    chat_id: str,
    settings: Settings,
) -> tuple[LLMProvider, LLMLoggerProvider | None]:
    if needs_live_llm(args) and not settings.openrouter_api_key:
        raise SystemExit("Error: OPENROUTER_API_KEY is required.")

    if settings.openrouter_api_key:
        llm_base: OpenRouterProvider | _OfflineCLIProvider = build_openrouter_provider(settings)
    else:
        llm_base = _OfflineCLIProvider()

    if not args.debug:
        return llm_base, None

    logger_provider = LLMLoggerProvider(
        llm_base,
        chat_id=chat_id,
        base_dir=Path(settings.memory_data_dir) / "debug_logs",
    )
    return logger_provider, logger_provider


async def _dispatch_console_runtime(
    args: Any,
    *,
    chat_id: str,
    user_id: str,
    llm: LLMProvider,
    settings: Settings,
    external_mcp_pool: ExternalMCPPool | None = None,
) -> ConsoleMessenger:
    messenger = ConsoleMessenger(stream_debug=args.stream_debug)
    build_bot_handler(
        messenger,
        llm,
        settings,
        test_mode=not args.real,
        external_mcp_pool=external_mcp_pool,
    )

    is_start = bool(args.start)
    if is_start:
        await messenger.dispatch_message(_incoming_command(chat_id, user_id, "start"))
        if args.model:
            await messenger.dispatch_callback(
                chat_id=chat_id,
                user_id=user_id,
                callback_data=f"model:{args.model}",
            )
        if args.prompt:
            await messenger.dispatch_callback(
                chat_id=chat_id,
                user_id=user_id,
                callback_data=f"prompt:{args.prompt}",
            )
    if args.command:
        command, command_args = _parse_command_text(args.command)
        await messenger.dispatch_message(_incoming_command(chat_id, user_id, command, command_args))
    if args.callbacks:
        for cb_data in args.callbacks:
            await messenger.dispatch_callback(
                chat_id=chat_id,
                user_id=user_id,
                callback_data=cb_data,
            )
    if args.message:
        await messenger.dispatch_text(
            chat_id=chat_id,
            user_id=user_id,
            text=args.message,
        )
    if is_start:
        template = getattr(args, "template", None) or "empty"
        await messenger.dispatch_callback(
            chat_id=chat_id,
            user_id=user_id,
            callback_data=f"template:{template}",
        )
        raw_tpl_params = getattr(args, "template_params", None)
        if raw_tpl_params:
            kv_lines = "\n".join(raw_tpl_params)
            await messenger.dispatch_text(
                chat_id=chat_id,
                user_id=user_id,
                text=kv_lines,
            )
        else:
            from mai_gram.response_templates.registry import get_template as _get_tpl

            tpl_obj = _get_tpl(template if template != "empty" else None)
            if tpl_obj.get_params():
                await messenger.dispatch_callback(
                    chat_id=chat_id,
                    user_id=user_id,
                    callback_data="tpl_params:__defaults__",
                )

    if not (args.start or args.command or args.callbacks or args.message):
        raise SystemExit(
            "Error: nothing to do. Provide a message, --start, --command, --cb, "
            "or an inspection flag."
        )

    messenger.flush_edits()
    return messenger


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(_run(args))
