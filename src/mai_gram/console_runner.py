"""Console CLI for interacting with mai-gram chats."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mai_gram.config import Settings, get_settings
from mai_gram.console_cli import (
    ConsoleStateStore,
    build_parser,
    needs_live_llm,
    resolve_chat_id,
    resolve_user_id,
)
from mai_gram.console_inspection import print_chat_list, print_prompt
from mai_gram.console_output import print_debug_session_stats
from mai_gram.core.adapter_runtime import (
    build_bot_handler,
    build_external_mcp_pool,
    build_openrouter_provider,
)
from mai_gram.db import close_db, get_session, init_db, run_migrations
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


def _parse_reasoning_template_params(
    raw_params: list[str] | None,
) -> dict[str, str] | None:
    """Parse ``key=value`` pairs from the CLI into a params dict."""
    if not raw_params:
        return None
    result: dict[str, str] = {}
    for item in raw_params:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result or None


def _resolve_reasoning_template(
    template_name: str | None,
    template_params: dict[str, str] | None = None,
) -> ResponseTemplate | None:
    """Resolve a reasoning template by name with optional params."""
    if not template_name:
        return None

    from mai_gram.response_templates.registry import get_template, list_template_names

    available = list_template_names()
    if template_name not in available:
        raise SystemExit(
            f"Error: unknown reasoning template '{template_name}'. "
            f"Available: {', '.join(available)}"
        )
    return get_template(template_name, params=template_params)


async def _import_json_dialogue(
    chat_id: str,
    json_path: str,
    *,
    reasoning_template_name: str | None = None,
    reasoning_template_params: dict[str, str] | None = None,
) -> int:
    """Import a dialogue from a JSON file using the shared importer module."""
    import json

    from mai_gram.core.import_chat_service import import_into_existing_chat, parse_import_payload
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

    reasoning_template = _resolve_reasoning_template(
        reasoning_template_name, reasoning_template_params
    )

    template_params_json: str | None = None
    if reasoning_template_params:
        template_params_json = json.dumps(reasoning_template_params, ensure_ascii=False)

    async with get_session() as session:
        try:
            imported = await import_into_existing_chat(
                session,
                chat_id=chat_id,
                payload=payload,
                reasoning_template=reasoning_template,
                response_template_name=reasoning_template_name,
                template_params_json=template_params_json,
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
            await print_chat_list()
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
            await print_prompt(
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
    from mai_gram.console_inspection import print_history, print_wiki, repair_wiki

    if args.history:
        await print_history(chat_id)
        return True
    if args.repair_wiki:
        await repair_wiki(chat_id, settings.memory_data_dir)
        return True
    if args.wiki:
        await print_wiki(chat_id, settings.memory_data_dir)
        return True
    if args.import_json:
        raw_params = getattr(args, "reasoning_template_params", None)
        parsed_params = _parse_reasoning_template_params(raw_params)
        count = await _import_json_dialogue(
            chat_id,
            args.import_json,
            reasoning_template_name=getattr(args, "reasoning_template", None),
            reasoning_template_params=parsed_params,
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

    if args.start:
        await _dispatch_start_flow(messenger, args, chat_id, user_id)
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
    if args.start:
        await _dispatch_start_template(messenger, args, chat_id, user_id)

    if not (args.start or args.command or args.callbacks or args.message):
        raise SystemExit(
            "Error: nothing to do. Provide a message, --start, --command, --cb, "
            "or an inspection flag."
        )

    messenger.flush_edits()
    return messenger


async def _dispatch_start_flow(
    messenger: ConsoleMessenger,
    args: Any,
    chat_id: str,
    user_id: str,
) -> None:
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


async def _dispatch_start_template(
    messenger: ConsoleMessenger,
    args: Any,
    chat_id: str,
    user_id: str,
) -> None:
    template = getattr(args, "template", None) or "empty"
    await messenger.dispatch_callback(
        chat_id=chat_id,
        user_id=user_id,
        callback_data=f"tpl_group:__single__:{template}",
    )
    raw_tpl_params = getattr(args, "template_params", None)
    if raw_tpl_params:
        kv_lines = "\n".join(raw_tpl_params)
        await messenger.dispatch_text(chat_id=chat_id, user_id=user_id, text=kv_lines)
    else:
        from mai_gram.response_templates.registry import get_template as _get_tpl

        tpl_obj = _get_tpl(template if template != "empty" else None)
        if tpl_obj.get_params():
            await messenger.dispatch_callback(
                chat_id=chat_id,
                user_id=user_id,
                callback_data="tpl_params:__defaults__",
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(_run(args))
