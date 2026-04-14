"""Console CLI for interacting with mai-gram chats."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import func as sql_func, select

from mai_gram.bot.handler import BotHandler
from mai_gram.config import get_settings
from mai_gram.core.prompt_builder import PromptBuilder
from mai_gram.db import close_db, get_session, init_db, run_migrations
from mai_gram.db.models import Chat, Message
from mai_gram.debug import LLMLoggerProvider
from mai_gram.llm.openrouter import OpenRouterProvider
from mai_gram.llm.provider import LLMProvider
from mai_gram.memory.knowledge_base import WikiStore
from mai_gram.memory.messages import MessageStore
from mai_gram.messenger.base import IncomingMessage, MessageType
from mai_gram.messenger.console import ConsoleMessenger

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mai-chat",
        description="Console interface for mai-gram.",
    )
    parser.add_argument("message", nargs="?", help="Message text to send.")
    parser.add_argument("-c", "--chat-id", help="Chat ID (persisted for future runs).")
    parser.add_argument("--user-id", help="User ID for synthetic events.")
    parser.add_argument(
        "--cb",
        action="append",
        dest="callbacks",
        metavar="DATA",
        help="Dispatch a callback payload (button press). Can be repeated.",
    )
    parser.add_argument("--start", action="store_true", help="Dispatch /start command.")
    parser.add_argument("--history", action="store_true", help="Show conversation history.")
    parser.add_argument("--wiki", action="store_true", help="Show wiki entries.")
    parser.add_argument("--show-prompt", action="store_true", help="Print assembled LLM prompt.")
    parser.add_argument("--debug", action="store_true", help="Enable LLM debug logging.")
    parser.add_argument("--list", action="store_true", help="List all chats with message counts.")
    parser.add_argument(
        "--import-json",
        dest="import_json",
        metavar="PATH",
        help=(
            "Import a dialogue from a JSON file. "
            "Expected format: [{role, content, tool_calls?, reasoning?, timestamp?}, ...]"
        ),
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Real conversation mode (disables test mode transparency notice).",
    )
    return parser


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "---- -- --:--:--"
    if value.tzinfo is None:
        dt = value.replace(tzinfo=timezone.utc)
    else:
        dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class _ConsoleStateStore:
    """Persists console state (last chat ID) to a JSON file."""

    _STATE_FILE = Path("./data/.console_state.json")

    def load(self) -> dict:
        if self._STATE_FILE.exists():
            return json.loads(self._STATE_FILE.read_text(encoding="utf-8"))
        return {}

    def save(self, state: dict) -> None:
        self._STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_last_chat_id(self) -> str | None:
        return self.load().get("last_chat_id")

    def set_last_chat_id(self, chat_id: str) -> None:
        state = self.load()
        state["last_chat_id"] = chat_id
        self.save(state)


def _resolve_chat_id(args: argparse.Namespace, state_store: _ConsoleStateStore) -> str:
    chat_id = args.chat_id or state_store.get_last_chat_id()
    if not chat_id:
        raise SystemExit(
            "Error: no chat ID available. Use -c/--chat-id once, then it will be remembered."
        )
    state_store.set_last_chat_id(chat_id)
    return chat_id


def _resolve_user_id(args: argparse.Namespace, settings: object) -> str:
    if args.user_id:
        return args.user_id
    allowed = settings.get_allowed_user_ids()  # type: ignore[union-attr]
    if allowed:
        return sorted(allowed)[0]
    return "console-user"


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
        result = await session.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        messages = list(result.scalars().all())
    print(f"=== History: {chat_id} ===")
    if not messages:
        print("(no messages)")
        return
    for item in messages:
        print(f"[{_format_timestamp(item.timestamp)}] {item.role.upper()}: {item.content}")


async def _print_wiki(chat_id: str, data_dir: str) -> None:
    async with get_session() as session:
        wiki_store = WikiStore(session, data_dir=data_dir)
        entries = await wiki_store.list_entries(chat_id)
    print(f"=== Wiki: {chat_id} ===")
    if not entries:
        print("(no wiki entries)")
        return
    for entry in entries:
        print(f"- ({int(entry.importance)}) {entry.key}: {entry.value}")


async def _print_prompt(
    chat_id: str,
    data_dir: str,
    llm: LLMProvider,
    *,
    test_mode: bool = True,
) -> None:
    from mai_gram.mcp_servers.manager import MCPManager
    from mai_gram.mcp_servers.wiki_server import WikiMCPServer
    from mai_gram.mcp_servers.messages_server import MessagesMCPServer

    async with get_session() as session:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if chat is None:
            raise SystemExit(f"Error: no chat found for '{chat_id}'. Run --start first.")

        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=data_dir)

        prompt_builder = PromptBuilder(
            llm, message_store, wiki_store, test_mode=test_mode,
        )
        context = await prompt_builder.build_context(chat)

        mcp_manager = MCPManager()
        mcp_manager.register_server("wiki", WikiMCPServer(wiki_store, chat_id))
        mcp_manager.register_server("messages", MessagesMCPServer(message_store, chat_id))
        tools = await mcp_manager.list_all_tools()

    print("--- Prompt Preview ---")
    print(context[0].content)
    print("")
    print("--- Available Tools ---")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    print("")
    print("--- Message Context ---")
    for msg in context[1:]:
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
    token_count = await llm.count_tokens(context)
    print("")
    print(f"Approx tokens: {token_count}")


# -- Import command --

async def _import_json_dialogue(chat_id: str, json_path: str) -> int:
    """Import a dialogue from a JSON file in OpenAI chat completion format.

    Expected format: a JSON array of message objects:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "...", "timestamp": "2024-01-15T14:30:00Z"},
        {"role": "assistant", "content": "...", "reasoning": "...", "tool_calls": [...]},
        {"role": "tool", "content": "...", "tool_call_id": "..."},
        ...
    ]

    Reasoning content is prepended to the message content in a [reasoning] block.
    """
    path = Path(json_path)
    if not path.exists():
        raise SystemExit(f"Error: file not found: {json_path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit("Error: expected a JSON array of message objects.")

    async with get_session() as session:
        result = await session.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        if chat is None:
            raise SystemExit(f"Error: no chat found for '{chat_id}'. Run --start first.")

        message_store = MessageStore(session)
        imported = 0

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise SystemExit(f"Error: entry {i} is not a JSON object.")

            role = entry.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise SystemExit(f"Error: entry {i} has invalid role '{role}'.")

            content = entry.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            reasoning = entry.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                content = f"[reasoning]\n{reasoning.strip()}\n[/reasoning]\n\n{content}"

            tool_calls_raw = entry.get("tool_calls")
            tool_calls_json: str | None = None
            if isinstance(tool_calls_raw, list) and tool_calls_raw:
                tc_list = []
                for tc in tool_calls_raw:
                    if isinstance(tc, dict):
                        func = tc.get("function", tc)
                        tc_list.append({
                            "id": tc.get("id", f"import_{i}"),
                            "name": func.get("name", "unknown"),
                            "arguments": func.get("arguments", "{}"),
                        })
                if tc_list:
                    tool_calls_json = json.dumps(tc_list)

            tool_call_id = entry.get("tool_call_id")
            if tool_call_id is not None and not isinstance(tool_call_id, str):
                tool_call_id = str(tool_call_id)

            timestamp = None
            ts_raw = entry.get("timestamp")
            if isinstance(ts_raw, str):
                try:
                    timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    pass

            if role == "system":
                continue

            try:
                await message_store.save_message(
                    chat_id=chat_id,
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    tool_calls=tool_calls_json,
                    tool_call_id=tool_call_id,
                )
            except ValueError as exc:
                logger.warning("Skipping entry %d due to timestamp conflict: %s", i, exc)
                continue

            imported += 1

    return imported


# -- Main --

def _incoming_command(chat_id: str, user_id: str, command: str) -> IncomingMessage:
    now = datetime.now(timezone.utc)
    return IncomingMessage(
        platform="console",
        chat_id=chat_id,
        user_id=user_id,
        message_id=f"cmd-{int(now.timestamp())}",
        message_type=MessageType.COMMAND,
        text=f"/{command}",
        command=command,
        timestamp=now,
    )


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    state_store = _ConsoleStateStore()

    if args.list:
        engine = await init_db(settings.database_url, echo=settings.debug)
        await run_migrations(engine)
        try:
            await _print_chat_list()
        finally:
            await close_db()
        return

    user_id = _resolve_user_id(args, settings)
    chat_id = _resolve_chat_id(args, state_store)

    engine = await init_db(settings.database_url, echo=settings.debug)
    await run_migrations(engine)

    llm_base: OpenRouterProvider | None = None
    llm: LLMProvider | None = None
    try:
        if args.history:
            await _print_history(chat_id)
            return
        if args.wiki:
            await _print_wiki(chat_id, settings.memory_data_dir)
            return
        if args.import_json:
            count = await _import_json_dialogue(chat_id, args.import_json)
            print(f"Imported {count} messages into chat '{chat_id}'.")
            return

        if not settings.openrouter_api_key:
            raise SystemExit("Error: OPENROUTER_API_KEY is required.")

        llm_base = OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            default_model=settings.llm_model,
            base_url=settings.openrouter_base_url,
        )
        llm = llm_base
        logger_provider: LLMLoggerProvider | None = None
        if args.debug:
            logger_provider = LLMLoggerProvider(
                llm_base,
                chat_id=chat_id,
                base_dir=Path(settings.memory_data_dir) / "debug_logs",
            )
            llm = logger_provider

        if args.show_prompt:
            test_mode = not args.real
            await _print_prompt(chat_id, settings.memory_data_dir, llm, test_mode=test_mode)
            return

        messenger = ConsoleMessenger()
        test_mode = not args.real
        _handler = BotHandler(
            messenger,
            llm,
            memory_data_dir=settings.memory_data_dir,
            wiki_context_limit=settings.wiki_context_limit,
            short_term_limit=settings.short_term_limit,
            tool_max_iterations=settings.tool_max_iterations,
            test_mode=test_mode,
        )

        if args.start:
            await messenger.dispatch_message(_incoming_command(chat_id, user_id, "start"))
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

        if not (args.start or args.callbacks or args.message):
            raise SystemExit(
                "Error: nothing to do. Provide a message, --start, --cb, or an inspection flag."
            )

        if logger_provider is not None:
            stats = logger_provider.get_session_stats()
            print("")
            print("--- Debug Info ---")
            print(
                f"LLM calls: {stats['llm_calls']} "
                f"({stats['calls_with_tool_calls']} with tool calls)"
            )
            tools_used = ", ".join(stats["tools_used"]) if stats["tools_used"] else "none"
            print(f"Tools used: {tools_used}")
            print(
                f"Tokens: {stats['prompt_tokens']:,} prompt + "
                f"{stats['completion_tokens']:,} completion = "
                f"{stats['total_tokens']:,} total"
            )
            print("")
            print("--- Session Cost ---")
            print(
                f"This call: {stats['last_call_total_tokens']:,} tokens "
                f"(${stats['last_call_cost_usd']:.3f})"
            )
            print(
                f"Session total: {stats['total_tokens']:,} tokens "
                f"(${stats['session_cost_usd']:.3f})"
            )
            if stats["log_path"]:
                print(f"Full log: {stats['log_path']}")
    finally:
        if llm is not None:
            await llm.close()
        await close_db()


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(_run(args))
