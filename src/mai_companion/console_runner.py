"""Single-shot console CLI for interacting with a companion chat."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import func as sql_func, select

from mai_companion.bot.handler import BotHandler
from mai_companion.bot.onboarding import OnboardingSession, OnboardingState
from mai_companion.clock import Clock, ConsoleStateStore
from mai_companion.config import get_settings
from mai_companion.core.prompt_builder import PromptBuilder
from mai_companion.db import close_db, get_session, init_db, run_migrations
from mai_companion.db.models import Companion, Message
from mai_companion.debug import LLMLoggerProvider
from mai_companion.llm.openrouter import OpenRouterProvider
from mai_companion.llm.provider import LLMProvider
from mai_companion.memory.knowledge_base import WikiStore
from mai_companion.memory.messages import MessageStore
from mai_companion.memory.summaries import SummaryStore
from mai_companion.messenger.base import IncomingMessage, MessageType
from mai_companion.messenger.console import ConsoleMessenger
from mai_companion.personality.mood import MoodManager
from mai_companion.personality.character import CommunicationStyle, Gender, Verbosity

logger = logging.getLogger(__name__)


def _parse_target_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def _parse_week_period(value: str) -> str:
    """Validate and normalize week period format (YYYY-Www)."""
    import re
    match = re.match(r"^(\d{4})-W(\d{1,2})$", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid week period '{value}'. Expected YYYY-Www format (e.g., 2024-W03)."
        )
    year = int(match.group(1))
    week = int(match.group(2))
    if week < 1 or week > 53:
        raise argparse.ArgumentTypeError(
            f"Invalid week number {week}. Must be 1-53."
        )
    return f"{year}-W{week:02d}"


def _parse_month_period(value: str) -> str:
    """Validate and normalize month period format (YYYY-MM)."""
    import re
    match = re.match(r"^(\d{4})-(\d{1,2})$", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid month period '{value}'. Expected YYYY-MM format (e.g., 2024-01)."
        )
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        raise argparse.ArgumentTypeError(
            f"Invalid month number {month}. Must be 1-12."
        )
    return f"{year}-{month:02d}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mai-chat",
        description="Single-shot console interface for mAI Companion.",
    )
    parser.add_argument("message", nargs="?", help="Human message text to send.")
    parser.add_argument("-c", "--chat-id", help="Chat ID (persisted as default for future runs).")
    parser.add_argument("--user-id", help="User ID for synthetic events.")
    parser.add_argument("--cb", help="Dispatch a callback payload (button press).")
    parser.add_argument("--start", action="store_true", help="Dispatch /start onboarding command.")
    parser.add_argument("--date", type=_parse_target_date, help="Virtual target date (YYYY-MM-DD).")
    parser.add_argument("--history", action="store_true", help="Show conversation history.")
    parser.add_argument("--replay", action="store_true", help="Replay conversation by day with tool calls (wiki always, others need --debug).")
    parser.add_argument("--wiki", action="store_true", help="Show wiki entries.")
    parser.add_argument("--summaries", action="store_true", help="Show stored summaries.")
    parser.add_argument("--show-prompt", action="store_true", help="Print assembled LLM prompt.")
    parser.add_argument(
        "--seed",
        help='Seed messages from JSONL ({"role":"user","content":"...","date":"YYYY-MM-DD"}).',
    )
    parser.add_argument("--debug", action="store_true", help="Enable structured LLM debug logging.")
    parser.add_argument("--list", action="store_true", help="List all chat IDs with message counts.")
    
    # Consolidation commands
    parser.add_argument(
        "--list-consolidations",
        action="store_true",
        help="List all consolidations with version history.",
    )
    parser.add_argument(
        "--reconsolidate",
        choices=["daily", "weekly", "monthly"],
        help="Re-consolidate summaries of specified type.",
    )
    parser.add_argument(
        "--from",
        dest="from_period",
        help="Starting period for reconsolidation (YYYY-MM-DD for daily, YYYY-Www for weekly, YYYY-MM for monthly).",
    )
    parser.add_argument(
        "--until",
        dest="until_period",
        help="Ending period for reconsolidation (inclusive). Defaults to today/current period.",
    )
    return parser


def _resolve_chat_id(args: argparse.Namespace, state_store: ConsoleStateStore) -> str:
    chat_id = args.chat_id or state_store.get_last_chat_id()
    if not chat_id:
        raise SystemExit(
            "Error: no chat ID available. Use -c/--chat-id once, then it will be remembered."
        )
    state_store.set_last_chat_id(chat_id)
    return chat_id


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "---- -- --:--:--"
    if value.tzinfo is None:
        dt = value.replace(tzinfo=timezone.utc)
    else:
        dt = value.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_time(value: datetime | None) -> str:
    if value is None:
        return "--:--:--"
    if value.tzinfo is None:
        dt = value.replace(tzinfo=timezone.utc)
    else:
        dt = value.astimezone(timezone.utc)
    return dt.strftime("%H:%M:%S")


async def _print_chat_list() -> None:
    """List all chat IDs with message counts and last message dates."""
    async with get_session() as session:
        # Query companions with message stats
        query = (
            select(
                Companion.id,
                Companion.name,
                sql_func.count(Message.id).label("message_count"),
                sql_func.max(Message.timestamp).label("last_message"),
            )
            .outerjoin(Message, Companion.id == Message.companion_id)
            .group_by(Companion.id)
            .order_by(sql_func.max(Message.timestamp).desc().nulls_last())
        )
        result = await session.execute(query)
        rows = list(result.all())

    print("=== All Chats ===")
    if not rows:
        print("(no chats found)")
        return

    # Calculate column widths
    id_width = max(len("Chat ID"), max(len(row.id) for row in rows))
    name_width = max(len("Name"), max(len(row.name or "") for row in rows))

    # Print header
    print(f"{'Chat ID':<{id_width}}  {'Name':<{name_width}}  {'Messages':>8}  Last Message")
    print("-" * (id_width + name_width + 35))

    for row in rows:
        last_msg = _format_timestamp(row.last_message) if row.last_message else "(no messages)"
        name = row.name or ""
        print(f"{row.id:<{id_width}}  {name:<{name_width}}  {row.message_count:>8}  {last_msg}")


async def _print_history(chat_id: str) -> None:
    async with get_session() as session:
        result = await session.execute(
            select(Message)
            .where(Message.companion_id == chat_id)
            .order_by(Message.timestamp.asc(), Message.id.asc())
        )
        messages = list(result.scalars().all())
    print(f"=== History: {chat_id} ===")
    if not messages:
        print("(no messages)")
        return
    for item in messages:
        print(f"[{_format_timestamp(item.timestamp)}] {item.role.upper()}: {item.content}")


def _load_tool_events(
    chat_id: str,
    *,
    data_dir: str,
    target_date: date | None = None,
) -> dict[date, list[tuple[datetime, str]]]:
    events: dict[date, list[tuple[datetime, str]]] = defaultdict(list)

    # Load from wiki changelog (always available, written by WikiMCPServer)
    wiki_changelog = Path(data_dir) / chat_id / "wiki" / "changelog.jsonl"
    has_wiki_changelog = wiki_changelog.exists()
    if has_wiki_changelog:
        for raw in wiki_changelog.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp_raw = payload.get("timestamp")
            if not isinstance(timestamp_raw, str):
                continue
            try:
                timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            timestamp = timestamp.astimezone(timezone.utc)
            event_date = timestamp.date()
            if target_date is not None and event_date != target_date:
                continue

            action = payload.get("action", "unknown")
            key = payload.get("key", "")
            # Map changelog actions to tool names
            tool_name = f"wiki_{action}"
            args_parts = [f"key={key!r}"]
            if "content" in payload:
                content = payload["content"]
                # Truncate long content for display
                if len(content) > 50:
                    content = content[:47] + "..."
                args_parts.append(f"content={content!r}")
            if "importance" in payload:
                args_parts.append(f"importance={payload['importance']}")
            arg_text = ", ".join(args_parts)
            display = f"[tool] {tool_name}({arg_text})"
            events[event_date].append((timestamp, display))

    # Load from debug logs (only available when --debug was used)
    debug_dir = Path(data_dir) / "debug_logs" / chat_id
    if debug_dir.exists():
        if target_date is not None:
            candidate_paths = [debug_dir / f"{target_date.isoformat()}.jsonl"]
        else:
            candidate_paths = sorted(debug_dir.glob("*.jsonl"))

        for path in candidate_paths:
            if not path.exists():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("entry_type") != "tool_result":
                    continue

                timestamp_raw = payload.get("timestamp")
                if not isinstance(timestamp_raw, str):
                    continue
                try:
                    timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                timestamp = timestamp.astimezone(timezone.utc)
                event_date = timestamp.date()
                if target_date is not None and event_date != target_date:
                    continue

                tool_name = str(payload.get("tool_name", "unknown_tool"))
                # Skip wiki tools if already loaded from changelog to avoid duplicates
                if has_wiki_changelog and tool_name.startswith("wiki_"):
                    continue

                arguments = payload.get("arguments")
                if isinstance(arguments, str):
                    arg_text = arguments
                elif isinstance(arguments, dict):
                    arg_text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
                else:
                    arg_text = ""

                display = f"[tool] {tool_name}({arg_text})" if arg_text else f"[tool] {tool_name}()"
                events[event_date].append((timestamp, display))

    return events


async def _print_replay(chat_id: str, *, data_dir: str, target_date: date | None = None) -> None:
    async with get_session() as session:
        query = select(Message).where(Message.companion_id == chat_id)
        if target_date is not None:
            start_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)
            query = query.where(Message.timestamp >= start_dt, Message.timestamp <= end_dt)
        result = await session.execute(query.order_by(Message.timestamp.asc(), Message.id.asc()))
        messages = list(result.scalars().all())

    print(f"=== Conversation Replay: {chat_id} ===")
    if target_date is not None:
        print(f"=== Date Filter: {target_date.isoformat()} ===")

    if not messages:
        print("(no messages)")
        return

    tool_events = _load_tool_events(chat_id, data_dir=data_dir, target_date=target_date)
    messages_by_day: dict[date | None, list[Message]] = defaultdict(list)
    for item in messages:
        timestamp = item.timestamp
        if timestamp is None:
            day: date | None = None
        elif timestamp.tzinfo is None:
            day = timestamp.replace(tzinfo=timezone.utc).date()
        else:
            day = timestamp.astimezone(timezone.utc).date()
        messages_by_day[day].append(item)

    day_keys = sorted([d for d in messages_by_day if d is not None])
    if None in messages_by_day:
        day_keys.append(None)  # type: ignore[arg-type]

    for day in day_keys:
        print("")
        if day is None:
            print("=== Date: unknown ===")
            print("")
            for item in messages_by_day[day]:
                print(f"[{_format_time(item.timestamp)}] {item.role.upper()}: {item.content}")
            continue

        print(f"=== Date: {day.isoformat()} ===")
        print("")
        timeline: list[tuple[datetime, str, str]] = []
        for item in messages_by_day[day]:
            ts = item.timestamp
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)

            # Format the message line
            content = item.content.rstrip() if item.content else ""
            if item.role == "tool":
                # Tool result message
                tool_id = item.tool_call_id or "unknown"
                content_preview = content[:50] + "..." if len(content) > 50 else content
                timeline.append((ts, "tool_result", f"[tool result:{tool_id}] {content_preview}"))
            else:
                timeline.append((ts, "message", f"[{_format_time(ts)}] {item.role.upper()}: {content}"))

                # If this assistant message has tool calls, show them
                if item.role == "assistant" and item.tool_calls:
                    try:
                        tool_calls_data = json.loads(item.tool_calls)
                        for tc in tool_calls_data:
                            tc_name = tc.get("name", "unknown")
                            tc_args = tc.get("arguments", "")
                            # Parse arguments if it's a JSON string
                            if isinstance(tc_args, str) and tc_args.strip():
                                try:
                                    args_dict = json.loads(tc_args)
                                    args_text = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
                                except json.JSONDecodeError:
                                    args_text = tc_args
                            else:
                                args_text = ""
                            tc_line = f"[tool] {tc_name}({args_text})" if args_text else f"[tool] {tc_name}()"
                            timeline.append((ts, "tool_call", tc_line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed tool_calls JSON

        # Add tool events from changelog/debug logs ONLY if no tool calls in DB
        # This provides backwards compatibility for old data
        has_db_tool_calls = any(
            (item.tool_calls or item.role == "tool") for item in messages_by_day[day]
        )
        if not has_db_tool_calls:
            for tool_ts, tool_line in tool_events.get(day, []):
                timeline.append((tool_ts, "tool", tool_line))

        timeline.sort(key=lambda event: (event[0], 0 if event[1] == "message" else 1))
        for _, _, line in timeline:
            print(line)


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


def _print_summaries(chat_id: str, data_dir: str) -> None:
    summary_store = SummaryStore(data_dir=data_dir)
    summaries = summary_store.get_all_summaries(chat_id)
    print(f"=== Summaries: {chat_id} ===")
    if not summaries:
        print("(no summaries)")
        return
    for item in summaries:
        print(f"- [{item.summary_type}:{item.period}] {item.content.strip()}")


def _print_consolidations(chat_id: str, data_dir: str) -> None:
    """Print all consolidations with version history."""
    summary_store = SummaryStore(data_dir=data_dir)
    all_data = summary_store.get_all_summaries_with_versions(chat_id)
    
    print(f"=== Consolidations for {chat_id} ===")
    print()
    
    has_any = False
    
    for summary_type in ["daily", "weekly", "monthly"]:
        items = all_data.get(summary_type, [])
        print(f"{summary_type.title()} Summaries:")
        
        if not items:
            print("  (none)")
        else:
            has_any = True
            for summary, versions in items:
                version_count = len(versions) + 1  # +1 for current
                content_preview = summary.content[:60].replace("\n", " ")
                if len(summary.content) > 60:
                    content_preview += "..."
                
                print(f"  {summary.period}  [{version_count} version{'s' if version_count > 1 else ''}]")
                print(f"    - current: {len(summary.content)} chars")
                print(f"      {content_preview}")
                
                for ver in versions:
                    ver_preview = ver.content[:50].replace("\n", " ")
                    if len(ver.content) > 50:
                        ver_preview += "..."
                    print(f"    - {ver.version_id}: {ver.timestamp.strftime('%Y-%m-%d %H:%M')} ({len(ver.content)} chars)")
                    print(f"      {ver_preview}")
        print()
    
    if not has_any:
        print("(no consolidations found)")


async def _run_reconsolidation(
    chat_id: str,
    summary_type: str,
    from_period: str,
    until_period: str | None,
    data_dir: str,
    llm: LLMProvider,
) -> None:
    """Run reconsolidation for the specified type and period range."""
    from mai_companion.memory.summarizer import MemorySummarizer
    from mai_companion.memory.knowledge_base import WikiStore
    
    async with get_session() as session:
        # Get companion info
        result = await session.execute(select(Companion).where(Companion.id == chat_id))
        companion = result.scalar_one_or_none()
        if companion is None:
            raise SystemExit(f"Error: no companion found for chat '{chat_id}'.")
        
        message_store = MessageStore(session)
        summary_store = SummaryStore(data_dir=data_dir)
        wiki_store = WikiStore(session, data_dir=data_dir)
        
        summarizer = MemorySummarizer(
            message_store,
            summary_store,
            llm,
            wiki_store=wiki_store,
            companion_name=companion.name,
            companion_model=companion.llm_model,
        )
        
        def progress_callback(period: str, status: str) -> None:
            print(f"  {period}: {status}")
        
        print(f"=== Re-consolidating {summary_type} summaries ===")
        print(f"From: {from_period}")
        if summary_type == "daily":
            print(f"Until: {until_period or '(yesterday — today is excluded as incomplete)'}")
        else:
            print(f"Until: {until_period or '(current period)'}")
        print()
        
        if summary_type == "daily":
            from_date = _parse_target_date(from_period)
            until_date = _parse_target_date(until_period) if until_period else None
            results = await summarizer.reconsolidate_daily_from(
                chat_id,
                from_date,
                until_date=until_date,
                on_progress=progress_callback,
            )
            print()
            print(f"Re-consolidated {len(results)} daily summaries.")
            
        elif summary_type == "weekly":
            from_week = _parse_week_period(from_period)
            until_week = _parse_week_period(until_period) if until_period else None
            results = await summarizer.reconsolidate_weekly_from(
                chat_id,
                from_week,
                until_period=until_week,
                on_progress=progress_callback,
            )
            print()
            print(f"Re-consolidated {len(results)} weekly summaries.")
            
        elif summary_type == "monthly":
            from_month = _parse_month_period(from_period)
            until_month = _parse_month_period(until_period) if until_period else None
            results = await summarizer.reconsolidate_monthly_from(
                chat_id,
                from_month,
                until_period=until_month,
                on_progress=progress_callback,
            )
            print()
            print(f"Re-consolidated {len(results)} monthly summaries.")


async def _print_prompt(chat_id: str, data_dir: str, llm: LLMProvider, clock: Clock) -> None:
    from mai_companion.mcp_servers.manager import MCPManager
    from mai_companion.mcp_servers.wiki_server import WikiMCPServer
    from mai_companion.mcp_servers.messages_server import MessagesMCPServer
    from mai_companion.mcp_servers.sleep_server import SleepMCPServer

    async with get_session() as session:
        result = await session.execute(select(Companion).where(Companion.id == chat_id))
        companion = result.scalar_one_or_none()
        if companion is None:
            raise SystemExit(f"Error: no companion found for chat '{chat_id}'. Run --start first.")
        message_store = MessageStore(session)
        wiki_store = WikiStore(session, data_dir=data_dir)
        summary_store = SummaryStore(data_dir=data_dir)
        mood_manager = MoodManager(session)
        traits = json.loads(companion.personality_traits)
        mood = await mood_manager.get_current_mood(companion.id, traits)
        prompt_builder = PromptBuilder(
            llm,
            message_store,
            wiki_store,
            summary_store,
        )
        context = await prompt_builder.build_context(companion, mood, clock=clock)

        # Get available tools
        mcp_manager = MCPManager()
        mcp_manager.register_server("wiki", WikiMCPServer(wiki_store, chat_id))
        mcp_manager.register_server("messages", MessagesMCPServer(message_store, chat_id))
        mcp_manager.register_server("sleep", SleepMCPServer())
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
        # Print the message content
        if msg.role.value == "tool":
            # Tool result message - show with its tool_call_id
            print(f"[tool result:{msg.tool_call_id}] {msg.content}")
        else:
            print(f"[{msg.role.value}] {msg.content}")
        # If assistant message has tool calls, show them
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


def _resolve_user_id(args: argparse.Namespace, settings) -> str:
    if args.user_id:
        return args.user_id
    allowed_users = settings.get_allowed_user_ids()
    if allowed_users:
        return sorted(allowed_users)[0]
    return "console-user"


async def _seed_messages(chat_id: str, jsonl_path: str) -> int:
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(f"Error: seed file not found: {jsonl_path}")

    async with get_session() as session:
        result = await session.execute(select(Companion).where(Companion.id == chat_id))
        companion = result.scalar_one_or_none()
        if companion is None:
            raise SystemExit(f"Error: no companion found for chat '{chat_id}'. Run --start first.")

        message_store = MessageStore(session)
        seeded = 0
        offsets_by_day: dict[date, int] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Error: invalid JSON at line {line_no}: {exc.msg}") from exc
                if not isinstance(payload, dict):
                    raise SystemExit(f"Error: line {line_no} must be a JSON object.")

                role = payload.get("role")
                content = payload.get("content")
                raw_date = payload.get("date")
                if role not in {"user", "assistant"}:
                    raise SystemExit(f"Error: line {line_no} has invalid role '{role}'.")
                if not isinstance(content, str) or not content.strip():
                    raise SystemExit(f"Error: line {line_no} has empty content.")

                timestamp = None
                if raw_date is not None:
                    if not isinstance(raw_date, str):
                        raise SystemExit(f"Error: line {line_no} date must be a string.")
                    try:
                        target_day = date.fromisoformat(raw_date)
                    except ValueError as exc:
                        raise SystemExit(
                            f"Error: line {line_no} has invalid date '{raw_date}'."
                        ) from exc
                    seconds = offsets_by_day.get(target_day, 0)
                    offsets_by_day[target_day] = seconds + 1
                    timestamp = datetime.combine(target_day, datetime.min.time(), tzinfo=timezone.utc)
                    timestamp += timedelta(seconds=seconds)

                try:
                    await message_store.save_message(
                        companion_id=chat_id,
                        role=role,
                        content=content,
                        timestamp=timestamp,
                        is_proactive=False,
                    )
                except ValueError as exc:
                    raise SystemExit(f"Error: line {line_no} failed monotonic check: {exc}") from exc

                seeded += 1
    return seeded


def _restore_onboarding_session(
    state_store: ConsoleStateStore,
    *,
    chat_id: str,
    user_id: str,
    handler: BotHandler,
) -> None:
    state = state_store.load()
    chats = state.get("chats", {})
    if not isinstance(chats, dict):
        return
    chat_state = chats.get(chat_id, {})
    if not isinstance(chat_state, dict):
        return
    raw = chat_state.get("onboarding")
    if not isinstance(raw, dict):
        return

    try:
        session = OnboardingSession(
            user_id=user_id,
            chat_id=chat_id,
            state=OnboardingState(raw.get("state", OnboardingState.NOT_STARTED.value)),
            language=str(raw.get("language", "English")),
            language_style=raw.get("language_style"),
            companion_name=str(raw.get("companion_name", "")),
            companion_gender=Gender(raw.get("companion_gender", Gender.NEUTRAL.value)),
            preset_name=raw.get("preset_name"),
            custom_traits=dict(raw.get("custom_traits", {})),
            current_trait_index=int(raw.get("current_trait_index", 0)),
            communication_style=CommunicationStyle(
                raw.get("communication_style", CommunicationStyle.CASUAL.value)
            ),
            verbosity=Verbosity(raw.get("verbosity", Verbosity.CONCISE.value)),
            appearance=raw.get("appearance"),
            last_message_id=raw.get("last_message_id"),
            warning_acknowledged=bool(raw.get("warning_acknowledged", False)),
        )
    except Exception:
        return
    handler._onboarding._sessions[user_id] = session


def _persist_onboarding_session(
    state_store: ConsoleStateStore,
    *,
    chat_id: str,
    user_id: str,
    handler: BotHandler,
) -> None:
    state = state_store.load()
    chats = state.setdefault("chats", {})
    if not isinstance(chats, dict):
        chats = {}
        state["chats"] = chats
    chat_state = chats.setdefault(chat_id, {})
    if not isinstance(chat_state, dict):
        chat_state = {}
        chats[chat_id] = chat_state

    session = handler._onboarding.get_session(user_id)
    if session is None:
        chat_state.pop("onboarding", None)
        state_store.save(state)
        return

    chat_state["onboarding"] = {
        "state": session.state.value,
        "language": session.language,
        "language_style": session.language_style,
        "companion_name": session.companion_name,
        "companion_gender": session.companion_gender.value,
        "preset_name": session.preset_name,
        "custom_traits": session.custom_traits,
        "current_trait_index": session.current_trait_index,
        "communication_style": session.communication_style.value,
        "verbosity": session.verbosity.value,
        "appearance": session.appearance,
        "last_message_id": session.last_message_id,
        "warning_acknowledged": session.warning_acknowledged,
    }
    state_store.save(state)


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
    state_store = ConsoleStateStore()

    # Handle --list before requiring chat_id
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
    if args.date is not None and not args.replay:
        clock = state_store.set_target_date(chat_id, args.date)
    else:
        clock = state_store.get_clock(chat_id)

    engine = await init_db(settings.database_url, echo=settings.debug)
    await run_migrations(engine)

    llm_base: OpenRouterProvider | None = None
    llm: LLMProvider | None = None
    try:
        if args.history:
            await _print_history(chat_id)
            return
        if args.replay:
            await _print_replay(chat_id, data_dir=settings.memory_data_dir, target_date=args.date)
            return
        if args.wiki:
            await _print_wiki(chat_id, settings.memory_data_dir)
            return
        if args.summaries:
            _print_summaries(chat_id, settings.memory_data_dir)
            return
        if args.list_consolidations:
            _print_consolidations(chat_id, settings.memory_data_dir)
            return
        if args.seed:
            count = await _seed_messages(chat_id, args.seed)
            print(f"Seeded {count} messages into chat '{chat_id}'.")
            return

        if not settings.openrouter_api_key:
            raise SystemExit("Error: OPENROUTER_API_KEY is required for this command.")

        llm_base = OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            default_model=settings.llm_model,
        )
        llm = llm_base
        logger_provider: LLMLoggerProvider | None = None
        if args.debug:
            logger_provider = LLMLoggerProvider(
                llm_base,
                chat_id=chat_id,
                base_dir=Path(settings.memory_data_dir) / "debug_logs",
                clock=clock,
            )
            llm = logger_provider

        if args.show_prompt:
            await _print_prompt(chat_id, settings.memory_data_dir, llm, clock)
            return

        if args.reconsolidate:
            if not args.from_period:
                raise SystemExit(
                    "Error: --from is required for reconsolidation. "
                    "Specify the starting period."
                )
            await _run_reconsolidation(
                chat_id,
                args.reconsolidate,
                args.from_period,
                args.until_period,
                settings.memory_data_dir,
                llm,
            )
            return

        messenger = ConsoleMessenger()
        _handler = BotHandler(
            messenger,
            llm,
            memory_data_dir=settings.memory_data_dir,
            summary_threshold=settings.summary_threshold,
            wiki_context_limit=settings.wiki_context_limit,
            short_term_limit=settings.short_term_limit,
            tool_max_iterations=settings.tool_max_iterations,
            clock_provider=lambda _chat_id: clock,
        )
        _restore_onboarding_session(
            state_store,
            chat_id=chat_id,
            user_id=user_id,
            handler=_handler,
        )
        if args.start:
            await messenger.dispatch_message(_incoming_command(chat_id, user_id, "start"))
        if args.cb:
            await messenger.dispatch_callback(
                chat_id=chat_id,
                user_id=user_id,
                callback_data=args.cb,
            )
        if args.message:
            await messenger.dispatch_text(
                chat_id=chat_id,
                user_id=user_id,
                text=args.message,
            )
        _persist_onboarding_session(
            state_store,
            chat_id=chat_id,
            user_id=user_id,
            handler=_handler,
        )

        if not (args.start or args.cb or args.message):
            raise SystemExit(
                "Error: nothing to do. Provide a message, --start, --cb, or an inspection flag."
            )

        if logger_provider is not None:
            stats = logger_provider.get_session_stats()
            print("")
            print("--- Debug Info ---")
            print(
                f"LLM calls: {stats['llm_calls']} ({stats['calls_with_tool_calls']} with tool calls)"
            )
            tools_used = ", ".join(stats["tools_used"]) if stats["tools_used"] else "none"
            print(f"Tools used: {tools_used}")
            print(
                f"Tokens: {stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} completion = "
                f"{stats['total_tokens']:,} total"
            )
            print("")
            print("--- Session Cost ---")
            print(
                f"This call: {stats['last_call_total_tokens']:,} tokens (${stats['last_call_cost_usd']:.3f})"
            )
            print(f"Session total: {stats['total_tokens']:,} tokens (${stats['session_cost_usd']:.3f})")
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

