"""CLI parser and argument/state helpers for mai-chat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mai-chat",
        description="Console interface for mai-gram.",
    )
    _add_interaction_arguments(parser)
    _add_inspection_arguments(parser)
    _add_setup_arguments(parser)
    return parser


def _add_interaction_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--command",
        metavar="COMMAND",
        help=(
            "Dispatch an arbitrary slash command. Accepts 'name' or "
            "'name args...'; for example: --command 'timezone Europe/Moscow'."
        ),
    )


def _add_inspection_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--reasoning-template",
        dest="reasoning_template",
        metavar="TEMPLATE_NAME",
        help=(
            "Transform native reasoning_content into a response template format "
            "during import. The reasoning is wrapped in the template's field tags "
            "(e.g. <thought>...<content>) so it's preserved in conversation history. "
            "Requires --import-json."
        ),
    )
    parser.add_argument(
        "--repair-wiki",
        action="store_true",
        dest="repair_wiki",
        help="Sync wiki DB from disk files (disk is source of truth).",
    )


def _add_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--real",
        action="store_true",
        help="Real conversation mode (disables test mode transparency notice).",
    )
    parser.add_argument(
        "--stream-debug",
        action="store_true",
        dest="stream_debug",
        help="Print every streaming edit (by default only the final response is shown).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="Model ID for --start setup (skips model selection step).",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT_NAME",
        help=(
            "Prompt name for --start setup (skips prompt selection step). "
            "Use a name from the prompts/ directory, or '__custom__' with a message."
        ),
    )
    parser.add_argument(
        "--template",
        metavar="TEMPLATE_NAME",
        help=(
            "Response template for --start setup (skips template selection step). "
            "Available: empty, xml, json, markdown_headers, xml_emotions."
        ),
    )
    parser.add_argument(
        "--template-params",
        metavar="KEY=VALUE",
        nargs="*",
        help=(
            "Template parameter overrides as key=value pairs. "
            "Example: --template-params reasoning_field=scratchpad num_reasoning_paragraphs=4"
        ),
    )


class ConsoleStateStore:
    """Persist console state such as the last chat ID."""

    _STATE_FILE = Path("./data/.console_state.json")

    def load(self) -> dict[str, Any]:
        if self._STATE_FILE.exists():
            result: dict[str, Any] = json.loads(self._STATE_FILE.read_text(encoding="utf-8"))
            return result
        return {}

    def save(self, state: dict[str, Any]) -> None:
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


def resolve_chat_id(args: argparse.Namespace, state_store: ConsoleStateStore) -> str:
    chat_id = args.chat_id or state_store.get_last_chat_id()
    if not chat_id:
        raise SystemExit(
            "Error: no chat ID available. Use -c/--chat-id once, then it will be remembered."
        )
    state_store.set_last_chat_id(chat_id)
    return chat_id


def resolve_user_id(args: argparse.Namespace, settings: object) -> str:
    if args.user_id:
        return str(args.user_id)
    get_fn = getattr(settings, "get_allowed_user_ids", None)
    if get_fn is not None:
        allowed = get_fn()
        if allowed:
            return str(sorted(allowed)[0])
    return "console-user"


def needs_live_llm(args: argparse.Namespace) -> bool:
    is_custom_prompt_setup = (
        bool(args.start)
        and bool(args.message)
        and (
            args.prompt == "__custom__"
            or any(cb_data == "prompt:__custom__" for cb_data in (args.callbacks or []))
        )
    )
    if is_custom_prompt_setup:
        return False
    if bool(args.message):
        return True
    if args.callbacks:
        return any(cb_data == "confirm_regen" for cb_data in args.callbacks)
    return False
