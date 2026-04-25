"""Artifact readers for black-box functional tests."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def fetch_chat(db_path: Path, chat_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return dict(row) if row is not None else None


def fetch_messages(db_path: Path, chat_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_knowledge_entries(db_path: Path, chat_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM knowledge_entries WHERE chat_id = ? ORDER BY key ASC",
            (chat_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def read_console_state(root: Path) -> dict[str, Any]:
    state_path = root / "data" / ".console_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def read_debug_log_entries(data_dir: Path, chat_id: str) -> list[dict[str, Any]]:
    log_dir = data_dir / "debug_logs" / chat_id
    if not log_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    return entries


def list_backups(data_dir: Path) -> list[Path]:
    backup_dir = data_dir / "backups"
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob("*.zip"))
