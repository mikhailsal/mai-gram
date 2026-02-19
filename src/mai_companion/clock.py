"""Clock utilities for virtual time simulation in console sessions."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONSOLE_STATE_PATH = Path("data/.console_state.json")


class Clock:
    """Per-chat time offset clock."""

    def __init__(self, offset: timedelta | None = None):
        self._offset = offset or timedelta()

    def now(self) -> datetime:
        return datetime.now(timezone.utc) + self._offset

    def today(self) -> date:
        return self.now().date()

    @classmethod
    def for_target_date(cls, target: date) -> "Clock":
        """Create a clock whose "today" is the given target date."""
        real_now = datetime.now(timezone.utc)
        target_start = datetime.combine(target, real_now.time(), tzinfo=timezone.utc)
        offset = target_start - real_now
        return cls(offset=offset)

    @property
    def offset(self) -> timedelta:
        return self._offset


class ConsoleStateStore:
    """Persistence layer for CLI chat state and per-chat time offsets."""

    def __init__(self, state_path: Path | str = DEFAULT_CONSOLE_STATE_PATH) -> None:
        self._state_path = Path(state_path)

    def load(self) -> dict[str, Any]:
        """Load persisted state from disk, returning defaults if file is missing."""
        if not self._state_path.exists():
            return {"last_chat_id": None, "chats": {}}

        data = json.loads(self._state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"last_chat_id": None, "chats": {}}

        chats = data.get("chats")
        if not isinstance(chats, dict):
            data["chats"] = {}
        if "last_chat_id" not in data:
            data["last_chat_id"] = None
        return data

    def save(self, state: dict[str, Any]) -> None:
        """Write state to disk, ensuring the parent directory exists."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)
        self._state_path.write_text(payload + "\n", encoding="utf-8")

    def get_last_chat_id(self) -> str | None:
        state = self.load()
        last = state.get("last_chat_id")
        return last if isinstance(last, str) else None

    def set_last_chat_id(self, chat_id: str) -> None:
        state = self.load()
        state["last_chat_id"] = chat_id
        self.save(state)

    def get_time_offset_seconds(self, chat_id: str) -> float:
        state = self.load()
        chats = state.get("chats", {})
        if not isinstance(chats, dict):
            return 0.0
        chat_state = chats.get(chat_id, {})
        if not isinstance(chat_state, dict):
            return 0.0
        raw_seconds = chat_state.get("time_offset_seconds", 0.0)
        if isinstance(raw_seconds, (float, int)):
            return float(raw_seconds)
        return 0.0

    def set_time_offset_seconds(self, chat_id: str, offset_seconds: float) -> None:
        state = self.load()
        chats = state.setdefault("chats", {})
        if not isinstance(chats, dict):
            chats = {}
            state["chats"] = chats

        chat_state = chats.setdefault(chat_id, {})
        if not isinstance(chat_state, dict):
            chat_state = {}
            chats[chat_id] = chat_state

        chat_state["time_offset_seconds"] = float(offset_seconds)
        self.save(state)

    def get_clock(self, chat_id: str) -> Clock:
        """Build a clock from the persisted offset for the chat."""
        offset_seconds = self.get_time_offset_seconds(chat_id)
        return Clock(offset=timedelta(seconds=offset_seconds))

    def set_target_date(self, chat_id: str, target: date) -> Clock:
        """Persist an offset so that this chat's virtual "today" equals target."""
        clock = Clock.for_target_date(target)
        self.set_time_offset_seconds(chat_id, clock.offset.total_seconds())
        return clock
