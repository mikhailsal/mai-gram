"""Tests for console replay helper parsing."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from mai_companion.console_runner import _load_tool_events


class TestConsoleReplay:
    def test_load_tool_events_reads_tool_results_from_debug_logs(self, tmp_path: Path) -> None:
        debug_dir = tmp_path / "debug_logs" / "chat-1"
        debug_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "entry_type": "tool_result",
            "timestamp": "2026-01-05T14:31:08Z",
            "tool_name": "wiki_create",
            "arguments": {"key": "human_name", "content": "Alex", "importance": 9999},
        }
        (debug_dir / "2026-01-05.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

        events = _load_tool_events("chat-1", data_dir=str(tmp_path), target_date=date(2026, 1, 5))

        assert date(2026, 1, 5) in events
        assert len(events[date(2026, 1, 5)]) == 1
        _, line = events[date(2026, 1, 5)][0]
        assert line.startswith("[tool] wiki_create(")
