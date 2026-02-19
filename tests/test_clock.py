"""Tests for clock offset behavior and console state persistence."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from mai_companion.clock import Clock, ConsoleStateStore


class TestClock:
    def test_now_applies_offset(self) -> None:
        clock = Clock(offset=timedelta(hours=2))

        delta = clock.now() - datetime.now(timezone.utc)
        assert abs(delta.total_seconds() - 7200) < 2

    def test_for_target_date_sets_virtual_today(self) -> None:
        target = date(2026, 1, 5)
        clock = Clock.for_target_date(target)
        assert clock.today() == target


class TestConsoleStateStore:
    def test_stores_and_loads_last_chat_id(self, tmp_path) -> None:
        state_file = tmp_path / ".console_state.json"
        store = ConsoleStateStore(state_file)

        assert store.get_last_chat_id() is None
        store.set_last_chat_id("chat-1")
        assert store.get_last_chat_id() == "chat-1"

    def test_stores_and_loads_offset_seconds(self, tmp_path) -> None:
        state_file = tmp_path / ".console_state.json"
        store = ConsoleStateStore(state_file)

        assert store.get_time_offset_seconds("chat-1") == 0.0
        store.set_time_offset_seconds("chat-1", 3600.0)
        assert store.get_time_offset_seconds("chat-1") == 3600.0

    def test_set_target_date_persists_clock_offset(self, tmp_path) -> None:
        state_file = tmp_path / ".console_state.json"
        store = ConsoleStateStore(state_file)

        target = date(2026, 2, 19)
        persisted_clock = store.set_target_date("chat-2", target)
        loaded_clock = store.get_clock("chat-2")

        assert persisted_clock.today() == target
        assert loaded_clock.today() == target
