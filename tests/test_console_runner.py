"""Tests for the console CLI helpers."""

from __future__ import annotations

import pytest

from mai_gram.console_runner import _incoming_command, _parse_command_text


def test_parse_command_text_without_args() -> None:
    command, command_args = _parse_command_text("help")

    assert command == "help"
    assert command_args is None


def test_parse_command_text_with_slash_and_args() -> None:
    command, command_args = _parse_command_text("/timezone Europe/Moscow")

    assert command == "timezone"
    assert command_args == "Europe/Moscow"


def test_parse_command_text_rejects_empty_input() -> None:
    with pytest.raises(SystemExit, match="--command requires a command name"):
        _parse_command_text("   ")


def test_incoming_command_includes_command_args() -> None:
    incoming = _incoming_command("test-chat", "test-user", "timezone", "Europe/Moscow")

    assert incoming.command == "timezone"
    assert incoming.command_args == "Europe/Moscow"
    assert incoming.text == "/timezone Europe/Moscow"
