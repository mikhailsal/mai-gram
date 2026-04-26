"""Tests for console CLI parser and state helpers."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from mai_gram.console_cli import (
    ConsoleStateStore,
    build_parser,
    needs_live_llm,
    resolve_chat_id,
    resolve_user_id,
)


def test_build_parser_parses_repeated_callbacks_and_prompt_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["hello", "--cb", "regen", "--cb", "confirm_regen", "--prompt", "default"]
    )

    assert args.message == "hello"
    assert args.callbacks == ["regen", "confirm_regen"]
    assert args.prompt == "default"


def test_console_state_store_persists_last_chat_id(tmp_path) -> None:
    store = ConsoleStateStore()
    store._STATE_FILE = tmp_path / ".console_state.json"

    store.set_last_chat_id("chat-1")

    assert store.get_last_chat_id() == "chat-1"


def test_resolve_chat_id_prefers_explicit_argument() -> None:
    args = Namespace(chat_id="explicit-chat")
    store = MagicMock(get_last_chat_id=MagicMock(return_value="stored-chat"))

    chat_id = resolve_chat_id(args, store)

    assert chat_id == "explicit-chat"
    store.set_last_chat_id.assert_called_once_with("explicit-chat")


def test_resolve_chat_id_requires_stored_or_explicit_value() -> None:
    args = Namespace(chat_id=None)
    store = MagicMock(get_last_chat_id=MagicMock(return_value=None))

    with pytest.raises(SystemExit, match="no chat ID available"):
        resolve_chat_id(args, store)


def test_resolve_user_id_prefers_allowed_users_when_missing_explicit_value() -> None:
    args = Namespace(user_id=None)
    settings = MagicMock(get_allowed_user_ids=MagicMock(return_value={"b-user", "a-user"}))

    user_id = resolve_user_id(args, settings)

    assert user_id == "a-user"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (Namespace(start=False, message="hello", prompt=None, callbacks=None), True),
        (Namespace(start=False, message=None, prompt=None, callbacks=["confirm_regen"]), True),
        (
            Namespace(
                start=True,
                message="custom prompt",
                prompt="__custom__",
                callbacks=None,
            ),
            False,
        ),
    ],
)
def test_needs_live_llm_classifies_console_modes(args: Namespace, expected: bool) -> None:
    assert needs_live_llm(args) is expected
