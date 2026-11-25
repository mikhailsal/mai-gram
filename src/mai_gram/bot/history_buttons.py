"""Callback data shared by history actions and response keyboards."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mai_gram.messenger.base import InlineKeyboardSpec

_MAX_MESSAGE_ID = 2**63 - 1


def parse_message_id(value: str) -> int | None:
    """Parse a positive SQLite or Telegram message ID."""
    if len(value) > 19 or not value.isascii() or not value.isdecimal():
        return None
    message_id = int(value)
    return message_id if 0 < message_id <= _MAX_MESSAGE_ID else None


def parse_history_target(value: str) -> int | str | None:
    """Read either a historical numeric ID or a stable message token."""
    if re.fullmatch(r"[0-9a-f]{32}", value):
        return value
    return parse_message_id(value)


def history_buttons(token: str) -> InlineKeyboardSpec:
    return [[("🔄 Regenerate", f"regen:{token}"), ("✂ Cut this & above", f"cut:{token}")]]


def retry_buttons(target: int | str | None, pending_token: str | None) -> InlineKeyboardSpec:
    """Retry the original action after a generation error."""
    if target is not None:
        return [[("🔄 Regenerate", f"regen:{target}")]]
    if pending_token is not None:
        return [[("🔄 Regenerate", f"retry:{pending_token}")]]
    return []
