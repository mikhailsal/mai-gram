"""Messenger abstraction layer for platform-agnostic communication.

This module provides an abstract interface that allows mAI Companion
to communicate through different messaging platforms (Telegram, Discord,
Matrix, etc.) with a unified API.
"""

from mai_companion.messenger.base import (
    IncomingMessage,
    Messenger,
    MessengerError,
    OutgoingMessage,
    SendResult,
)

__all__ = [
    "IncomingMessage",
    "Messenger",
    "MessengerError",
    "OutgoingMessage",
    "SendResult",
]
