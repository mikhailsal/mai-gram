"""Messenger abstraction layer for platform-agnostic communication.

This module provides an abstract interface that allows mai-gram
to communicate through different messaging platforms (Telegram, Discord,
Matrix, etc.) with a unified API.
"""

from mai_gram.messenger.base import (
    IncomingMessage,
    Messenger,
    MessengerError,
    OutgoingMessage,
    SendResult,
)
from mai_gram.messenger.console import ConsoleMessenger

__all__ = [
    "ConsoleMessenger",
    "IncomingMessage",
    "Messenger",
    "MessengerError",
    "OutgoingMessage",
    "SendResult",
]
