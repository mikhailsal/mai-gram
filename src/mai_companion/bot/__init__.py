"""Telegram bot handlers and middleware.

This module provides the Telegram integration for mAI Companion:
- BotHandler: Main message handling coordinator
- OnboardingManager: Character creation flow
- Middleware: Rate limiting and logging
"""

from mai_companion.bot.handler import BotHandler
from mai_companion.bot.middleware import (
    MessageLogger,
    RateLimiter,
    RateLimitConfig,
    RequestContext,
)
from mai_companion.bot.onboarding import OnboardingManager, OnboardingSession, OnboardingState

__all__ = [
    "BotHandler",
    "MessageLogger",
    "OnboardingManager",
    "OnboardingSession",
    "OnboardingState",
    "RateLimiter",
    "RateLimitConfig",
    "RequestContext",
]
