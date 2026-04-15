# Changelog

All notable changes to mai-gram are documented here.

## [1.0.0] - 2026-04-15

First stable release. The project was forked from mai-companion and rebuilt as
**mai-gram** — a lightweight, self-hosted Telegram-to-LLM bridge via OpenRouter.

### Core

- Fork and rebrand from mai-companion into mai-gram with clean architecture
- Multi-bot support — run up to 3 Telegram bots simultaneously from one process
- Streaming LLM responses with real-time message editing in Telegram
- Access control via `ALLOWED_USERS` to restrict bot usage
- Per-user model and system-prompt selection through `/start` onboarding
- Model whitelist configuration via `config/models.toml` with mtime-based caching
- System prompt templates loaded from `prompts/` directory
- Dialogue import from JSON (OpenAI message format with timestamps)

### Memory & Knowledge

- MCP tool-calling integration for wiki (knowledge base) read/write
- Memory consolidation system with daily summaries and backfill
- Summary versioning and reconsolidation for long-running conversations
- Personal information handling through prompt builder
- Tool call persistence to database for auditability
- Message search tools for retrieving conversation history

### Console CLI (`mai-chat`)

- Full console messenger for testing and debugging without Telegram
- `--list` command to show all chats with message stats
- Interactive setup flow (model + prompt selection)
- One-shot setup via `make chat-setup`
- Message sending, history viewing, wiki inspection, prompt debugging
- Button extraction logic for simulating Telegram keyboards in terminal

### Telegram UX

- Tool calls displayed as separate blockquote messages instead of merged text
- "Cut this & above" with in-place visual markers for conversation trimming
- Actionable error messages with Regenerate button on LLM failures
- Consecutive user messages merged to prevent LLM confusion
- Sleep tool for multi-message delivery pacing
- Improved timeout settings and retry logic for message sending

### Developer Experience

- Makefile with 25+ targets: run, test, lint, format, typecheck, docker, chat
- Auto-reload development mode (`make run-dev`) via watchfiles
- Pre-commit hook with full quality gate (lint + format + typecheck + tests)
- 204 tests passing with 90%+ coverage enforcement
- Ruff linting and formatting, mypy strict type checking
- Docker support (build, up, down, logs, shell)

### Documentation

- Comprehensive README centered on Make targets
- Getting Started, Configuration, Development, and Debugging guides
- Project philosophy documenting the ethical AI companion model

## [0.1.0] - 2025-01-01

Initial prototype as mai-companion (pre-fork).
