# Changelog

All notable changes to mai-gram are documented here.

## [1.1.0] - 2026-04-17

### Wiki

- Add `wiki_list` MCP tool — AI can now browse all wiki entries with sorting
  (by importance, key, or last update) and pagination
- Implement disk-to-DB synchronization (`sync_from_disk`): `.md` files on disk
  are the source of truth; the database index is rebuilt automatically and can
  be repaired manually with `mai-chat --repair-wiki`
- Remove deprecated `get_top_entries` and `list_entries` methods in favour of
  the unified `list_entries_sorted`

### LLM Provider

- Reduce default LLM timeout from 120s flat to granular timeouts:
  connect=10s, read=45s, write=10s, pool=10s — the bot now recovers ~3x
  faster from hung upstream providers
- Add `active_requests` counter to track in-flight LLM calls
- Log "LLM stream started (model=..., messages=N)" at the beginning of every
  request for hang diagnostics

### Shutdown

- Log "Waiting for N in-flight LLM request(s) to complete..." during shutdown
  when LLM requests are still pending, explaining why the process may be slow
  to stop

### Multi-Bot

- Fix missing `[[bots]]` header in `config/bots.toml` that caused TOML parse
  errors

### Documentation

- Document wiki dual-storage architecture (files = source of truth, DB = index)
- Add wiki troubleshooting guide (repair, manual editing, backup restore)
- Document LLM timeout configuration and shutdown diagnostics
- Update architecture diagram to reflect file-based wiki storage

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
