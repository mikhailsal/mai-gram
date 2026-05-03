# Changelog

All notable changes to mai-gram are documented here.

## [1.2.0] - 2026-05-03

### Response Template Plugin System

- Add a template plugin architecture that constrains LLM responses into
  structured formats (XML, JSON, markdown headers), validates compliance with
  automatic retries, and renders each field appropriately in Telegram — this
  preserves reasoning across turns, since providers often truncate native
  reasoning from history but template-based structured output stays in message
  content
- Built-in templates: EmptyTemplate (no-op passthrough for backward
  compatibility), XmlTemplate (`<thought>`/`<content>`), JsonTemplate,
  MarkdownHeadersTemplate, and XmlWithEmotionsTemplate (adds `<emotions>` field)
- Template selection during `/start` onboarding with per-bot filtering via
  `allowed_templates` in `bots.toml`
- `/toggle <field>` command to hide/show template fields per chat
- CLI: `--template` argument for non-interactive chat creation
- Database migration v7: add `response_template` and `hidden_template_fields`
  columns to Chat model

### Template Parameters

- Add user-configurable parameter support to response template plugins —
  parameters are declared by each template via `get_params()` and applied
  through `with_params()` at chat creation time
- Configurable parameters per template: `reasoning_field` name,
  `num_reasoning_paragraphs` (1–8), `emotions_field` name, `num_emotions`
  (1–12) — all with min/max clamping and default fallbacks
- Smart dynamic example generation: examples automatically adapt to parameter
  values (e.g., `num_emotions=8` produces exactly 8 emotions in examples)
- New `CONFIGURING_TEMPLATE_PARAMS` setup state with "Use defaults" button or
  free-text `key=value` entry
- CLI: `--template-params KEY=VALUE` for headless chat creation
- Database migration v8: add `template_params` column to Chat model
- Fix: show actual parameter keys (not labels) in template configuration UI so
  users can discover the correct override syntax

### Code Quality Refactoring

- Decompose monolithic `handler.py` (2100+ lines) into 14 focused service
  modules: ConversationExecutor, ResponseRenderer, CallbackRouter,
  SetupWorkflow, ResetWorkflow, ImportWorkflow, RegenerateService,
  ResendService, ConversationService, AssistantTurnBuilder,
  ToolActivityNotifier, HistoryActions, AccessControl, McpManagerFactory
- Unify conversation and regenerate into a shared assistant-turn execution
  pipeline
- Narrow exception handling across the entire codebase: replace broad
  `except Exception` blocks with specific exception types in OpenRouter,
  config loaders, MCP bridge, conversation executor, reset workflow, import
  workflow, external MCP, and main process watcher
- Quarantine the summarization subsystem into `memory/consolidation/`
  subpackage with clear documentation that modules are architecturally complete
  but not invoked from production paths
- Tighten messenger boundary: add `max_message_length` property to Messenger
  base class so bot services read platform-specific limits from the interface
  instead of importing constants directly
- Remove `get_settings()` singleton from downstream services — all remaining
  usages confined to application entry points with explicit dependency injection
- Type JSON-RPC parsing for MCP bridge through typed helper models
- Hide callback-origin message lookup behind messenger adapter helpers
- Normalize wiki importance boundary to strict positive integers
- Unify adapter wiring for Telegram and CLI through shared `AdapterRuntime`
- Add report-only code size audit (`scripts/check_code_limits.py`) with
  500-line file and 60-line function thresholds, wired into `make check` and
  pre-commit
- Raise coverage gate from 90% to 92% with `pyproject.toml` as single source
  of truth (remove inline overrides from Makefile and pre-commit script)

### Telegram UX

- Progressive streaming for long messages: when accumulated text exceeds the
  Telegram limit, the current placeholder is finalized and streaming continues
  into a new message — users see continuous text flow instead of truncation
- Enhance markdown parsing with LaTeX symbol conversion (50+ commands including
  Greek letters, arrows, set operations), header rendering, tiered nested list
  bullets (• → ◦ → ▪ → ▫), and full markdown inside reasoning blockquotes
- Add confirmation dialog and automatic backup to `/reset` command — creates a
  timestamped zip archive of the database and wiki directory before deletion
- Add `/resend_last` command to re-deliver the last AI message with proper
  splitting and formatting

### Bug Fixes

- Fix placeholder leak when `***` horizontal rule precedes headers — the bold
  regex would match across multiple lines swallowing header placeholders,
  causing raw "HH0"/"HH1" tokens in output
- Preserve tool call chain when regenerating after post-tool LLM failure —
  previously Regenerate would delete executed tool calls and re-trigger them,
  duplicating side effects like wiki entries
- Detect and display provider errors during streaming that were silently
  swallowed (non-SSE error JSON, zero-content streams)
- Handle edit failures in streaming response delivery — `_finalize_placeholder`
  and overflow commits now fall back to sending fresh messages when edits fail
- Fix `ConversationExecutor._send_or_edit_placeholder` returning stale message
  ID when both edit attempts fail, causing streaming retries against a "dead"
  placeholder
- Harden replay engine for large imports with intelligent message splitting,
  flood control awareness (parse Telegram's "retry in N" duration), and strict
  message ordering
- Fix `PythonFilter` instance attribute access for watchfiles 1.1.1
  compatibility (was accessing non-existent `allowed_extensions` class
  attribute)

### Console CLI

- Add `--command` flag so slash commands can be exercised from the CLI
  (`mai-chat -c test-demo --command help`)
- Add `--template` and `--template-params` flags for non-interactive setup

### Prompts

- Add psychotherapist (Marcus) system prompt template for evidence-based
  psychotherapy sessions with wiki-based progress tracking (CBT, ACT, DBT, MI,
  SFBT, mindfulness, behavioral activation)

### Developer Experience

- 717 tests (up from 274), 92%+ coverage enforcement
- Add black-box `mai-chat` functional integration suite running as isolated
  subprocesses against the real binary
- Pre-commit hooks now include live functional tests when `OPENROUTER_API_KEY`
  is available
- Parallel test execution via pytest-xdist with work-stealing (6 workers,
  ~55s vs ~170s serial)
- Add `.github/copilot-instructions.md`

### Documentation

- Document explicit wiki sync transaction boundaries in DEVELOPMENT.md
- Add comprehensive code quality refactoring plan (`plans/`)

## [1.1.0] - 2026-04-18

### Multi-Bot

- Scalable multi-bot support via `config/bots.toml` — run 20+ Telegram bots
  from a single process, each with per-bot user whitelists, model restrictions,
  and prompt restrictions (replaces the hardcoded 3-bot env var limit)
- Fix missing `[[bots]]` header in `config/bots.toml` that caused TOML parse
  errors

### Telegram Import

- Add `/import` command for importing conversation history directly through
  Telegram: select a model, upload a JSON file, and messages are replayed into
  the chat with rate limiting, progress updates, and "Cut this & above" buttons
- Support both OpenAI chat format and AI Proxy v2 request JSON, including
  reasoning extraction and tool call preservation
- Extract shared import logic into `core/importer.py` and rate-limited replay
  engine into `core/replay.py`
- Add document upload handling to the Messenger abstraction layer (Telegram and
  console)

### Per-Prompt Configuration

- Per-prompt display config: each system prompt template (`.toml` file) can set
  default values for `show_reasoning`, `show_tool_calls`, and `send_datetime`
  when a new chat is created
- Per-prompt tool and MCP server filtering: prompt TOML configs can
  whitelist/blacklist individual tools and MCP server groups (e.g. the
  "creative" prompt disables message history search)
- Store `prompt_name` in the Chat model (migration v5) so per-prompt config is
  available at conversation time, not only during setup
- Change default visibility: `show_reasoning` and `show_tool_calls` now default
  to `true` (everything visible); individual prompts can override this
- Add companion TOML configs for all bundled prompts: `default.toml`,
  `coder.toml`, `creative.toml`, `independent.toml`

### Per-Message Datetime

- Replace the retroactive chat-level `send_datetime` toggle with a per-message
  `show_datetime` flag (migration v6): toggling `/datetime` now only affects
  future messages, imported messages default to `show_datetime=false`, and
  timezone changes no longer retroactively re-render old messages

### Wiki

- Add `wiki_list` MCP tool — AI can now browse all wiki entries with sorting
  (by importance, key, or last update) and pagination
- Implement disk-to-DB synchronization (`sync_from_disk`): `.md` files on disk
  are the source of truth; the database index is rebuilt automatically and can
  be repaired manually with `mai-chat --repair-wiki`
- Truncate `wiki_list` and `wiki_search` output to short previews (first line,
  max 120 chars) to avoid flooding the LLM context window; AI uses `wiki_read`
  for full content
- Remove deprecated `get_top_entries` and `list_entries` methods in favour of
  the unified `list_entries_sorted`

### LLM Provider

- Reduce default LLM timeout from 120s flat to granular timeouts:
  connect=10s, read=45s, write=10s, pool=10s — the bot now recovers ~3× faster
  from hung upstream providers
- Add `active_requests` counter to track in-flight LLM calls
- Log "LLM stream started (model=..., messages=N)" at the beginning of every
  request for hang diagnostics
- Log "Waiting for N in-flight LLM request(s) to complete..." during shutdown
  when LLM requests are still pending

### Console CLI

- Fix multiple critical bugs in console debugging mode: HTML tags rendering
  raw, LLMLoggerProvider.generate_stream() not tracking stats, setup flow
  broken across separate invocations, stale setup callbacks failing silently
- Add `--model` and `--prompt` CLI arguments for single-command setup
- Add edit buffering: intermediate streaming edits are suppressed by default,
  only the final response is printed; `--stream-debug` flag shows all edits
- Preserve raw HTML tags and add parse mode annotations for debugging
  Telegram-facing behaviour
- Fix Unicode case-insensitive search in SQLite by registering a custom
  `unicode_lower()` SQL function (fixes Cyrillic search misses)

### Import Fixes

- Imported messages now use import-date timestamps with `show_datetime=false`
  instead of parsing original JSON timestamps, preventing the AI from seeing
  misleading dates
- `get_messages_by_timerange` no longer defaults `end_date` to `start_date`
  when omitted — searching from a start date now returns all messages onward

### Prompts

- Add "english-teacher" system prompt template for language learning via
  translation exercises with wiki-based progress tracking

### Documentation

- Rewrite README with seller-oriented structure and updated feature descriptions
- Document wiki dual-storage architecture (files = source of truth, DB = index)
- Add wiki troubleshooting guide (repair, manual editing, backup restore)
- Document LLM timeout configuration and shutdown diagnostics
- Document per-prompt TOML configuration system and bundled templates
- Document Telegram-based dialogue import (`/import` command)
- Rewrite DEBUGGING.md with comprehensive CLI reference
- Update architecture diagram to reflect file-based wiki storage

### Developer Experience

- 274 tests (up from 204), 90%+ coverage enforcement maintained
- New test suites for importer, replay engine, wiki sync, LLM timeouts,
  wiki content previews, Unicode search, and console edit buffering

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
