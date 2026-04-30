# Development

## Setup

```bash
pip install -e ".[dev]"
```

This also installs the git pre-commit hook automatically.

## Project Structure

```
src/mai_gram/
├── bot/          # Telegram message handling and setup flow
├── core/         # Prompt builder, markdown converter
├── db/           # SQLAlchemy models, migrations, database
├── debug/        # LLM call logger and cost tracker
├── llm/          # LLM provider abstraction (OpenRouter)
├── mcp_servers/  # MCP tool servers (wiki, messages, external)
├── memory/       # Message store, wiki store, summarizer (inactive)
├── messenger/    # Messenger abstraction (Telegram, console)
├── config.py     # Settings via pydantic-settings
├── main.py       # Application entry point
└── console_runner.py  # CLI interface
```

## Wiki (Knowledge Base) Architecture

The wiki system stores facts that the AI learns about each human. Understanding
its dual-storage model is critical for troubleshooting.

### Dual-storage: Files + Database

Each wiki entry exists in **two places**:

1. **Markdown files on disk** — `data/<chat_id>/wiki/<importance>_<key>.md`
2. **Database rows** — `KnowledgeEntry` table in SQLite

**The `.md` files on disk are the source of truth.** The database is a
queryable index that mirrors the files. If the two ever disagree, the
file wins.

### File naming convention

Wiki files follow the pattern `<importance>_<key>.md`:

```
data/186215217@maicompaniontestbot/wiki/
├── 0500_fresh_test_entry.md      # importance=500,  key="fresh_test_entry"
├── 3000_цвет_змея.md             # importance=3000, key="цвет_змея"
├── 7000_first_conversation.md    # importance=7000, key="first_conversation"
├── 9999_human_name.md            # importance=9999, key="human_name"
└── changelog.jsonl               # (not a wiki entry — change log)
```

- **Importance** — 4-digit zero-padded integer (0000–9999). Higher = more
  important, included first in the LLM context.
- **Key** — snake_case identifier. Supports Unicode (Cyrillic, CJK, etc.).
- **Content** — free-form markdown inside the file.

### How the AI interacts with wiki

The AI uses MCP tools to manage wiki entries:

| Tool | Purpose |
|------|---------|
| `wiki_create` | Create a new entry (file + DB row) |
| `wiki_edit` | Edit content and/or importance (renames file if importance changes) |
| `wiki_read` | Read a single entry by key |
| `wiki_search` | Search entries by key or content substring |
| `wiki_list` | Browse all entries with sorting and pagination |

### Disk-to-DB synchronization

The database can become out of sync with disk files if files are manually
edited, deleted, or created outside the application (e.g. by hand or by
restoring from a backup).

**`sync_from_disk`** reconciles the database with disk state:

- **File exists, no DB row** → creates a new DB row from the file
  (importance and key parsed from the filename, content from the file body).
- **File exists, DB row exists but differs** → updates the DB row
  (content and/or importance).
- **DB row exists, no file on disk** → deletes the orphaned DB row.
- **Files that don't match the `NNNN_key.md` pattern** → skipped (logged).

This sync runs automatically at explicit workflow boundaries:
1. **Before assistant-turn context assembly**
  (`AssistantTurnBuilder._build_request()`)
2. **Before prompt preview assembly**
  (`PromptPreviewService.build_preview()`)
3. **Before any wiki MCP tool call** (once per `WikiMCPServer` instance)
4. **Before CLI wiki list/repair inspection output**
  (`ChatInspectionService.list_wiki()` / `repair_wiki()`)

### Sync transaction boundary

`sync_from_disk` can create, update, or delete `KnowledgeEntry` rows in the
current DB session. The caller owns transaction control:

- **Mutating workflows** should commit after sync (for example, CLI
  `--repair-wiki` always commits).
- **Read-oriented workflows** that trigger sync should commit only when the
  returned `SyncReport` has changes (for example, CLI `--wiki` commits when
  `total_changes > 0`).

Keeping this boundary explicit avoids hidden writes and makes read-only
prompt-building code easier to reason about.

### Manual repair

If wiki entries are missing or stale, run:

```bash
mai-chat -c <chat_id> --repair-wiki
```

This triggers `sync_from_disk` and prints a detailed report:

```
Wiki sync report for <chat_id>:
  3 created, 1 updated, 2 orphaned DB rows removed
  Created: human_name, favorite_color, birthday
  Updated: profession
  Deleted (DB only): old_stale_entry, renamed_key
```

### Common pitfalls

- **Editing a file's importance** — rename the file (e.g. `3000_name.md` →
  `9999_name.md`). The sync will detect the change.
- **Renaming a key** — rename the file. The old DB row will be deleted and a
  new one created. The AI will see it as a new entry.
- **Deleting an entry** — delete the `.md` file. The DB row will be cleaned
  up on next sync.
- **Restoring from backup** — copy the `.md` files back, then run
  `--repair-wiki` to rebuild the DB index.
- **Empty wiki directory** — if the wiki directory doesn't exist or is empty,
  all DB rows for that chat will be removed on sync.

## LLM Provider & Timeout Architecture

### Timeout configuration

The LLM client uses granular timeouts (not a single flat value):

| Phase | Timeout | Purpose |
|-------|---------|---------|
| **connect** | 10s | Time to establish TCP connection to the API |
| **read** | 45s | Time waiting for first token or between consecutive bytes |
| **write** | 10s | Time to send the request body |
| **pool** | 10s | Time waiting for a connection from the pool |

The **read timeout** is the critical one for detecting hung upstream
providers. If the LLM provider accepts the connection but never sends
data, the request will fail after 45 seconds (not 2 minutes as it was
historically).

### Retry behavior

On transient failures (5xx, network errors, rate limits), the provider
retries up to 2 times (3 attempts total). Non-transient errors (401, 404)
are not retried.

### In-flight request tracking

The provider tracks how many LLM requests are currently active via the
`active_requests` counter. During shutdown, if there are pending requests,
the application logs:

```
INFO - Waiting for N in-flight LLM request(s) to complete...
```

This explains why the process may take time to stop — it waits for the
current LLM response before shutting down the Telegram polling.

### Request lifecycle logging

Every LLM request is logged at INFO level:
- **Start**: `LLM stream started (model=..., messages=N)`
- **Completion**: `Usage data: {...} (resolved cost=..., byok=...)`

If a request starts but no completion appears, the request is hung. Check
the read timeout (45s) and the upstream provider status.

## Testing

```bash
pytest                     # Run all tests
pytest -x                  # Stop on first failure
pytest --cov=mai_gram      # With coverage
make test-cov              # Coverage with 90% enforcement
```

## Linting & Formatting

```bash
ruff check .               # Lint
ruff format .              # Format
mypy src/mai_gram          # Type check
python scripts/check_code_limits.py  # Report-only size audit
make check                 # All four at once
make fix                   # Auto-fix lint + reformat
```

## Pre-commit Hook

A git pre-commit hook enforces quality gates before every commit:

1. **Lint** — `ruff check` passes with zero errors
2. **Format** — `ruff format --check` confirms consistent formatting
3. **Type check** — `mypy` passes with zero errors (strict mode)
4. **Code size audit** — oversized Python files and functions are reported to keep refactoring pressure visible while the current hotspots are being decomposed
5. **Tests + coverage** — all tests pass with ≥ 90% code coverage
6. **Live functional tests** — the real `mai-chat` integration suite passes with a valid `OPENROUTER_API_KEY`

If `OPENROUTER_API_KEY` is not already exported, the hook and `make precommit` will try to load it from `.env`. If no key is available, the live functional step fails instead of silently skipping the real-provider coverage.

The hook is installed automatically by `make install-dev`. To install manually:

```bash
make install-hooks
```

To skip the hook for a single commit (use sparingly):

```bash
git commit --no-verify
```

To run the same checks manually without committing:

```bash
make precommit
```

## Auto-Reload

```bash
python -m mai_gram.main --reload
```

Watches `src/` for Python file changes and auto-restarts.
