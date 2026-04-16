# mai-gram

**Lightweight Telegram-to-LLM bridge via OpenRouter.**

A self-hosted service that connects Telegram bots to LLM models through OpenRouter. Each user picks their model and system prompt. No frills, no personality systems -- just a clean chat bridge.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mikhailsal/mai-gram.git
cd mai-gram
make install-dev

# Configure
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY

# Run
make run
```

Then message your Telegram bot and use `/start` to set up.

---

## Make Targets

Run `make help` to see all available commands. Here's the summary:

### Running the Bot

| Command | Description |
|---------|-------------|
| `make run` | Run the Telegram bot (production mode) |
| `make run-dev` | Run with auto-reload (restarts on code changes) |

### Console Chat (mai-chat CLI)

All console commands accept `CHAT=<id>` to specify the chat. Default: `test-makefile`.

| Command | Description |
|---------|-------------|
| `make chat-start` | Start setup flow (model + prompt selection) |
| `make chat-setup` | One-shot setup: `CHAT=test-demo MODEL=openai/gpt-4o-mini PROMPT=default` |
| `make chat MSG="Hello!"` | Send a message |
| `make chat-list` | List all chats with message counts |
| `make chat-history` | Show conversation history |
| `make chat-wiki` | Show wiki entries |
| `make chat-prompt` | Show the assembled LLM prompt (debug) |
| `make chat-debug MSG="Hello!"` | Send a message with full LLM debug logging |
| `make chat-import FILE=conv.json` | Import dialogue from JSON file |

**Examples:**

```bash
# Create a new chat with gpt-4o-mini and the default prompt
make chat-setup CHAT=test-demo

# Send a message
make chat CHAT=test-demo MSG="What is 2+2?"

# View history
make chat-history CHAT=test-demo

# Import a conversation from another system
make chat-import CHAT=test-demo FILE=exported_chat.json
```

### Testing & Code Quality

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-v` | Run tests with verbose output |
| `make test-cov` | Run tests with coverage report (enforces 90% minimum) |
| `make lint` | Check code with ruff |
| `make format` | Format code with ruff |
| `make typecheck` | Run mypy type checker |
| `make check` | Run all quality checks (lint + format + typecheck) |
| `make fix` | Auto-fix lint issues and reformat |
| `make precommit` | Run all pre-commit checks (lint, format, typecheck, tests + coverage) |
| `make install-hooks` | Install git pre-commit hook |

### Docker

| Command | Description |
|---------|-------------|
| `make docker-build` | Build Docker image |
| `make docker-up` | Start in Docker (detached) |
| `make docker-down` | Stop and remove containers |
| `make docker-logs` | View container logs (follow mode) |
| `make docker-restart` | Restart containers |
| `make docker-shell` | Open a shell in the running container |

### Cleanup

| Command | Description |
|---------|-------------|
| `make clean` | Remove Python bytecode and caches |
| `make clean-test` | Remove test artifacts (coverage, htmlcov) |
| `make clean-all` | Remove everything generated |

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-bot support** | Run 20+ Telegram bots with per-bot settings (`config/bots.toml`) |
| **Per-bot restrictions** | Whitelist users, models, and prompts per bot |
| **Per-user configuration** | Each user selects their own model and system prompt |
| **Model whitelist** | Configurable list of allowed models (`config/models.toml`) |
| **System prompt templates** | Predefined prompts in `prompts/` or custom user input |
| **Wiki (knowledge base)** | AI can save, edit, search, list, and recall facts via MCP tools; `.md` files on disk are the source of truth |
| **Dialogue import** | Import conversations from JSON (OpenAI format) |
| **Custom OpenRouter URL** | Point to a local proxy for debugging |
| **Console CLI** | Debug and inspect chats from the command line |
| **Self-hosted** | All data stays on your hardware |

---

## Configuration

### Multi-Bot Setup (config/bots.toml) — Recommended

For running multiple bots with per-bot restrictions, create `config/bots.toml` (gitignored for security):

```bash
cp config/bots.toml.example config/bots.toml
# Edit config/bots.toml: add your bot tokens and settings
```

```toml
# Bot for Alice — full access
[[bots]]
token = "123456:ABC-DEF..."
allowed_users = [111111111]

# Bot for Bob — restricted models and prompts
[[bots]]
token = "789012:GHI-JKL..."
allowed_users = [222222222]
allowed_models = ["google/gemini-2.5-flash"]
allowed_prompts = ["default", "coder"]
```

Each bot can have:
- **`allowed_users`** — Telegram user IDs that may use this bot (omit to use global `ALLOWED_USERS`)
- **`allowed_models`** — subset of models from `config/models.toml` (omit for all)
- **`allowed_prompts`** — subset of prompt templates from `prompts/` (omit for all + custom)

When `config/bots.toml` exists, legacy `TELEGRAM_BOT_TOKEN*` env vars are ignored.

### Environment Variables (.env)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Legacy | - | Primary bot token (ignored if bots.toml exists) |
| `TELEGRAM_BOT_TOKEN_2` | Legacy | - | Second bot token (ignored if bots.toml exists) |
| `TELEGRAM_BOT_TOKEN_3` | Legacy | - | Third bot token (ignored if bots.toml exists) |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL (for proxy) |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Default model |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_gram.db` | Database URL |
| `ALLOWED_USERS` | No | - | Comma-separated Telegram user IDs (global fallback) |
| `DEBUG` | No | `false` | Enable debug mode |

### Model Whitelist (config/models.toml)

```toml
[models]
allowed = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
]
default = "openai/gpt-4o-mini"
```

### System Prompt Templates (prompts/)

Place `.txt` or `.md` files in the `prompts/` directory. Users can select from these during `/start` or type a custom prompt.

---

## Dialogue Import Format

```json
[
    {"role": "user", "content": "Hello!", "timestamp": "2024-01-15T14:30:00Z"},
    {"role": "assistant", "content": "Hi there!", "reasoning": "User greeted me."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "function": {"name": "wiki_create", "arguments": "{\"key\": \"greeting\", \"content\": \"User says hello\", \"importance\": 5000}"}}]},
    {"role": "tool", "content": "Created wiki entry.", "tool_call_id": "tc1"}
]
```

---

## Architecture

```
Telegram User  -->  Telegram Bot(s)  -->  mai-gram  -->  OpenRouter  -->  LLM
                                             |
                               ┌─────────────┼─────────────┐
                               │             │             │
                          SQLite DB     Wiki .md files   MCP tools
                       (messages,       (source of     (wiki, messages,
                        chat config,     truth for       external)
                        wiki index)      knowledge)
```

Wiki entries live as markdown files on disk (`data/<chat_id>/wiki/*.md`)
and are indexed in SQLite for fast querying. The files are the source of
truth — the database index is automatically rebuilt from disk on every
message and can be manually repaired with `mai-chat --repair-wiki`. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#wiki-knowledge-base-architecture)
for the full architecture.

---

## License

MIT
