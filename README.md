<p align="center">
  <h1 align="center">mai-gram</h1>
  <p align="center">
    <strong>Your personal AI companion on Telegram — powered by any LLM, fully self-hosted.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
    <a href="#features">Features</a> &nbsp;·&nbsp;
    <a href="#configuration">Configuration</a> &nbsp;·&nbsp;
    <a href="docs/DEVELOPMENT.md">Development</a> &nbsp;·&nbsp;
    <a href="CHANGELOG.md">Changelog</a>
  </p>
</p>

---

**mai-gram** turns any Telegram bot into a smart AI companion backed by 200+ LLM models via [OpenRouter](https://openrouter.ai). Each user picks their own model, system prompt, and conversation style — the AI remembers facts through a built-in knowledge base (wiki), supports tool calling, and streams responses in real time. Deploy it on your own hardware and keep full control of your data.

## Why mai-gram?

| | |
|---|---|
| **Any model, one interface** | Switch between GPT-4o, Claude, Gemini, Llama, and 200+ other models without leaving Telegram. |
| **Persistent memory** | The AI builds a wiki of facts about each user — names, preferences, conversation history — and recalls them automatically. |
| **Multi-bot, multi-user** | Run 20+ bots from a single process, each with its own user whitelist, model restrictions, and prompt templates. |
| **Conversation import** | Migrate existing chats from other AI tools — upload a JSON file via Telegram or CLI and the full history is replayed into the bot. |
| **Per-prompt configuration** | Each system prompt template controls which tools are available, whether reasoning is visible, and how timestamps are displayed. |
| **Self-hosted & private** | All data stays on your machine. No cloud dependencies beyond the LLM API itself. |
| **Developer-friendly** | 274 tests, 90%+ coverage, strict typing, auto-reload, Docker support, and a full CLI for debugging. |

## Quick Start

```bash
git clone https://github.com/mikhailsal/mai-gram.git
cd mai-gram
make install-dev

cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY

make run
```

Open your Telegram bot, send `/start`, pick a model and a system prompt — you're chatting.

For development with auto-reload:

```bash
make run-dev
```

## Features

### Streaming AI Chat

Responses stream token-by-token with real-time message editing in Telegram — just like ChatGPT, but in your messenger. Tool calls are displayed as separate blockquote messages for transparency.

### Wiki Knowledge Base

The AI automatically saves important facts about each user into a personal wiki. Entries are stored as plain `.md` files on disk (the source of truth) and indexed in SQLite for fast querying. You can browse, search, or edit wiki files manually — changes are picked up automatically.

```
data/<chat_id>/wiki/
├── 9999_human_name.md
├── 7000_profession.md
├── 5000_favorite_topics.md
└── 3000_recent_context.md
```

### Multi-Bot with Per-Bot Restrictions

Define all your bots in a single `config/bots.toml` file. Each bot can have its own:

- **User whitelist** — who can use it
- **Model restrictions** — which LLMs are available
- **Prompt restrictions** — which system prompts can be selected

```toml
[[bots]]
token = "123456:ABC-DEF..."
allowed_users = [111111111]

[[bots]]
token = "789012:GHI-JKL..."
allowed_users = [222222222]
allowed_models = ["google/gemini-2.5-flash"]
allowed_prompts = ["default", "coder"]
```

### Per-Prompt Configuration

Each system prompt template is paired with a TOML config file that controls:

- **Tool visibility** — enable/disable specific tools or MCP servers per prompt
- **Display defaults** — show or hide reasoning and tool calls
- **Datetime behavior** — whether messages include timestamps

This means a "creative writing" prompt can hide technical tool calls, while a "coder" prompt shows everything for full transparency.

### Conversation Import

Migrate your chat history from other AI tools:

- **Via Telegram**: Send `/import`, pick a model, upload a JSON file — messages are replayed into the chat with formatting and rate limiting
- **Via CLI**: `mai-chat -c test-demo --import-json path/to/messages.json`

Supports OpenAI chat format and AI Proxy v2 request JSON.

### Console CLI (`mai-chat`)

A full command-line interface for testing and debugging without Telegram:

```bash
# Create a new chat
mai-chat -c test-demo --start --model openai/gpt-4o-mini --prompt default

# Chat
mai-chat -c test-demo "What is 2+2?"

# Slash commands
mai-chat -c test-demo --command help
mai-chat -c test-demo --command "timezone Europe/Moscow"

# Inspect
mai-chat -c test-demo --history
mai-chat -c test-demo --wiki
mai-chat -c test-demo --show-prompt

# Repair wiki index
mai-chat -c test-demo --repair-wiki
```

See [docs/DEBUGGING.md](docs/DEBUGGING.md) for the full CLI reference.

## Configuration

### Environment Variables (.env)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Legacy | - | Primary bot token (ignored if bots.toml exists) |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL (for proxy) |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Default model |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_gram.db` | Database URL |
| `ALLOWED_USERS` | No | - | Comma-separated Telegram user IDs |
| `DEBUG` | No | `false` | Enable debug mode |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference including `MEMORY_DATA_DIR`, `WIKI_CONTEXT_LIMIT`, `SHORT_TERM_LIMIT`, LLM timeout settings, and data directory layout.

### Model Whitelist (`config/models.toml`)

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

Changes are hot-reloaded — no restart required.

### System Prompt Templates (`prompts/`)

Place `.txt` files in the `prompts/` directory. Users see these as options during `/start` alongside "Custom (type your own)". Each prompt can have a companion `.toml` config for tool filtering and display settings.

## Architecture

```
Telegram User ──▶ Telegram Bot(s) ──▶ mai-gram ──▶ OpenRouter ──▶ LLM
                                          │
                            ┌──────────────┼──────────────┐
                            │              │              │
                       SQLite DB     Wiki .md files   MCP tools
                    (messages,       (source of      (wiki, messages,
                     chat config,     truth for        external)
                     wiki index)      knowledge)
```

Wiki entries live as markdown files on disk (`data/<chat_id>/wiki/*.md`) and are indexed in SQLite for fast querying. The files are the source of truth — the database index is automatically rebuilt from disk on every message and can be manually repaired with `mai-chat --repair-wiki`.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full architecture and [docs/DEBUGGING.md](docs/DEBUGGING.md) for troubleshooting.

## Make Targets

Run `make help` to see all available commands. Key highlights:

| Category | Command | Description |
|----------|---------|-------------|
| **Run** | `make run` | Run the bot (production) |
| | `make run-dev` | Run with auto-reload |
| **Chat CLI** | `make chat MSG="Hi"` | Send a message |
| | `make chat-setup` | One-shot setup (model + prompt) |
| | `make chat-list` | List all chats |
| | `make chat-import FILE=conv.json` | Import dialogue from JSON |
| **Quality** | `make test` | Run all tests |
| | `make test-cov` | Tests + 90% coverage enforcement |
| | `make check` | Lint + format + typecheck |
| | `make precommit` | Full quality gate |
| **Docker** | `make docker-up` | Start in Docker |
| | `make docker-logs` | View container logs |

## Development

```bash
make install-dev    # Install with dev dependencies + git hooks
make test           # Run 274 tests
make check          # Lint + format + typecheck
make precommit      # Full pre-commit quality gate
```

The project enforces strict quality standards:

- **Ruff** for linting and formatting
- **mypy** in strict mode for type checking
- **90%+ coverage** enforced on every commit
- **Pre-commit hooks** run the full quality gate automatically, including live functional tests when `OPENROUTER_API_KEY` is available via the environment or `.env`

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, project structure, and testing.

## License

MIT
