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

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-92%25%2B-brightgreen">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-blue">
  <img alt="Ruff" src="https://img.shields.io/badge/linter-ruff-orange">
  <img alt="Tests" src="https://img.shields.io/badge/tests-717-brightgreen">
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
| **Structured output** | Response template plugins (XML, JSON, markdown) constrain LLM output into validated structured formats — reasoning is preserved across turns instead of being silently truncated. |
| **High-quality codebase** | 717 tests across 4 tiers (unit → integration → functional → live), 92%+ coverage enforced on every commit, mypy strict, ruff, code-size audit, Docker, full CLI. |

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

Responses stream token-by-token with real-time message editing in Telegram — just like ChatGPT, but in your messenger. Long responses automatically overflow into multiple messages with progressive streaming across message boundaries. Tool calls are displayed as separate blockquote messages for transparency. LaTeX symbols, headers, and nested lists are rendered with proper Unicode formatting.

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
# Let this user type arbitrary model ids + params (see "Arbitrary models" below)
custom_model_allowed_users = [111111111]

[[bots]]
token = "789012:GHI-JKL..."
allowed_users = [222222222]
allowed_models = ["google/gemini-2.5-flash"]
allowed_prompts = ["default", "coder"]
```

### Arbitrary Models with Custom Parameters

Users listed in a bot's `custom_model_allowed_users` get a **Custom model (type
your own)** option during `/start` and `/model`. They can pick any OpenRouter
model id that isn't in `config/models.toml` and supply request parameters in a
chat message — the first line is the model id, each following `key = value` line
becomes a request parameter (dotted keys nest, values are type-coerced):

```
openai/gpt-5.4-mini
reasoning.effort = "high"
temperature = 0.7
provider.order = ["OpenAI"]
```

The capability is opt-in per bot and per user; if `custom_model_allowed_users`
is omitted, the option is hidden for everyone on that bot. From the CLI:

```bash
mai-chat -c test-demo --start --custom-model openai/gpt-5.4-mini \
  --custom-model-params reasoning.effort=high temperature=0.7 --prompt default
```

### Per-Prompt Configuration

Each system prompt template is paired with a TOML config file that controls:

- **Tool visibility** — enable/disable specific tools or MCP servers per prompt
- **Display defaults** — show or hide reasoning and tool calls
- **Datetime behavior** — whether messages include timestamps

This means a "creative writing" prompt can hide technical tool calls, while a "coder" prompt shows everything for full transparency.

### Response Template Plugins Support

Most LLM providers strip `reasoning` tokens from the conversation history before sending it back to the model. If you rely on native reasoning output, the model loses its chain of thought on every subsequent turn. This was verified empirically across multiple providers and models in [ai-thought-preserved-bench](https://github.com/mikhailsal/ai-thought-preserved-bench).

Response templates solve this by moving reasoning into regular message content via a structured format. Each template injects format instructions into the system prompt, parses the response fields (e.g. `<thought>` and `<content>` in XML), validates them — retrying if malformed — and renders each field separately in Telegram. Because the reasoning is stored as content, not as a provider-specific metadata field, it survives round-trips and stays visible to the model on the next turn.

Built-in templates:

- **Empty** — no-op passthrough (default for backward compatibility)
- **XML** — `<thought>` / `<content>` tags with regex extraction
- **JSON** — `{"thought", "content"}` object with lenient parsing
- **Markdown Headers** — `## Thought` / `## Content` section splitting
- **XML with Emotions** — extends XML with an `<emotions>` field

Templates are selected during `/start` onboarding (or via `--template` in CLI). Each template supports user-configurable parameters — field names, paragraph counts, emotion counts — that adapt instructions, examples, parsing, and validation dynamically. Use `/toggle <field>` to hide or show individual fields per chat.

Per-bot template filtering is supported via `allowed_templates` in `bots.toml`.

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

### Model Registry (`config/models.toml`)

Each `[models."<key>"]` section defines an available model. Models are enabled by default; set `enabled = false` to hide one without removing its config.

```toml
[models]
default = "openai/gpt-4o-mini"

[models."openai/gpt-4o"]
title = "GPT-4o"

[models."openai/gpt-4o-mini"]
title = "GPT-4o Mini"
temperature = 0.7

[models."google/gemini-2.5-flash"]
title = "Gemini 2.5 Flash"
reasoning.effort = "medium"
provider.order = ["Google AI Studio"]

# Same model with different parameters via alias
[models."gemini-flash-creative"]
id = "google/gemini-2.5-flash"
title = "Gemini Flash (creative)"
temperature = 1.5
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

### MCP-First Tool Architecture

All AI tools are implemented as [Model Context Protocol](https://modelcontextprotocol.io/) servers — the open standard for LLM tool interfaces. This means:

- **Wiki MCP server** — the AI manages its own knowledge base by calling `wiki_create`, `wiki_edit`, `wiki_search`, `wiki_list` tools
- **Messages MCP server** — lets the AI search and reference past conversation history
- **External MCP server** — plug in any third-party MCP-compatible tool server via config. The bot reads the same `~/.cursor/mcp.json` format that Cursor uses, so any MCP server you've already set up for Cursor is one line away from being available to the bot too:

```toml
# config/models.toml
[mcp]
mcp_config_path = "~/.cursor/mcp.json"
external_servers = ["exa"]   # pick which servers to expose to the AI
```

Using MCP as the internal tool protocol means the codebase is compatible with the broader AI tooling ecosystem out of the box, and adding new capabilities follows a well-defined interface rather than custom glue code.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the full architecture and [docs/DEBUGGING.md](docs/DEBUGGING.md) for troubleshooting.

## Development

```bash
make install-dev    # Install with dev dependencies + git hooks
make test           # Run all tests
make check          # Lint + format + typecheck + size audit
make precommit      # Full pre-commit quality gate
```

Run `make help` to see all available targets (run, chat CLI, docker, quality gates).

### Test Architecture

The test suite is organized into four tiers, ordered from fastest to slowest:

| Tier | Count | What it tests |
|------|-------|---------------|
| **Unit** | majority | Pure logic, isolated modules with mocks |
| **Integration** | — | Multi-module workflows with a stub LLM provider, real in-process DB |
| **Functional (local)** | — | Black-box CLI subprocess tests — spawns the real `mai-chat` binary, no API key needed |
| **Functional (live)** | — | End-to-end tests hitting the real OpenRouter API |

The black-box functional tier deserves special mention: tests spawn `mai-chat` as an isolated subprocess and exercise the full stack through its CLI surface — no mocking of internals. This catches integration issues that unit tests miss and documents expected CLI behavior as executable specs.

Parallel execution via `pytest-xdist` with work-stealing runs unit and functional tiers concurrently (~55s vs ~170s serial). Integration tests run serially in a separate step to avoid global-state conflicts.

The pre-commit hook runs all four tiers in order — fast failures are caught before expensive live API calls.

### Code Quality Standards

- **Ruff** — linting and formatting with an extensive rule set (`E, F, I, N, W, UP, B, A, SIM, TCH, C4, PTH, DTZ, S` and more)
- **mypy strict** — full strict type checking, no implicit `Any`, no untyped defs
- **Code-size audit** — automated check enforces a 500-line file and 60-line function limit to keep refactoring pressure visible
- **92%+ coverage** — enforced by `pyproject.toml` as the single source of truth, checked on every commit
- **Pre-commit hooks** — run the full quality gate automatically, including live functional tests when `OPENROUTER_API_KEY` is available

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, project structure, and the full testing guide.

## License

MIT
