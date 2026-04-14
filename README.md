# mai-gram

**Lightweight Telegram-to-LLM bridge via OpenRouter.**

A self-hosted service that connects Telegram bots to LLM models through OpenRouter. Each user picks their model and system prompt. No frills, no personality systems -- just a clean chat bridge.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-bot support** | Run up to 3 Telegram bots simultaneously |
| **Per-user configuration** | Each user selects their own model and system prompt |
| **Model whitelist** | Configurable list of allowed models |
| **System prompt templates** | Predefined prompts or custom user input |
| **Wiki (knowledge base)** | AI can save and recall facts using MCP tools |
| **Dialogue import** | Import conversations from JSON (OpenAI format) |
| **Custom OpenRouter URL** | Point to a local proxy for debugging |
| **Console CLI** | Debug and inspect chats from the command line |
| **Self-hosted** | All data stays on your hardware |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mikhailsal/mai-gram.git
cd mai-gram

# 2. Create your .env file
cp .env.example .env
# Edit .env with your Telegram token and OpenRouter API key

# 3. Install
pip install -e ".[dev]"

# 4. Run
python -m mai_gram.main
```

Then message your Telegram bot and use `/start` to set up.

---

## Configuration

### Environment Variables (.env)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | - | Primary bot token from @BotFather |
| `TELEGRAM_BOT_TOKEN_2` | No | - | Second bot token |
| `TELEGRAM_BOT_TOKEN_3` | No | - | Third bot token |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL (for proxy) |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Default model |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_gram.db` | Database URL |
| `ALLOWED_USERS` | No | - | Comma-separated Telegram user IDs |
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

Place `.txt` or `.md` files in the `prompts/` directory. Users can select from these during setup or type a custom prompt.

---

## CLI (mai-chat)

```bash
# List all chats
mai-chat --list

# Start a new chat
mai-chat -c test-mychat --start

# Send a message
mai-chat -c test-mychat "Hello, how are you?"

# View history
mai-chat -c test-mychat --history

# View wiki
mai-chat -c test-mychat --wiki

# Show assembled prompt
mai-chat -c test-mychat --show-prompt

# Import dialogue from JSON
mai-chat -c test-mychat --import-json conversation.json

# Debug mode (logs LLM calls)
mai-chat -c test-mychat --debug "What is 2+2?"
```

### Dialogue Import Format

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
                                        SQLite DB (messages, wiki, chat config)
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Format
ruff format .

# Type check
mypy src/mai_gram

# Auto-reload during development
python -m mai_gram.main --reload
```

---

## Docker

```bash
docker compose up -d
```

---

## License

MIT
