# Configuration

> **All settings and environment variables for mAI Companion.**

Configuration is managed through environment variables, which can be set in a `.env` file or directly in your environment.

---

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Primary Telegram Bot API token |
| `TELEGRAM_BOT_TOKEN_2` | No | — | Second Telegram bot token (for multi-bot) |
| `TELEGRAM_BOT_TOKEN_3` | No | — | Third Telegram bot token (for multi-bot) |
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key |
| `LLM_MODEL` | No | `openai/gpt-4o` | LLM model identifier |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_companion.db` | Database connection |
| `ALLOWED_USERS` | No | — | Comma-separated Telegram user IDs |
| `TIMEZONE` | No | `UTC` | IANA timezone |
| `DEBUG` | No | `false` | Enable debug mode |

---

## Required Settings

### `TELEGRAM_BOT_TOKEN`

Your primary Telegram Bot API token from [@BotFather](https://t.me/BotFather).

```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
```

**How to get:**
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`
3. Follow the prompts to create your bot
4. Copy the token provided

---

### `OPENROUTER_API_KEY`

Your API key from [OpenRouter](https://openrouter.ai).

```bash
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**How to get:**
1. Create an account at [openrouter.ai](https://openrouter.ai)
2. Add credits (pay-as-you-go)
3. Generate an API key in your dashboard

---

## Multi-Bot Settings

### `TELEGRAM_BOT_TOKEN_2` / `TELEGRAM_BOT_TOKEN_3`

Additional Telegram bot tokens for running multiple bots simultaneously. Each bot acts as a separate "window" where a human can create an independent companion.

```bash
TELEGRAM_BOT_TOKEN_2=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_BOT_TOKEN_3=0987654321:ZYXwvuTSRqpoNMLkjiHGFedcBA
```

**Default:** Empty (only the primary bot runs)

When configured, the application starts a separate `TelegramMessenger` and `BotHandler` for each token. Each companion is identified by a composite ID (`user_id@bot_username`), ensuring full isolation between bots.

See [GETTING_STARTED.md](GETTING_STARTED.md#multi-bot-setup-optional) for a detailed setup guide.

---

## LLM Settings

### `LLM_MODEL`

The language model to use. Must be a valid OpenRouter model identifier.

```bash
LLM_MODEL=openai/gpt-4o
```

**Recommended models:**

| Model | Cost | Quality | Speed |
|-------|------|---------|-------|
| `openai/gpt-4o` | $$ | Excellent | Fast |
| `anthropic/claude-3.5-sonnet` | $$ | Excellent | Fast |
| `anthropic/claude-3-opus` | $$$ | Best | Slower |
| `meta-llama/llama-3.1-70b-instruct` | $ | Good | Fast |
| `google/gemini-pro-1.5` | $$ | Very Good | Fast |

Check [OpenRouter models](https://openrouter.ai/models) for current pricing and availability.

---

## Database Settings

### `DATABASE_URL`

SQLAlchemy async database URL.

```bash
DATABASE_URL=sqlite+aiosqlite:///./data/mai_companion.db
```

**Default:** SQLite database in the `data/` directory.

**Note:** Only SQLite is currently tested. PostgreSQL may work but is not officially supported.

---

### `CHROMA_PERSIST_DIR`

Directory for ChromaDB vector store persistence.

```bash
CHROMA_PERSIST_DIR=./data/chroma_data
```

**Default:** `./data/chroma_data`

Used for semantic search across conversation history.

---

## Memory Settings

### `MEMORY_DATA_DIR`

Base directory for wiki and summary files.

```bash
MEMORY_DATA_DIR=./data
```

**Default:** `./data`

Wiki entries and summaries are stored as markdown files under this directory.

---

### `SUMMARY_THRESHOLD`

Minimum number of same-day messages before daily summarization triggers.

```bash
SUMMARY_THRESHOLD=20
```

**Default:** `20`

Lower values = more frequent summaries (more LLM calls, higher cost).
Higher values = less frequent summaries (may miss important details).

---

### `WIKI_CONTEXT_LIMIT`

Maximum number of wiki entries to include in the prompt context.

```bash
WIKI_CONTEXT_LIMIT=20
```

**Default:** `20`

Entries are sorted by importance; top N are included.

---

### `SHORT_TERM_LIMIT`

Number of recent messages to include in short-term context.

```bash
SHORT_TERM_LIMIT=30
```

**Default:** `30`

Higher values = more context but more tokens used.

---

### `TOOL_MAX_ITERATIONS`

Maximum agentic tool-calling iterations per response.

```bash
TOOL_MAX_ITERATIONS=5
```

**Default:** `5`

Limits how many tool calls the AI can make in a single response cycle.

---

## Access Control

### `ALLOWED_USERS`

Comma-separated list of Telegram user IDs allowed to use the AI.

```bash
ALLOWED_USERS=123456789,987654321
```

**Default:** Empty (anyone can use the AI — not recommended for production)

**How to get your user ID:**
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your user ID

**Security note:** Without this setting, anyone who finds your bot can use it and incur API costs.

---

## Time Settings

### `TIMEZONE`

IANA timezone for proactive messaging and quiet hours.

```bash
TIMEZONE=America/New_York
```

**Default:** `UTC`

**Examples:**
- `America/New_York`
- `Europe/London`
- `Asia/Tokyo`
- `Australia/Sydney`

See [IANA timezone list](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

---

### `QUIET_HOURS_START`

Hour (0-23) when quiet hours begin. The AI won't send proactive messages during quiet hours.

```bash
QUIET_HOURS_START=23
```

**Default:** `23` (11 PM)

---

### `QUIET_HOURS_END`

Hour (0-23) when quiet hours end.

```bash
QUIET_HOURS_END=7
```

**Default:** `7` (7 AM)

---

## Debug Settings

### `DEBUG`

Enable debug mode for verbose logging and SQL echo.

```bash
DEBUG=true
```

**Default:** `false`

When enabled:
- Verbose logging output
- SQL queries are logged
- Additional diagnostic information

**Warning:** Don't enable in production — logs may contain sensitive data.

---

### `LOG_LEVEL`

Python logging level.

```bash
LOG_LEVEL=INFO
```

**Default:** `INFO`

**Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

---

## Example `.env` File

```bash
# === Required ===
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# === Multi-Bot (Optional) ===
# Additional bots for multiple companions per human
TELEGRAM_BOT_TOKEN_2=
TELEGRAM_BOT_TOKEN_3=

# === Recommended ===
# Restrict access to your Telegram user ID
ALLOWED_USERS=123456789

# Set your timezone
TIMEZONE=America/New_York

# === Optional ===
# Choose a different model
LLM_MODEL=anthropic/claude-3.5-sonnet

# Adjust memory settings
SUMMARY_THRESHOLD=15
SHORT_TERM_LIMIT=40
WIKI_CONTEXT_LIMIT=25

# Quiet hours (no proactive messages)
QUIET_HOURS_START=22
QUIET_HOURS_END=8

# Debug mode (development only)
DEBUG=false
LOG_LEVEL=INFO
```

---

## Docker Configuration

When using Docker, settings can be passed via:

### 1. `.env` file (Recommended)

```yaml
# docker-compose.yml
services:
  mai-companion:
    env_file:
      - .env
```

### 2. Environment section

```yaml
# docker-compose.yml
services:
  mai-companion:
    environment:
      - TELEGRAM_BOT_TOKEN=your_token
      - OPENROUTER_API_KEY=your_key
```

### 3. Docker run

```bash
docker run -e TELEGRAM_BOT_TOKEN=your_token \
           -e OPENROUTER_API_KEY=your_key \
           mai-companion
```

---

## Volume Mounts

For data persistence with Docker:

```yaml
# docker-compose.yml
services:
  mai-companion:
    volumes:
      - ./data:/app/data
```

This ensures:
- Database persists across container restarts
- Wiki and summary files are accessible
- Backups can be made from the host

---

## Configuration Precedence

Settings are loaded in this order (later overrides earlier):

1. Default values in code
2. `.env` file
3. Environment variables
4. Command-line arguments (if applicable)

---

## Validation

On startup, mAI Companion validates:

- Required settings are present
- Values are within valid ranges
- Database is accessible
- API keys are formatted correctly

Invalid configuration will prevent startup with a descriptive error message.

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) — Setup guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical overview
- [DEVELOPMENT.md](DEVELOPMENT.md) — Development settings
