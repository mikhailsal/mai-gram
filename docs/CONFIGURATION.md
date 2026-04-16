# Configuration

All settings are loaded from environment variables or a `.env` file.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Legacy | - | Primary bot token (ignored if `config/bots.toml` exists) |
| `TELEGRAM_BOT_TOKEN_2` | Legacy | - | Second bot token (ignored if `config/bots.toml` exists) |
| `TELEGRAM_BOT_TOKEN_3` | Legacy | - | Third bot token (ignored if `config/bots.toml` exists) |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Default model |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_gram.db` | Database URL |
| `MEMORY_DATA_DIR` | No | `./data` | Base directory for wiki files and companion data |
| `WIKI_CONTEXT_LIMIT` | No | `20` | Max wiki entries in context |
| `SHORT_TERM_LIMIT` | No | `50` | Recent messages in context |
| `TOOL_MAX_ITERATIONS` | No | `5` | Max tool-calling rounds per response |
| `ALLOWED_USERS` | No | - | Comma-separated Telegram user IDs (global fallback) |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Debug mode |

## Multi-Bot Setup (config/bots.toml)

For running multiple bots with per-bot restrictions, create
`config/bots.toml` (gitignored for security):

```bash
cp config/bots.toml.example config/bots.toml
```

When `config/bots.toml` exists, legacy `TELEGRAM_BOT_TOKEN*` env vars
are ignored. See `README.md` for the full format.

## Data Directory Layout

The `MEMORY_DATA_DIR` (default `./data`) contains all per-companion
data:

```
data/
├── mai_gram.db                              # SQLite database
├── <chat_id>/
│   └── wiki/
│       ├── 9999_human_name.md               # Wiki entry (importance_key.md)
│       ├── 7000_first_conversation.md
│       └── changelog.jsonl                  # Wiki change log
├── debug_logs/<chat_id>/                    # LLM call logs (--debug mode)
└── .console_state.json                      # mai-chat last-used chat ID
```

**Important:** The `.md` files in `wiki/` are the source of truth for
the knowledge base — not the database. If the database is lost, the wiki
can be fully rebuilt from the files using `mai-chat --repair-wiki`. See
[DEVELOPMENT.md](DEVELOPMENT.md#wiki-knowledge-base-architecture) for
details.

## Model Whitelist

Edit `config/models.toml` to control which models users can select:

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

Changes to `config/models.toml` are picked up automatically without
restarting (mtime-based polling every 2 seconds).

## System Prompt Templates

Place `.txt` or `.md` files in the `prompts/` directory. During `/start`,
users see these as selectable options alongside "Custom (type your own)".

## LLM Timeout Configuration

The LLM client uses granular timeouts to protect against hung upstream
providers:

| Phase | Timeout | What it means |
|-------|---------|---------------|
| Connect | 10s | TCP connection to the API server |
| Read | 45s | Waiting for first token or between bytes |
| Write | 10s | Sending the request body |
| Pool | 10s | Waiting for a free connection |

If the upstream provider accepts the connection but never responds, the
request fails after **45 seconds** and is retried (up to 3 total
attempts). This prevents the old scenario where a hung provider could
block the bot for 2+ minutes.

## Custom OpenRouter URL

Set `OPENROUTER_BASE_URL` to route API calls through a local proxy:

```bash
OPENROUTER_BASE_URL=http://localhost:8080/v1
```
