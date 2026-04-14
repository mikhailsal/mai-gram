# Configuration

All settings are loaded from environment variables or a `.env` file.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | - | Primary bot token |
| `TELEGRAM_BOT_TOKEN_2` | No | - | Second bot token |
| `TELEGRAM_BOT_TOKEN_3` | No | - | Third bot token |
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | API base URL |
| `LLM_MODEL` | No | `openai/gpt-4o-mini` | Default model |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/mai_gram.db` | Database URL |
| `MEMORY_DATA_DIR` | No | `./data` | Base directory for wiki files |
| `WIKI_CONTEXT_LIMIT` | No | `20` | Max wiki entries in context |
| `SHORT_TERM_LIMIT` | No | `50` | Recent messages in context |
| `TOOL_MAX_ITERATIONS` | No | `5` | Max tool-calling rounds per response |
| `ALLOWED_USERS` | No | - | Comma-separated Telegram user IDs |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `DEBUG` | No | `false` | Debug mode |

## Model Whitelist

Edit `config/models.toml` to control which models users can select:

```toml
[models]
allowed = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
]
default = "openai/gpt-4o-mini"
```

## System Prompt Templates

Place `.txt` or `.md` files in the `prompts/` directory. During `/start`, users see these as selectable options alongside "Custom (type your own)".

## Custom OpenRouter URL

Set `OPENROUTER_BASE_URL` to route API calls through a local proxy:

```bash
OPENROUTER_BASE_URL=http://localhost:8080/v1
```
