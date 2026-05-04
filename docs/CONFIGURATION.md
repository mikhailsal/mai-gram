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

## Model Registry

Edit `config/models.toml` to control which models users can select.
Each `[models."<key>"]` section defines one selectable model:

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
```

Per-model meta-keys (not sent to the LLM API):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Show this model in the `/start` selection UI |
| `title` | string | derived from key | Custom display label (e.g. `"Gemini Flash @high"`) |
| `id` | string | section key | Real OpenRouter model ID when the key is an alias |

All other keys (`temperature`, `reasoning.effort`, `provider.order`, etc.)
are merged into the OpenRouter request body as parameter overrides.

To register the same model with different parameter presets, use a unique
alias key and point `id` to the real model:

```toml
[models."flash-creative"]
id = "google/gemini-2.5-flash"
title = "Gemini Flash (creative)"
temperature = 1.5

[models."flash-precise"]
id = "google/gemini-2.5-flash"
title = "Gemini Flash (precise)"
reasoning.effort = "high"
temperature = 0.2
```

Changes to `config/models.toml` are picked up automatically without
restarting (mtime-based polling every 2 seconds).

## System Prompt Templates

Place `.txt` or `.md` files in the `prompts/` directory. During `/start`,
users see these as selectable options alongside "Custom (type your own)".

### Per-Prompt Configuration (TOML)

Each prompt template can have a companion `.toml` file with the same
base name (e.g. `coder.txt` + `coder.toml`). This TOML file controls
display defaults and tool availability for chats using that prompt.

```toml
# prompts/coder.toml
[display]
show_reasoning = true
show_tool_calls = true
send_datetime = false

[tools]
# disabled = ["wiki_search"]   # blacklist specific tools

[mcp_servers]
# disabled = ["messages"]      # blacklist MCP server groups
```

Available settings:

| Section | Key | Type | Description |
|---------|-----|------|-------------|
| `[display]` | `show_reasoning` | bool | Show LLM reasoning to the user (default: true) |
| `[display]` | `show_tool_calls` | bool | Show tool call messages (default: true) |
| `[display]` | `send_datetime` | bool | Include timestamps in messages sent to the LLM |
| `[tools]` | `enabled` | list | Whitelist — only these tools are available |
| `[tools]` | `disabled` | list | Blacklist — these tools are hidden |
| `[mcp_servers]` | `enabled` | list | Whitelist for MCP server groups (`messages`, `wiki`) |
| `[mcp_servers]` | `disabled` | list | Blacklist for MCP server groups |

When both `enabled` and `disabled` are set, `enabled` takes precedence.
Per-prompt config overrides global config when present. Users can still
toggle display settings at runtime with `/reasoning` and `/toolcalls`.

### Bundled Prompt Templates

| Template | Description |
|----------|-------------|
| `default` | General-purpose AI companion with full tool access |
| `coder` | Programming assistant with reasoning and tool calls visible |
| `creative` | Creative writing — message history search disabled |
| `independent` | Companion mode — reasoning and tool calls hidden |
| `english-teacher` | Language learning via translation exercises with wiki-based progress tracking |

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

## Dialogue Import

mai-gram can import conversation history from other AI tools in two ways:

### Via Telegram (`/import` command)

1. Send `/import` to the bot
2. Select an LLM model from the inline keyboard
3. Upload a `.json` file with the conversation
4. Messages are replayed into the chat with formatting and rate limiting

### Via CLI

```bash
mai-chat -c test-demo --import-json path/to/messages.json
```

### Supported Formats

- **OpenAI chat format**: `[{role, content, timestamp?, reasoning?, tool_calls?}]`
- **AI Proxy v2 request JSON**: automatically detected and converted

Imported messages receive synthetic timestamps (not the original dates)
and are marked as imported — the AI sees `[imported, real date unknown]`
instead of potentially misleading original timestamps.

## Custom OpenRouter URL

Set `OPENROUTER_BASE_URL` to route API calls through a local proxy:

```bash
OPENROUTER_BASE_URL=http://localhost:8080/v1
```
