# Debugging

## Console CLI (`mai-chat`)

The `mai-chat` CLI is the primary debugging tool. It runs the full
BotHandler pipeline locally — same code path as Telegram, but output
goes to stdout. **This is designed for AI agents** that only have
terminal access and need to verify Telegram-facing behaviour.

Console output preserves raw HTML tags and parse mode annotations so
you can see exactly what would be sent to the Telegram API.

### Creating a test chat

Create a new chat with model and prompt in a single command:

```bash
mai-chat -c test-myfeature --start --model google/gemma-4-31b-it:free --prompt coder
```

Or use the callback-based flow (equivalent):

```bash
mai-chat -c test-myfeature --start \
  --cb "model:google/gemma-4-31b-it:free" \
  --cb "prompt:coder"
```

**Important:** The `--start` command and model/prompt callbacks must
be in the **same invocation**. Setup sessions are in-memory and do not
survive across separate process runs. Once a chat is created, the model
and prompt are persisted in the database — subsequent calls only need
the chat ID.

### Dispatch slash commands

Use `--command` to exercise the real command handler path from the CLI:

```bash
mai-chat -c test-myfeature --command help
mai-chat -c test-myfeature --command "timezone Europe/Moscow"
mai-chat -c test-myfeature --command reasoning
```

### Sending messages

```bash
mai-chat -c test-myfeature "Hello, world!"
mai-chat -c test-myfeature --debug "Tell me about yourself"
```

The `-c` flag is remembered between invocations, so after the first
use you can omit it:

```bash
mai-chat "Follow-up message"
```

### Streaming output

By default, only the **final version** of the AI response is printed.
Intermediate streaming edits (the partial token-by-token updates that
Telegram shows as "typing") are suppressed to reduce noise.

To see every streaming edit (useful for debugging the streaming
behaviour itself):

```bash
mai-chat -c test-myfeature --stream-debug "Hello"
```

### Show assembled prompt

```bash
mai-chat -c test-myfeature --show-prompt
```

Shows the system prompt, wiki entries, available tools, and message
history as they would be sent to the LLM. Useful for verifying that
context is assembled correctly.

### Debug mode (LLM logging)

```bash
mai-chat -c test-myfeature --debug "Hello"
```

Logs all LLM calls to `data/debug_logs/<chat_id>/`. Each call is a
JSONL entry with:
- Full request (messages, tools, model, temperature)
- Full response (content, reasoning, tool calls, token usage)
- Both streaming (`llm_stream_call`) and non-streaming (`llm_call`) calls are logged

### Inspect history and wiki

```bash
mai-chat -c test-myfeature --history
mai-chat -c test-myfeature --wiki
mai-chat --list
```

### Import a dialogue

```bash
mai-chat -c test-myfeature --import-json path/to/messages.json
```

Supports OpenAI chat format (`[{role, content, ...}]`) and AI Proxy v2
request JSON.

### Available flags

| Flag | Description |
|------|-------------|
| `-c ID` | Set chat ID (remembered for future runs) |
| `--start` | Run /start setup flow |
| `--command CMD` | Dispatch a slash command (`name` or `name args...`) |
| `--model MODEL` | Model ID for setup (use with `--start`) |
| `--prompt NAME` | Prompt template name for setup (use with `--start`) |
| `--cb DATA` | Dispatch a callback (button press), repeatable |
| `--debug` | Enable LLM call logging to JSONL |
| `--stream-debug` | Show every streaming edit (verbose) |
| `--show-prompt` | Print the assembled LLM prompt |
| `--history` | Show conversation history |
| `--wiki` | Show wiki entries |
| `--repair-wiki` | Sync wiki DB from disk files (see below) |
| `--list` | List all chats with message counts |
| `--import-json PATH` | Import dialogue from JSON file |
| `--real` | Disable test mode transparency notice |
| `--user-id ID` | Override synthetic user ID |

### Per-prompt tool filtering

When a prompt's `.toml` config restricts tools or MCP servers, those
restrictions apply in the console CLI the same way as in Telegram. To
verify which tools are active for a prompt, use `--show-prompt` to
inspect the assembled context including the tool list.

### Test companion naming

Test companions must use prefixes: `test-`, `func-`, `manual-`, or
`demo-`. Never use numeric IDs (those are real companions).

```bash
# Good
mai-chat -c test-streaming --start --model google/gemma-4-31b-it:free --prompt default
# Bad — this is a real user's companion
mai-chat -c 186215217 "Hello"
```

## Wiki Troubleshooting

Wiki entries are stored as `.md` files on disk and indexed in the
database. The files are the source of truth — see
[DEVELOPMENT.md](DEVELOPMENT.md#wiki-knowledge-base-architecture) for
the full architecture.

### "wiki_list shows 0 entries" but files exist

This means the database index is out of sync. Run:

```bash
mai-chat -c <chat_id> --repair-wiki
```

This scans all `.md` files in `data/<chat_id>/wiki/` and rebuilds the
database rows. It reports exactly what changed (created, updated,
deleted).

### Restoring wiki from backup

1. Copy the `.md` files back into `data/<chat_id>/wiki/`
2. Run `mai-chat -c <chat_id> --repair-wiki`
3. The database will be rebuilt from the files

### Manually editing wiki entries

You can edit `.md` files directly with any text editor. Changes are
picked up automatically on the next message (via `sync_from_disk`), or
immediately with `--repair-wiki`.

To change importance, rename the file prefix (e.g. `3000_name.md` →
`9999_name.md`).

To delete an entry, delete the `.md` file. The orphaned DB row is
cleaned up on the next sync.

## Diagnosing Slow or Hung Responses

### The bot is silent after receiving a message

Check the logs for:

1. `LLM stream started (model=..., messages=N)` — confirms the LLM
   request was sent. If this line is missing, the problem is before the
   LLM call (DB, wiki sync, MCP server startup).
2. A gap between "LLM stream started" and the next log line — the LLM
   provider or upstream proxy is slow/hung. The read timeout is 45
   seconds, after which it retries.
3. `Network error during stream (attempt N/3)` — the request timed out
   and is being retried.
4. `Usage data: ...` — the LLM response was received. If this never
   appears, the provider is down.

### The bot won't stop on Ctrl+C

During shutdown, the application must wait for any in-flight LLM
requests to complete (or time out). Look for:

```
INFO - Waiting for N in-flight LLM request(s) to complete...
```

The maximum wait is bounded by the read timeout (45 seconds). If the
provider is completely unresponsive, the worst case is 45s × 3 attempts
= ~135 seconds before the request gives up and shutdown proceeds.

A second Ctrl+C (SIGINT) will trigger a less graceful shutdown — the
`finally` block runs but pending HTTP connections are abandoned.

## Custom OpenRouter URL

For request-level debugging, route through a local proxy:

```bash
OPENROUTER_BASE_URL=http://localhost:8080/v1 python -m mai_gram.main
```

## Debug Mode (env)

Set `DEBUG=true` in `.env` to enable:
- SQLAlchemy echo (SQL query logging)
- Verbose logging output
