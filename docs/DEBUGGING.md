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
| `--model MODEL` | Model ID for setup (use with `--start`) |
| `--prompt NAME` | Prompt template name for setup (use with `--start`) |
| `--cb DATA` | Dispatch a callback (button press), repeatable |
| `--debug` | Enable LLM call logging to JSONL |
| `--stream-debug` | Show every streaming edit (verbose) |
| `--show-prompt` | Print the assembled LLM prompt |
| `--history` | Show conversation history |
| `--wiki` | Show wiki entries |
| `--list` | List all chats with message counts |
| `--import-json PATH` | Import dialogue from JSON |
| `--real` | Disable test mode transparency notice |
| `--user-id ID` | Override synthetic user ID |

### Test companion naming

Test companions must use prefixes: `test-`, `func-`, `manual-`, or
`demo-`. Never use numeric IDs (those are real companions).

```bash
# Good
mai-chat -c test-streaming --start --model google/gemma-4-31b-it:free --prompt default
# Bad — this is a real user's companion
mai-chat -c 186215217 "Hello"
```

## Custom OpenRouter URL

For request-level debugging, route through a local proxy:

```bash
OPENROUTER_BASE_URL=http://localhost:8080/v1 python -m mai_gram.main
```

## Debug Mode (env)

Set `DEBUG=true` in `.env` to enable:
- SQLAlchemy echo (SQL query logging)
- Verbose logging output
