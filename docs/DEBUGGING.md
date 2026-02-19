# Debugging

> **How to inspect, debug, and troubleshoot your companion's behavior.**

Whether your companion is running via Telegram or Console mode, the `mai-chat` CLI provides powerful inspection tools that let you see exactly what's happening behind the scenes — conversation history, wiki entries, memory summaries, the full LLM prompt, and detailed tool call logs.

---

## Finding Your Chat ID

Every companion is identified by a **chat ID**. For Telegram, this is your Telegram user ID (the same number shown in your `ALLOWED_USERS` config).

To find it, check the application logs when you send a message:

```
mai_companion.bot.middleware - INFO - Incoming message: platform=telegram chat=186215217 ...
```

The number after `chat=` is your chat ID. You'll use this with all `mai-chat` inspection commands.

---

## Inspecting Telegram Conversations

The `mai-chat` CLI can inspect data from **any** companion — including those running via Telegram. All inspection commands use the `-c` flag to specify the chat ID.

### View Conversation History

See all messages exchanged between the human and the companion:

```bash
mai-chat -c <chat-id> --history
```

Example output:

```
=== History: 186215217 ===
[2026-02-09 17:12:40] ASSISTANT: Здарова. Чё как сам?
[2026-02-09 17:12:50] USER: Норм
[2026-02-09 17:12:52] ASSISTANT: Это хорошо, братан. Чё мутишь там?
```

### View Wiki Entries

Check what the companion has stored in its long-term knowledge base:

```bash
mai-chat -c <chat-id> --wiki
```

Example output:

```
=== Wiki: 186215217 ===
- (9999) human_name: Alexander
- (8000) favorite_food: pizza
- (5000) hobby: programming
```

If the output shows `(no wiki entries)`, the companion hasn't saved any knowledge yet — even if it *said* it remembered something.

### View Memory Summaries

See the daily, weekly, and monthly conversation summaries:

```bash
mai-chat -c <chat-id> --summaries
```

### Replay Conversation with Tool Events

The replay view shows messages interleaved with tool calls (wiki writes, searches, etc.), giving you a complete picture of what happened:

```bash
# Replay all conversations
mai-chat -c <chat-id> --replay

# Replay a specific date
mai-chat -c <chat-id> --replay --date 2026-02-19
```

> **Note:** Tool events only appear in replay when the conversation was run with `--debug` mode enabled (see [Debug Logging](#debug-logging) below).

### Inspect the LLM Prompt

See exactly what the LLM receives — the full system prompt, personality, wiki context, memory summaries, and recent messages:

```bash
mai-chat -c <chat-id> --show-prompt
```

This is invaluable for understanding why the companion behaves a certain way. The output includes:

- The system prompt (personality, rules, ethical boundaries)
- Current date/time context
- Wiki entries ("Things you know")
- Memory summaries ("Your memories")
- Recent conversation messages
- Approximate token count

---

## Debug Logging

### Console Mode Debug Logging

When using `mai-chat` interactively, add `--debug` to capture full LLM request/response details:

```bash
mai-chat -c <chat-id> --debug "Hello, how are you?"
```

This creates structured JSONL log files in `data/debug_logs/<chat-id>/` with:

- Full LLM request (messages, tools, temperature)
- Full LLM response (content, tool calls, token usage)
- Tool execution results (wiki writes, message searches)
- Cost estimates per call and per session

After the response, a debug summary is printed:

```
--- Debug Info ---
LLM calls: 2 (1 with tool calls)
Tools used: wiki_write
Tokens: 1,234 prompt + 56 completion = 1,290 total

--- Session Cost ---
This call: 1,290 tokens ($0.002)
Session total: 1,290 tokens ($0.002)
Full log: data/debug_logs/186215217/2026-02-19.jsonl
```

### Reading Debug Logs

Debug logs are stored as JSONL (one JSON object per line):

```bash
# View today's debug log
cat data/debug_logs/<chat-id>/2026-02-19.jsonl | python -m json.tool

# Search for tool calls
grep '"entry_type": "tool_result"' data/debug_logs/<chat-id>/2026-02-19.jsonl
```

Each entry has an `entry_type` field:

| Entry Type | Description |
|------------|-------------|
| `llm_call` | Full LLM request and response |
| `tool_result` | Result of a tool execution (wiki write, search, etc.) |

### Application-Level Debug Mode

For the Telegram bot, enable verbose logging:

```bash
DEBUG=true python -m mai_companion.main
```

This enables:

- SQL query logging
- Verbose internal logging
- Additional diagnostic output

You can also adjust the log level independently:

```bash
LOG_LEVEL=DEBUG python -m mai_companion.main
```

---

## Common Debugging Scenarios

### "The companion said it remembered, but didn't"

1. Check wiki entries:
   ```bash
   mai-chat -c <chat-id> --wiki
   ```
2. If empty, the LLM didn't actually call the wiki tool — it just *said* it remembered. This can happen with some models that "pretend" instead of using tools.

### "The companion's responses seem wrong"

Inspect the full prompt to see what context the LLM receives:

```bash
mai-chat -c <chat-id> --show-prompt
```

Check for:
- Missing or incorrect wiki entries
- Outdated summaries
- Token budget truncation warnings in the logs

### "Messages aren't being delivered via Telegram"

Check the application logs for timeout or network errors:

```
mai_companion.messenger.telegram - WARNING - Failed to send typing indicator: Timed out
mai_companion.messenger.telegram - ERROR - Failed to send Telegram message: Timed out
```

Common causes:
- Network instability (DNS resolution failures)
- Telegram API rate limiting
- Bot token issues

### "I want to see what happened on a specific day"

```bash
mai-chat -c <chat-id> --replay --date 2026-02-19
```

---

## Data Directory Structure

Understanding where data lives helps with manual inspection:

```
data/
├── mai_companion.db              # SQLite database (messages, companions, moods, wiki metadata)
├── chroma_data/                  # Vector store for semantic search
├── debug_logs/
│   └── <chat-id>/
│       ├── 2026-02-19.jsonl      # Debug logs for that date
│       └── ...
└── <chat-id>/
    ├── wiki/                     # Knowledge base entries (markdown files)
    │   ├── 9999_human-name.md
    │   └── 5000_favorite-food.md
    └── summaries/
        ├── daily/                # Daily conversation summaries
        │   └── 2026-02-19.md
        ├── weekly/               # Weekly rollups
        └── monthly/              # Monthly rollups
```

Wiki files are named `<importance>_<key>.md` where importance ranges from 0 (low) to 9999 (critical).

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `mai-chat -c ID --history` | View conversation history |
| `mai-chat -c ID --wiki` | View wiki entries |
| `mai-chat -c ID --summaries` | View memory summaries |
| `mai-chat -c ID --replay` | Replay with tool events |
| `mai-chat -c ID --replay --date YYYY-MM-DD` | Replay specific date |
| `mai-chat -c ID --show-prompt` | Inspect full LLM prompt |
| `mai-chat -c ID --debug "message"` | Send with debug logging |

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) — Installation and first conversation
- [DEVELOPMENT.md](DEVELOPMENT.md) — Development setup and guidelines
- [CONFIGURATION.md](CONFIGURATION.md) — All settings
- [MEMORY.md](MEMORY.md) — How the memory system works
- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical overview
