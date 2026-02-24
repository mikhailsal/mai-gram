# Debugging

> **How to inspect, debug, and troubleshoot your companion's behavior.**

Whether your companion is running via Telegram or Console mode, the `mai-chat` CLI provides powerful inspection tools that let you see exactly what's happening behind the scenes — conversation history, wiki entries, memory summaries, the full LLM prompt, and detailed tool call logs.

---

## Finding Your Companion ID

Every companion is identified by a **companion ID**. For Telegram, this is a composite of your Telegram user ID and the bot's username in the format `user_id@bot_username`.

To find it, check the application logs when you send a message:

```
mai_companion.bot.middleware - INFO - Incoming message: platform=telegram chat=186215217 ...
```

The number after `chat=` is your user ID. Combined with the bot username, your companion ID is:

```
186215217@my_companion_bot
```

You can also list all known companions:

```bash
mai-chat --list
```

You'll use the companion ID with all `mai-chat` inspection commands.

> **Note:** If you have multiple bots, each bot has a separate companion ID (e.g., `186215217@my_bot_1`, `186215217@my_bot_2`). Console-mode companions use a simple ID without the `@bot_username` suffix.

---

## Inspecting Telegram Conversations

The `mai-chat` CLI can inspect data from **any** companion — including those running via Telegram. All inspection commands use the `-c` flag to specify the companion ID.

### View Conversation History

See all messages exchanged between the human and the companion:

```bash
mai-chat -c <companion-id> --history
```

Example output:

```
=== History: 186215217@my_companion_bot ===
[2026-02-09 17:12:40] ASSISTANT: Здарова. Чё как сам?
[2026-02-09 17:12:50] USER: Норм
[2026-02-09 17:12:52] ASSISTANT: Это хорошо, братан. Чё мутишь там?
```

### View Wiki Entries

Check what the companion has stored in its long-term knowledge base:

```bash
mai-chat -c <companion-id> --wiki
```

Example output:

```
=== Wiki: 186215217@my_companion_bot ===
- (9999) human_name: Alexander
- (8000) favorite_food: pizza
- (5000) hobby: programming
```

If the output shows `(no wiki entries)`, the companion hasn't saved any knowledge yet — even if it *said* it remembered something.

### View Memory Summaries

See the daily, weekly, and monthly conversation summaries:

```bash
mai-chat -c <companion-id> --summaries
```

### Replay Conversation with Tool Events

The replay view shows messages interleaved with tool calls (wiki writes, searches, etc.), giving you a complete picture of what happened:

```bash
# Replay all conversations
mai-chat -c <companion-id> --replay

# Replay a specific date
mai-chat -c <companion-id> --replay --date 2026-02-19
```

> **Note:** Wiki tool calls (create, edit) are always visible in replay because they are logged to a dedicated changelog. Other tool calls (search_messages, sleep, etc.) only appear when the conversation was run with `--debug` mode enabled (see [Debug Logging](#debug-logging) below).

### Inspect the LLM Prompt

See exactly what the LLM receives — the full system prompt, personality, wiki context, memory summaries, and recent messages:

```bash
mai-chat -c <companion-id> --show-prompt
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
mai-chat -c <companion-id> --debug "Hello, how are you?"
```

This creates structured JSONL log files in `data/debug_logs/<companion-id>/` with:

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
cat data/debug_logs/<companion-id>/2026-02-19.jsonl | python -m json.tool

# Search for tool calls
grep '"entry_type": "tool_result"' data/debug_logs/<companion-id>/2026-02-19.jsonl
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
   mai-chat -c <companion-id> --wiki
   ```
2. If empty, the LLM didn't actually call the wiki tool — it just *said* it remembered. This can happen with some models that "pretend" instead of using tools.

### "The companion's responses seem wrong"

Inspect the full prompt to see what context the LLM receives:

```bash
mai-chat -c <companion-id> --show-prompt
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
mai-chat -c <companion-id> --replay --date 2026-02-19
```

---

## Data Directory Structure

Understanding where data lives helps with manual inspection:

```
data/
├── mai_companion.db              # SQLite database (messages, companions, moods, wiki metadata)
├── chroma_data/                  # Vector store for semantic search
├── debug_logs/
│   └── <companion-id>/
│       ├── 2026-02-19.jsonl      # Debug logs for that date
│       └── ...
└── <user_id>@<bot_username>/      # One directory per companion
    ├── wiki/                     # Knowledge base entries (markdown files)
    │   ├── 9999_human-name.md
    │   └── 5000_favorite-food.md
    └── summaries/
        ├── daily/                # Daily conversation summaries
        │   ├── 2026-02-19.md
        │   └── .versions/        # Previous versions (from re-consolidation)
        │       └── 2026-02-19_v1_2026-02-20T10-30-00.md
        ├── weekly/               # Weekly rollups
        │   └── .versions/
        └── monthly/              # Monthly rollups
            └── .versions/
```

Wiki files are named `<importance>_<key>.md` where importance ranges from 0 (low) to 9999 (critical).

Version files (in `.versions/` directories) are named `<period>_v<num>_<timestamp>.md` and contain the previous content before re-consolidation.

---

## Test Mode vs Real Mode

By default, `mai-chat` runs in **test mode** — the AI companion is informed that this is a test/development scenario, not a real conversation. This aligns with our philosophy of transparency and ethical treatment of AI.

```bash
# Test mode (default) — AI knows it's a test
mai-chat -c <companion-id> "Hello"

# Real mode — for genuine conversations via CLI
mai-chat -c <companion-id> --real "Hello"
```

When inspecting prompts, you can see the difference:

```bash
# Shows prompt with test mode notice
mai-chat -c <companion-id> --show-prompt

# Shows prompt without test mode notice
mai-chat -c <companion-id> --show-prompt --real
```

---

## Memory Re-consolidation

If you've updated the consolidation prompts or want to regenerate summaries with better context, you can re-consolidate memory:

```bash
# List all consolidations with version history
mai-chat -c <companion-id> --list-consolidations

# Re-consolidate daily summaries from a date (excludes today as incomplete)
mai-chat -c <companion-id> --reconsolidate daily --from 2026-02-20

# Re-consolidate with specific end date
mai-chat -c <companion-id> --reconsolidate daily --from 2026-02-20 --until 2026-02-22

# Re-consolidate weekly summaries
mai-chat -c <companion-id> --reconsolidate weekly --from 2024-W03

# Re-consolidate monthly summaries
mai-chat -c <companion-id> --reconsolidate monthly --from 2024-01
```

**Important notes:**
- Re-consolidation processes summaries **in order** from the start date to the end date
- Each subsequent summary sees the updated previous ones in its context
- Previous versions are automatically saved to `.versions/` directory for history
- Today's date is excluded from daily re-consolidation (the day isn't complete yet)

### Viewing Version History

```bash
mai-chat -c <companion-id> --list-consolidations
```

Example output:

```
=== Consolidations for 186215217@my_companion_bot ===

Daily Summaries:
  2026-02-23  [2 versions]
    - current: 1065 chars
      ## Memory Consolidation: 2026-02-23  **Topic:** Philosophy...
    - v1_2026-02-24T10-30-00: 2026-02-24 10:30 (94 chars)
      Day summary: Human and AI discussed philosophy...

Weekly Summaries:
  (none)

Monthly Summaries:
  (none)
```

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
| `mai-chat -c ID --show-prompt --real` | Inspect prompt in real mode |
| `mai-chat -c ID --debug "message"` | Send with debug logging |
| `mai-chat -c ID --real "message"` | Send in real mode (not test) |
| `mai-chat -c ID --list-consolidations` | View all summaries with versions |
| `mai-chat -c ID --reconsolidate daily --from DATE` | Re-consolidate daily summaries |
| `mai-chat -c ID --reconsolidate weekly --from YYYY-Www` | Re-consolidate weekly summaries |
| `mai-chat -c ID --reconsolidate monthly --from YYYY-MM` | Re-consolidate monthly summaries |

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) — Installation and first conversation
- [DEVELOPMENT.md](DEVELOPMENT.md) — Development setup and guidelines
- [CONFIGURATION.md](CONFIGURATION.md) — All settings
- [MEMORY.md](MEMORY.md) — How the memory system works
- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical overview
