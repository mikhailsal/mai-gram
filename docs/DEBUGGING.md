# Debugging

## Console CLI

The `mai-chat` CLI is the primary debugging tool.

### Show assembled prompt

```bash
mai-chat -c test-mychat --show-prompt
```

Shows the system prompt, wiki entries, available tools, and message history as they would be sent to the LLM.

### Debug mode

```bash
mai-chat -c test-mychat --debug "Hello"
```

Logs all LLM calls to `data/debug_logs/<chat_id>/`. Each call is a JSONL entry with:
- Full request (messages, tools)
- Full response (content, tool calls, token usage)
- Timing and cost information

### Inspect history and wiki

```bash
mai-chat -c test-mychat --history
mai-chat -c test-mychat --wiki
mai-chat --list
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
