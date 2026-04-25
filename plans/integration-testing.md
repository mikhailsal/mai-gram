# Black-Box Integration Testing Plan

## Goal

Build a real black-box integration suite that launches the console application through `mai-chat`, sends real requests to the LLM provider, uses the free-model alias `openrouter/free`, and verifies the project through observable behavior only:

- process exit code
- stdout and stderr
- files created under the test data directory
- SQLite state written by the application
- debug logs produced by `--debug`

The suite is intended to protect future refactoring. It must catch behavioral regressions in the real `mai-chat -> ConsoleMessenger -> BotHandler -> PromptBuilder -> MCP/tools -> OpenRouter` path, not just mocked units.

## Implementation Checklist

Use this section as the working checklist while the plan is being implemented.

- [x] Phase 0 completed: CLI reachability for arbitrary commands exists
- [x] Phase 0 completed: subprocess helpers and temp-environment fixtures exist
- [x] Phase 0 completed: temp `models.toml` with `openrouter/free` is used by functional tests
- [x] Phase 1 completed: CLI smoke scenarios are implemented
- [x] Phase 1 completed: onboarding scenarios are implemented
- [x] Phase 1 completed: minimal real conversation scenario is implemented
- [x] Phase 1 completed: debug-log scenario is implemented
- [x] Phase 1 completed: history and prompt-preview scenarios are implemented
- [x] Phase 2 completed: wiki creation and recall scenarios are implemented
- [x] Phase 2 completed: repair-wiki scenarios are implemented
- [x] Phase 2 completed: import and replay scenarios are implemented
- [x] Phase 2 completed: access-control scenarios are implemented
- [x] Phase 3 completed: regenerate parity scenarios are implemented
- [x] Phase 3 completed: reset and backup scenarios are implemented
- [x] Phase 3 completed: cut-above scenarios are implemented
- [x] Phase 3 completed: visibility-toggle and timezone scenarios are implemented
- [x] Canonical-versus-rare behavior policy is encoded in test assertions
- [x] The full suite runs through the real `mai-chat` binary
- [x] The full suite uses real LLM calls through `openrouter/free`
- [x] The full suite uses isolated temp DB and temp memory directories

## What The Repository Already Tells Us

The current codebase already gives a strong outline for the test strategy:

- `mai-chat` is the correct black-box entrypoint. It is the documented debugging interface and runs the real BotHandler pipeline locally.
- The repository already has a `functional` pytest marker and skip logic for real-provider tests, but no actual implemented functional tests.
- The default CLI surface already covers `--start`, `--cb`, `--history`, `--wiki`, `--show-prompt`, `--debug`, `--stream-debug`, `--list`, `--import-json`, `--repair-wiki`, and plain message sending.
- The console CLI persists state across invocations via `data/.console_state.json`, so the suite must isolate filesystem state per test.
- Several important behaviors are duplicated in multiple places, especially the main generation flow versus regeneration flow. That duplication is exactly the kind of thing this suite should freeze.

## Contract Rule For Duplicated Behavior

This rule must be explicit in the test plan and in future implementation notes:

**When similar functionality exists in 2-3 places and behavior differs, the most frequent user scenario is the temporary source of truth.**

For now, the suite should treat the most common path as correct behavior and make rarer divergent paths fail until they are aligned.

Current examples:

- The normal conversation path is more important than the regenerate path. If `_handle_regenerate()` diverges from ordinary message handling, the ordinary message path defines the expected behavior.
- One-shot CLI onboarding with `--start --model --prompt` is the canonical setup path because it is repeated in README, Makefile, and docs. Callback-only setup should match it; if it does not, that is a failing test.
- CLI import via `--import-json` is the canonical black-box import contract for this suite. Telegram upload-specific details are secondary unless the console CLI gains equivalent reachability.

This rule avoids test paralysis while the code still contains duplicated branches.

## Hard Constraint Discovered During Study

Full black-box coverage is **not reachable yet** through the current CLI surface.

Today, `mai-chat` can dispatch `/start`, callbacks, plain text, and several inspection flags, but it cannot generically dispatch arbitrary commands such as:

- `/help`
- `/model`
- `/reset`
- `/timezone`
- `/datetime`
- `/reasoning`
- `/toolcalls`
- `/resend_last`
- `/import`

Typing `"/help"` as a plain message will not exercise the command handler, because the console runner sends it as text, not as a command event.

Because of that, the first implementation stage for the integration suite must be a small CLI reachability improvement:

1. Add a generic command-dispatch option such as `--command help` or `--slash "/help"`.
2. The command-dispatch option must support command arguments. For example, `/timezone Europe/Moscow` requires passing both the command name (`timezone`) and the argument (`Europe/Moscow`). The implementation must populate `IncomingMessage.command_args` so that handlers like `_handle_timezone` (which reads `message.command_args`) work correctly. A natural CLI syntax is `--command "timezone Europe/Moscow"` where the first token is the command and the rest is `command_args`.
3. Keep `--start` as shorthand for the existing onboarding path.
4. Continue to use `--cb` for callback-only flows.

Without this first step, the suite cannot honestly claim black-box coverage of all functionality.

## Test Philosophy

These tests must be black-box and real-provider, but they should still be stable.

That means:

- assert structure and persisted effects, not fragile full-text LLM snapshots
- use purpose-built temporary prompts where determinism matters
- keep messages short to reduce cost and rate-limit exposure
- use unique `test-...` chat IDs only
- isolate every test in its own temp database and temp memory directory
- prefer proving features through durable side effects such as history rows, wiki files, debug logs, and prompt previews

## Execution Model

### Test Harness

Implement a subprocess-based helper that always launches the real installed CLI:

```text
mai-chat [args...]
```

Every invocation must set `cwd` to the temp directory **and** provide an explicit environment:

- `cwd=<tmp>` (critical: `_ConsoleStateStore._STATE_FILE` is hardcoded to `./data/.console_state.json` relative to `cwd`, not controlled by any env var)
- `OPENROUTER_API_KEY=<real key>`
- `OPENROUTER_BASE_URL=<real provider base URL if needed>`
- `DATABASE_URL=sqlite+aiosqlite:///<tmp>/mai_gram.db`
- `MEMORY_DATA_DIR=<tmp>/data`
- `MODELS_CONFIG_PATH=<tmp>/models.toml`
- `PROMPTS_DIR=<tmp>/prompts`
- `BOTS_CONFIG_PATH=<tmp>/bots.toml` (set to a nonexistent or empty file to prevent the real `config/bots.toml` from being loaded when `cwd` matches the project root)
- `DEFAULT_TIMEZONE=UTC` (controls the initial timezone for new chats; set explicitly so tests don't inherit the host's system timezone, which would make timezone-related assertions unpredictable)
- optional `LOG_LEVEL=DEBUG`

To assist with debugging, these temporary directories must automatically be deleted if a test passes, but preserved if a test fails (with their paths printed to `stdout`).

The temporary `models.toml` for functional tests should whitelist only the alias required by this plan:

```toml
[models]
allowed = ["openrouter/free"]
default = "openrouter/free"
```

That keeps model selection deterministic and matches the required pseudo-model contract.

### Suggested Helper API

Create thin helpers around subprocess execution:

- `run_cli(*args, env, cwd, timeout=60) -> CompletedRun` (must enforce a strict timeout to prevent hangs if the API stalls)
- `start_chat(chat_id, prompt="default", model=FREE_MODEL, user_id=None)`
- `send_message(chat_id, text, user_id=None, ...)`
- `send_callback(chat_id, callback, user_id=None)`
- `run_command(chat_id, command, args=None, user_id=None)` once generic command dispatch exists (must support command arguments, e.g. `run_command(chat_id, "timezone", args="Europe/Moscow")`)
- `read_history(chat_id)`
- `read_wiki(chat_id)`
- `show_prompt(chat_id, real=False)`
- `list_chats()`
- `import_json(chat_id, path)`

All helpers that interact with a chat should accept an optional `user_id` parameter (forwarded to `--user-id`) to support access-control scenarios (Section 8). When omitted, the default user resolution logic applies.

Add output parsers for the console sections already emitted by the app:

- `--- AI Response ---`
- `--- AI Response (final edit of ...) ---`
- `--- Buttons ---`
- `--- Prompt Preview ---`
- `--- Available Tools ---`
- `--- Message Context ---`
- `--- Debug Info ---`
- `--- Session Cost ---`

### Observable Sources

The suite should use only these observation channels:

- stdout and stderr
- exit status
- `mai_gram.db`
- `data/<chat_id>/wiki/*.md`
- `data/debug_logs/<chat_id>/...jsonl`
- `data/backups/*.zip`
- `data/.console_state.json`

No direct imports from production code should be needed for assertions beyond small helpers for parsing raw artifacts.

## Proposed Test Layout

Suggested file structure for the future suite:

- `tests/functional/conftest.py`
- `tests/functional/helpers/cli.py`
- `tests/functional/helpers/parsing.py`
- `tests/functional/helpers/artifacts.py`
- `tests/functional/test_cli_smoke.py`
- `tests/functional/test_onboarding.py`
- `tests/functional/test_conversation.py`
- `tests/functional/test_tools_and_wiki.py`
- `tests/functional/test_prompt_and_visibility.py`
- `tests/functional/test_import_and_replay.py`
- `tests/functional/test_reset_and_regenerate.py`
- `tests/functional/test_command_flows.py`

All of them should be marked `@pytest.mark.functional` and remain opt-in behind the existing real-provider gating.

## Scenario Matrix

### 1. CLI Smoke And Environment Contract

- [x] This scenario group is implemented

Purpose: verify that the test harness is invoking the real console application correctly.

Scenarios:

1. `mai-chat --help` prints the expected options and exits successfully.
2. `mai-chat --list` on a fresh temp database prints `(no chats found)`.
3. Sending a message without any chat ID or remembered state fails with the expected guidance.
4. Running a real LLM action without `OPENROUTER_API_KEY` fails with the expected configuration error.
5. `.console_state.json` remembers the last chat ID between subprocess invocations.
6. Callback payloads outside an active setup session print the documented ignore hint.

### 2. Onboarding And Chat Creation

- [x] This scenario group is implemented

Purpose: freeze the canonical way a conversation is created.

Scenarios:

1. `mai-chat -c test-onboarding --start --model <FREE_MODEL> --prompt default` creates a chat and prints the creation summary.
2. `mai-chat -c test-onboarding --start --cb "model:<FREE_MODEL>" --cb "prompt:default"` produces equivalent persisted state.
3. Re-running `--start` for an existing chat reports that the chat is already configured.
4. Custom prompt setup works once the CLI can dispatch arbitrary commands or plain setup text in the custom-prompt state.
5. The created chat appears in `--list` with the expected model and message count baseline.

Assertions should check:

- stdout summary
- chat row exists in SQLite
- selected model is `openrouter/free`
- prompt selection is persisted
- display defaults from prompt config are persisted

### 3. Core Conversation With Real LLM Calls

- [x] This scenario group is implemented

Purpose: verify the real chat loop, persistence, and response rendering.

Scenarios:

1. A simple message produces a non-empty AI response and exits successfully.
2. `--debug` produces the AI response plus the debug summary and writes JSONL logs.
3. `--stream-debug` prints intermediate edits rather than only the final edit.
4. `--history` after one exchange shows both the user message and the assistant message.
5. `--show-prompt` shows the assembled prompt, tool list, and message context for the same chat.
6. Default console mode includes test-mode context in prompt inspection, while `--real` removes that test-mode notice.

Because the provider is real, these tests should not assert exact full answers. They should assert:

- response section exists
- response is non-empty
- history contains the new exchange
- debug log entry contains the selected model and request metadata
- token and cost footer behavior is present when usage is returned

### 4. Tool Calling And Wiki Persistence

- [x] This scenario group is implemented

Purpose: verify the most important stateful feature through real provider behavior.

Use a temporary prompt that strongly encourages wiki usage in a predictable way. The prompt should instruct the model to save explicit user facts and retrieve them on follow-up.

Scenarios:

1. Send a fact such as `My favorite color is orange. Remember this.`
2. Verify that either tool-call display appears on stdout or the debug log records wiki tool usage.
3. Verify that `--wiki` shows at least one matching persisted fact.
4. Ask a follow-up question such as `What is my favorite color?` and assert that the answer contains `orange`.
5. Manually edit the wiki file on disk, run `--repair-wiki`, and assert the repair summary plus updated `--wiki` output.
6. Manually create a new valid wiki file on disk, run `--repair-wiki`, and assert that it is imported into the DB-backed view.
7. Delete a wiki file on disk, run `--repair-wiki`, and assert the orphan cleanup behavior.

This group should be treated as the main memory contract for refactoring.

### 5. Prompt Assembly And Visibility Controls

- [x] This scenario group is implemented

Purpose: freeze how the app assembles prompt context and filters tools.

Use temporary prompt files and companion TOML configs placed under `PROMPTS_DIR` to make assertions stable. Specifically, Scenarios 2-5 require per-prompt companion TOML files (e.g. `<PROMPTS_DIR>/testprompt.toml` alongside `<PROMPTS_DIR>/testprompt.txt`). These per-prompt TOMLs control tool and MCP server filtering separately from the global `[tools]` section in `models.toml`. The temporary `models.toml` used by this suite intentionally omits `[tools]` and `[mcp]` sections, so all tool/MCP filtering assertions exercise the per-prompt config path.

Scenarios:

1. `--show-prompt` includes system prompt text, message context, and tool list.
2. A per-prompt companion TOML with disabled tools hides those tools from the prompt preview.
3. A per-prompt companion TOML with disabled MCP server groups removes those tool families from the available tool list.
4. A per-prompt companion TOML with `send_datetime = false` removes timestamp decoration from newly assembled context.
5. A per-prompt companion TOML with `show_reasoning = false` and `show_tool_calls = false` is persisted at chat creation.

Once generic command dispatch exists, add command-level assertions:

6. `/reasoning` toggles reasoning visibility.
7. `/toolcalls` toggles tool-call visibility.
8. `/datetime` toggles the persisted `send_datetime` flag on the Chat DB record. Note that Scenario 5.4 tests the initial value set by the prompt config at creation time; this scenario verifies that the runtime `/datetime` command flips the flag and that the change is reflected in subsequent `--show-prompt` output.
9. `/timezone Europe/Moscow` updates timezone and affects future prompt assembly.
10. `/help` prints the expected command list (all registered commands should appear: `/start`, `/import`, `/reset`, `/model`, `/timezone`, `/datetime`, `/reasoning`, `/toolcalls`, `/resend_last`, `/help`).
11. `/model` on an existing chat prints the current model name and suggests `/reset` + `/start` to change it.
12. `/model` on a non-existent chat prints the "No chat exists yet" guidance.

### 6. Import And Replay

- [x] This scenario group is implemented

Purpose: verify the black-box import flow that already exists in the CLI.

Scenarios:

1. `--import-json` with a valid OpenAI-style message array imports messages and prints the imported count.
2. `--import-json` with AI Proxy v2 JSON is accepted and imported.
3. `--history` after import shows the replayed content.
4. Imported chats can continue with a real follow-up message through the LLM.
5. Invalid JSON input fails with the documented import error.
6. Empty import input fails with the documented no-messages behavior.

Because `--import-json` is the actual CLI contract today, it should be considered the canonical black-box import path.

### 7. Reset, Backup, Regenerate, And Cut

- [x] This scenario group is implemented

Purpose: freeze the highest-risk duplicated and stateful paths.

This area depends on the CLI command-dispatch gap being fixed.

Scenarios:

1. `/reset` shows a confirmation dialog with message count and model info.
2. `confirm_reset:<chat_id>` creates a backup zip and deletes the chat history. **Note:** the `<chat_id>` in the callback payload is the internal composite ID produced by `_chat_id_for()` (format `user_id@bot_id`). In console mode `bot_id` is always empty, so the composite ID equals the raw chat ID passed via `-c`. Tests that assert callback payloads should be aware of this mapping.
3. After reset, `--history` shows no messages and the chat must be re-created.
4. After a successful answer, the response output exposes `regen` and `cut:<id>` callbacks.
5. `regen` followed by `confirm_regen` produces a new answer and preserves the normal response contract.
6. `cut` followed by confirmation hides earlier history from prompt assembly while keeping the old messages searchable/persisted.
7. `/resend_last` re-sends the last assistant message when available.

This group is the most important place to apply the duplication rule:

- the ordinary conversation path is the reference behavior
- regenerate must be tested against that reference
- if the two diverge, the regenerate test should fail and remain failing until fixed

### 8. Access Control And Multi-User Edges

- [x] This scenario group is implemented

Purpose: verify that console mode still respects user scoping and access rules.

Important implementation note: the production code has a per-bot `allowed_users` whitelist (loaded from `bots.toml`) that takes precedence over the global `ALLOWED_USERS` env var. In console mode, `bot_id` is always empty because `ConsoleMessenger` doesn't set it — so the per-bot lookup should not fire, and the global `ALLOWED_USERS` check applies. The test setup must ensure `BOTS_CONFIG_PATH` points to a nonexistent or empty file so that a leftover `bots.toml` in the project root does not silently override global access control in test runs.

Scenarios:

1. With `ALLOWED_USERS` configured, an unapproved `--user-id` gets the access-denied message.
2. An approved `--user-id` can create and use a chat.
3. Different chat IDs keep separate history and wiki state.
4. Remembered last-chat state does not cross isolated test environments.

### 9. Error Mapping And Resilience

- [ ] This scenario group is implemented

Purpose: verify user-visible error behavior for the real provider path where practical.

Not every provider error is easy to trigger against the real service, so this group should be split into required and optional cases.

Required:

1. Missing API key.
2. Invalid chat state such as `--show-prompt` or `--history` on a missing chat.
3. Invalid callback without an active session.

Optional or nightly-only:

4. Model unavailable.
5. Rate limit.
6. provider 5xx retry surface.
7. context-too-long guidance.

These optional scenarios may need a dedicated provider configuration, a forced proxy, or controlled fault injection later.

## Stabilizing Real-Provider Tests

To keep the suite useful rather than flaky:

1. Use very small prompts and short user messages.
2. Prefer assertions on artifacts and keywords over exact prose.
3. Use temporary prompts crafted for deterministic behavior where needed.
4. Implement a 429-specific backoff-and-retry strategy in the test helper to gracefully wait (e.g., up to 60s) for rate limits on the free tier. Other clearly transient provider errors should be retried at most once.
5. Keep one chat per test unless a scenario explicitly verifies cross-invocation persistence.
6. Separate the suite into `smoke` and `full` subsets if runtime or cost becomes noticeable.

## Rollout Order

Implement the suite in this order:

### Phase 0: Reachability

- [x] Add generic command dispatch support to `mai-chat`.
- [x] Add subprocess helpers and temp-environment fixtures.
- [x] Add a temp `models.toml` using `openrouter/free`.

### Phase 1: Minimal Protection

- [x] CLI smoke tests.
- [x] onboarding tests.
- [x] one real conversation test.
- [x] debug-log creation test.
- [x] history and prompt-preview tests.

### Phase 2: Stateful Behavior

- [x] wiki creation and recall.
- [x] repair-wiki.
- [x] import and replay.
- [x] access-control checks.

### Phase 3: Divergence Guards

- [x] regenerate parity with normal conversation flow.
- [x] reset and backup.
- [x] cut-above behavior.
- [x] visibility toggles and timezone commands.

## Definition Of Done

This plan is complete when the future implementation provides:

- [x] a subprocess-driven pytest suite using the real `mai-chat` binary
- [x] real LLM calls through `openrouter/free`
- [x] isolated temp DB and temp memory directories per test
- [x] explicit coverage of onboarding, conversation, wiki/tool usage, prompt assembly, import, reset, and regenerate
- [x] explicit failing tests for rarer duplicated paths that do not match the canonical, more frequent behavior

That will give refactoring protection at the actual product boundary, not just at the unit boundary.