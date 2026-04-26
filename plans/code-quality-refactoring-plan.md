# Code Quality Refactoring Plan

## Goal

Refactor `mai-gram` to reduce architectural risk, tighten typing and error boundaries, and make the highest-risk workflows easier to change without behavioral regressions. The target quality bar should be the existing `mai-gram` contract plus the stronger Python standards already enforced in `ai-proxy2`.

This plan is intentionally implementation-oriented. Every issue below has a checkbox, a concrete solution, the `ai-proxy2` standard it borrows, and explicit validation work.

## Baseline Findings

The current codebase already has useful guardrails: Ruff, strict mypy, a pre-commit target, and a functional CLI test path. The main problems are structural rather than cosmetic.

- `src/mai_gram/bot/handler.py` is 2653 lines and contains multiple workflows that should be separate services.
- `src/mai_gram/console_runner.py` is 656 lines and duplicates application behavior that also exists in the Telegram flow.
- `src/mai_gram/llm/openrouter.py` is 621 lines and mixes payload shaping, retry policy, SSE parsing, and error translation.
- `src/mai_gram/memory/summarizer.py` is 857 lines and is large despite being mostly outside the current coverage focus.
- Oversized functions already exceed the `ai-proxy2` size policy by a wide margin: `_handle_regenerate` is 409 lines, `_handle_conversation` is 365, `_handle_document` is 198, `_run` is 153, `_print_prompt` is 90, and `_stream_sse` is 135.
- The coverage configuration omits several orchestrator modules from the coverage threshold, including `bot/handler.py`, `console_runner.py`, `llm/openrouter.py`, `core/prompt_builder.py`, and multiple memory modules.
- `mai-gram` uses a lighter Ruff ruleset than `ai-proxy2`; it currently lacks `C4`, `DTZ`, `T20`, `RUF`, `S`, and `PTH`.
- The project relies heavily on `dict[str, Any]`, `getattr`, `setattr`, ad hoc JSON parsing, and broad `except Exception` blocks around important runtime paths.

## AI Proxy2 Standards To Import

- [x] Import `ai-proxy2` code-size limits into local quality gates.
  Apply a `500` line file limit and `60` line Python function limit, with explicit exclusions only for migrations or similarly generated code. Start in report-only mode if necessary, then make the check blocking once the first decomposition pass lands.

- [ ] Expand Ruff to match the `ai-proxy2` backend rule categories.
  Add `C4`, `DTZ`, `T20`, `RUF`, `S`, and `PTH` to the existing selection set. Fix violations in production code first, then add narrow per-file ignores only when a rule is genuinely incompatible with the project.

- [x] Make mypy strictness explicit instead of relying only on `strict = true`.
  Add `warn_return_any`, `warn_unused_configs`, `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`, and `no_implicit_optional` as explicit policy, mirroring `ai-proxy2` so the intended bar remains visible even if mypy changes the meaning of `strict` later.

- [ ] Move toward the `ai-proxy2` coverage posture.
  `ai-proxy2` enforces `95%` coverage with minimal omissions. `mai-gram` should not jump straight to `95%` while major modules are still monoliths, but it should first shrink the omit list, then raise `fail_under` in steps until it reaches `95%`.

- [ ] Add a repository-owned pre-commit quality gate for maintainability, not only correctness.
  Mirror the `ai-proxy2` pattern of using tracked hooks to enforce both static analysis and structural limits, so code size and coverage regressions are blocked before they accumulate.

## Refactoring Checklist

### 1. Foundation: import the missing quality guardrails

- [ ] Adopt the missing `ai-proxy2` static-analysis and maintainability gates.

Issue:

`mai-gram` already runs Ruff, mypy, tests, and live functional checks, but the static ruleset is materially lighter than `ai-proxy2`, there is no code-size gate, and the coverage threshold is weakened by large omissions exactly where refactoring risk is highest.

Proposed solution:

Bring the repository configuration closer to `ai-proxy2` before large refactors begin, so the refactor is guided by stronger automated feedback instead of depending only on manual discipline.

How to perform:

1. Update `pyproject.toml` Ruff settings to include `C4`, `DTZ`, `T20`, `RUF`, `S`, and `PTH`.
2. Add the explicit mypy flags used in `ai-proxy2`, plus a narrow test override if needed.
3. Introduce a local code-size checker modeled on `ai-proxy2/scripts/check_code_limits.py`.
4. Wire the new size check into `make check` or `make precommit` so it is enforced the same way as the rest of the quality gate.
5. Change coverage policy in stages: first reduce omissions, then raise the threshold.

How the `ai-proxy2` standard applies:

The borrowed standard is not just “use Ruff and mypy”; it is “enforce maintainability rules in the same place as correctness rules.” That means code size, explicit type policy, and stronger lint categories must become part of the normal developer workflow.

Validation:

Run `make check`, `make precommit`, and verify the new size checker reports current hotspots clearly enough to drive the follow-on tasks.

Status:

The repository now has an explicit mypy policy and a tracked code-size audit wired into `make check` and the local pre-commit hook in report-only mode. Ruff rule expansion and coverage tightening remain follow-on work because they still require violation cleanup rather than a safe configuration-only change.

### 2. Decompose `BotHandler` into transport-facing dispatch plus application services

- [ ] Split `BotHandler` into focused services and reduce the class below the size-limit threshold.

Issue:

`BotHandler` owns setup, import, reset, prompt assembly, access control, message persistence, tool execution, streaming UI behavior, regeneration, and backup logic. The class is both the adapter entrypoint and the business workflow container.

Proposed solution:

Keep `BotHandler` as a thin transport coordinator and move real work into application services with clear dependencies and return types.

How to perform:

1. Extract command-level services such as `SetupService`, `ConversationService`, `ResetService`, `ImportService`, and `ResendService`.
2. Move database mutations and cross-module orchestration out of handler methods and into service methods that accept typed input objects.
3. Keep messenger-specific concerns in the handler: command registration, basic message decoding, and service invocation.
4. Introduce typed result objects for service responses so handlers no longer infer behavior from raw strings, JSON fragments, or side effects.
5. Refactor incrementally by moving one workflow at a time, starting with the most duplicated path: conversation and regenerate.

How the `ai-proxy2` standard applies:

The direct standard is the code-size rule, but the deeper import is that orchestration should be small enough for lint, typing, and coverage to protect it meaningfully. A 2653-line class cannot meet that bar in practice.

Validation:

Add service-level unit tests around the extracted workflows and keep `mai-chat` functional scenarios passing for `/start`, ordinary conversation, `/reset`, import, and regenerate.

Status:

The shared conversation/regenerate pipeline now lives in `src/mai_gram/bot/conversation_executor.py` with direct unit coverage, tool-call persistence and user-facing tool activity display now live in `src/mai_gram/bot/tool_activity_notifier.py`, the import/document state machine now lives in `src/mai_gram/bot/import_workflow.py` with focused workflow tests, resend-last now lives in `src/mai_gram/bot/resend_service.py` with direct service coverage, setup/onboarding now lives in `src/mai_gram/bot/setup_workflow.py` with direct workflow tests plus CLI onboarding coverage, reset/backup execution now lives in `src/mai_gram/bot/reset_workflow.py` with direct workflow tests plus reset functional coverage, the remaining history/regenerate actions now live in `src/mai_gram/bot/history_actions.py` and `src/mai_gram/bot/regenerate_service.py` with direct service coverage plus the existing reset/regenerate functional path, callback dispatch now lives in `src/mai_gram/bot/callback_router.py` with direct router tests plus the existing callback-driven functional flows, response rendering now lives in `src/mai_gram/bot/response_renderer.py` with direct rendering tests, ordinary conversation setup now lives in `src/mai_gram/bot/conversation_service.py` plus `src/mai_gram/bot/assistant_turn_builder.py` so regenerate and the ordinary message path share the same assistant-turn request assembly, MCP/tool composition now lives in `src/mai_gram/bot/mcp_manager_factory.py` with direct filter coverage, transport-facing access control now lives in `src/mai_gram/bot/access_control.py` with direct unit coverage, and service assembly now lives in `src/mai_gram/bot/handler_services.py` so `src/mai_gram/bot/handler.py` is no longer a file-size violation. The conversation-path refactor has now also removed `conversation_executor.py` and the adjacent resend/regenerate methods from the current size-audit report. `BotHandler` still owns transport registration and a few Telegram-facing helpers, so the broader decomposition remains incomplete.

### 3. Unify ordinary conversation and regenerate into one canonical generation pipeline

- [x] Replace duplicated generation logic with a single conversation execution pipeline.

Issue:

The ordinary conversation flow and regenerate flow duplicate prompt assembly, MCP setup, streaming, tool-call persistence, error handling, and response finalization. This contradicts the existing project rule that ordinary conversation is the canonical path when duplicated behaviors diverge.

Proposed solution:

Build one reusable execution pipeline with small strategy hooks for the few cases that are legitimately different, such as deleting the last assistant turn before regeneration.

How to perform:

1. Identify the shared sequence of steps in `_handle_conversation` and `_handle_regenerate`.
2. Extract those steps into a `ConversationExecutor` or similarly named service that accepts a typed execution request.
3. Model regeneration-specific differences as flags or dedicated pre-processing steps instead of copy-pasted branches.
4. Centralize streaming-update assembly, tool iteration, usage accounting, and message finalization so fixes happen once.
5. Make the ordinary path the reference implementation and route regeneration through the same core execution method.

How the `ai-proxy2` standard applies:

This task applies the same maintainability principle as the `ai-proxy2` file/function limits: one behavior should have one implementation unless duplication is deliberate and defended.

Validation:

Extend functional tests so a normal message and a regenerate action produce the same persistence and tool-call behavior for the same chat state, except for the intentional “replace last answer” semantics.

Status:

Ordinary conversation and regenerate now share a dedicated `ConversationExecutor` service in `src/mai_gram/bot/conversation_executor.py`, a common assistant-turn request builder in `src/mai_gram/bot/assistant_turn_builder.py`, and the ordinary message adapter path in `src/mai_gram/bot/conversation_service.py`; the regenerate tool-chain preservation path plus direct executor and builder/service branch coverage are both in place. Further decomposition is still needed around the remaining `BotHandler` workflows, but the canonical generation pipeline and its setup logic are no longer embedded directly inside the handler class.

### 4. Create shared application services for CLI and Telegram adapters

- [ ] Remove duplicated business logic between `console_runner.py` and the Telegram runtime.

Issue:

The console CLI and Telegram flow both exercise setup, prompt preview, import, history inspection, wiki sync, and conversation execution, but too much of that logic is duplicated or adapter-owned.

Proposed solution:

Promote the shared behaviors into application services that both adapters call, leaving each adapter responsible only for transport-specific input and output handling.

How to perform:

1. Define adapter-neutral use cases for setup, conversation submission, prompt preview, history listing, wiki inspection, and repair.
2. Move shared workflow steps out of `console_runner.py` and out of handler methods into these use cases.
3. Keep CLI argument parsing and Telegram event parsing thin; they should translate external input into the same internal request models.
4. Reuse the same service objects in `mai-chat` and Telegram startup so parity comes from shared execution, not from duplicated code.
5. Treat the documented CLI path as a first-class adapter, not a debugging sidecar.

How the `ai-proxy2` standard applies:

This uses the same standards philosophy seen in `ai-proxy2`: strong quality checks are most effective when the code under them has narrow, explicit boundaries instead of multi-entrypoint duplication.

Validation:

Run the CLI functional suite and targeted Telegram-facing tests after each extraction step. New shared-service tests should replace duplicated adapter-specific tests where possible.

### 5. Tighten the messenger boundary and stop leaking Telegram details into core logic

- [ ] Make transport abstractions real by moving Telegram-specific behavior behind the messenger adapter.

Issue:

Core workflows still inspect raw Telegram message objects, keyboard specifics, and transport-specific state. That makes the abstraction in `messenger/base.py` incomplete and keeps non-Telegram execution paths coupled to Telegram assumptions.

Proposed solution:

Push keyboard creation, callback acknowledgement, and raw update handling into the Telegram adapter, and expose transport-neutral operations from the messenger interface.

How to perform:

1. Inventory every place where `BotHandler` or shared services touch Telegram-specific objects or fields.
2. Replace those usages with adapter methods such as `ack_callback`, `build_confirmation_ui`, `edit_or_send_followup`, or other transport-neutral operations.
3. Extend `OutgoingMessage` and related abstractions only when the transport-neutral model genuinely needs a new concept.
4. Keep raw `Any` payloads at the edge and convert them into typed messenger-layer objects before they reach the business layer.

How the `ai-proxy2` standard applies:

This task maps to the `ai-proxy2` preference for typed boundaries and low `Any` leakage. The point is not abstraction for its own sake; it is making the domain layer independent of transport implementation details.

Validation:

Verify that the console adapter can exercise the same core services without Telegram-specific conditionals and that Telegram tests still cover callback and edit behavior.

### 6. Split configuration loading into typed loaders and remove global singleton behavior from runtime code

- [ ] Replace the overloaded `Settings` singleton with explicit configuration and runtime composition.

Issue:

`Settings` currently mixes environment settings, model config, prompt config, bot config, external MCP config, file caching, and reload behavior. Runtime code also reaches into private config internals and depends on a process-global singleton.

Proposed solution:

Separate static environment settings from reloadable file-backed configuration and compose them through an explicit runtime container instead of hidden globals.

How to perform:

1. Keep `BaseSettings` for environment variables only.
2. Extract typed loaders for models, prompts, bots, and external MCP config files.
3. Replace `dict[str, Any]` config accessors with typed dataclasses or Pydantic models.
4. Introduce an app context or dependency container that owns settings, DB lifecycle, provider instances, and config loader instances.
5. Remove direct calls into private methods such as `_load_toml` from unrelated runtime code.
6. Replace broad parse fallbacks with precise error reporting so invalid config does not silently degrade into defaults.

How the `ai-proxy2` standard applies:

This borrows two `ai-proxy2` habits: typed configuration boundaries and explicit validation before runtime behavior depends on config. It also reduces the need for `Any`, `getattr`, and hidden cache state.

Validation:

Add focused tests for config reload, invalid prompt companion TOML, external MCP config parsing, and multi-bot restrictions. Manual verification through `mai-chat` should still show the same prompt and model choices.

### 7. Replace weakly typed MCP and LLM boundaries with validated domain models

- [ ] Remove `dict[str, Any]` and ad hoc JSON handling from the core tool-calling path.

Issue:

MCP tool metadata, tool arguments, tool results, provider payloads, stream deltas, and persisted tool-call payloads are handled mostly as plain dictionaries and unvalidated JSON strings. The boundary is flexible but brittle.

Proposed solution:

Define typed domain models for tool definitions, tool invocations, tool results, streaming deltas, and provider payload conversion, with JSON serialization confined to the outermost provider and transport edges.

How to perform:

1. Introduce strongly typed models for registered tools, tool calls, tool results, and stream chunks where `Any` is currently prevalent.
2. Validate incoming JSON from external MCP servers and OpenRouter before it reaches business logic.
3. Replace repeated `json.loads` and unchecked dictionary indexing with parsing helpers that return typed objects or explicit parse errors.
4. Normalize provider payload shaping in one place so both streaming and non-streaming requests use the same conversion code.
5. Remove broad `Any` from method signatures where the valid shapes are known.

How the `ai-proxy2` standard applies:

This is the closest analogue to the typed backend models in `ai-proxy2`: flexible external APIs are still wrapped in explicit internal contracts so refactors and failures are easier to reason about.

Validation:

Expand unit tests for MCP bridging, streaming delta parsing, malformed tool-call JSON, and external server error handling. Keep the live CLI path passing for tool-enabled prompts.

### 8. Normalize persistence semantics and make wiki synchronization explicit

- [ ] Align database types and workflow semantics with the actual domain model.

Issue:

Structured data is stored in stringly typed fields, wiki importance is modeled as a float even though file naming uses integer importance, and several read-oriented flows hide write behavior because they call `sync_from_disk` as a side effect.

Proposed solution:

Use explicit domain types for persisted data, make synchronization an intentional workflow step, and keep read operations read-only wherever possible.

How to perform:

1. Replace raw string role values with an enum-backed domain type at the application boundary.
2. Define a structured representation for persisted tool calls and normalize serialization in one repository layer.
3. Align wiki importance with the actual disk contract, using integer importance consistently from file name to DB model to UI.
4. Move wiki reconciliation out of generic read paths such as prompt building and ad hoc inspection commands; trigger it explicitly at controlled workflow points.
5. Document transaction boundaries around sync so the team knows when read operations are allowed to mutate disk or DB state.

How the `ai-proxy2` standard applies:

This imports the same “explicit contracts over permissive data blobs” standard and applies it to persistence. The benefit is fewer parse-time surprises and cleaner repository-layer tests.

Validation:

Add migration tests if schema changes are needed, plus black-box tests for wiki repair, wiki listing, and context-building behavior before and after explicit sync.

### 9. Reduce broad exception handling and centralize retry and error translation policy

- [ ] Replace catch-all exception blocks with typed failure handling and shared retry policy.

Issue:

Critical flows use broad `except Exception` blocks in configuration parsing, MCP shutdown, streaming/tool execution, database helpers, and runtime startup. That makes genuine defects easy to hide and forces recovery logic to guess which failures are safe.

Proposed solution:

Catch only expected failures, centralize retry classification, and convert external errors into a small, typed internal error hierarchy.

How to perform:

1. Inventory every `except Exception` in production code and classify it as expected external failure, cleanup-only failure, or bug-masking fallback.
2. Replace bug-masking fallbacks with narrower exception types and richer logs.
3. Reuse one retry classification policy across live response delivery, import replay, and external tool interactions.
4. Ensure that cleanup paths still suppress only the narrow errors that are safe to ignore.
5. Where a catch-all remains necessary at a process boundary, log full context and convert to an explicit user-facing failure state.

How the `ai-proxy2` standard applies:

This maps directly to the stronger lint categories from `ai-proxy2`, especially the push toward explicit, auditable runtime behavior rather than blanket exception suppression.

Validation:

Add tests for timeout paths, malformed upstream payloads, rate-limit retries, MCP subprocess termination failures, and invalid config files.

### 10. Decide the fate of the summarization subsystem and stop carrying unowned complexity

- [ ] Either fully integrate or explicitly quarantine the large summarization codepath.

Issue:

`memory/summarizer.py`, `memory/summaries.py`, and related modules are large and contribute cognitive load, but they are not central to the currently documented behavior and remain outside much of the coverage contract.

Proposed solution:

Make an explicit product and architecture decision: either the summarization pipeline is an actively supported subsystem with tests and quality gates, or it is isolated behind a clear feature boundary until it is ready.

How to perform:

1. Audit which summarization features are actually invoked in production paths today.
2. If the feature is active, split the module into focused components and add service-level tests until it can live under the normal coverage threshold.
3. If the feature is inactive or experimental, isolate it behind a feature flag or move it into a clearly marked experimental package.
4. Remove dead wrappers and ambiguous partial abstractions so contributors understand whether the subsystem is part of the supported core.

How the `ai-proxy2` standard applies:

This follows the same maintainability discipline as the size gate: large subsystems should either be owned and tested or clearly separated from the supported core.

Validation:

The module should either pass the same quality gates as the rest of the project or be excluded by an explicit, documented architectural decision rather than by drift.

### 11. Rebuild the coverage strategy around services and shrink the omit list

- [ ] Move test coverage onto the refactoring seams so the omit list can be reduced safely.

Issue:

The current coverage policy excludes several of the exact files that need the most aggressive refactoring. That weakens the safety net and encourages continued avoidance of the hard-to-test orchestration code.

Proposed solution:

As services are extracted, add direct tests around those seams and remove broad file-level coverage omissions, then raise `fail_under` toward the `ai-proxy2` target.

How to perform:

1. For each extraction above, add focused tests at the new service layer before deleting old adapter-heavy tests.
2. Remove omissions for refactored files one by one instead of waiting for a single large change.
3. Increase the coverage threshold in stages after each omit reduction.
4. Keep the live functional suite as the end-to-end backstop, but stop using it as the only protection for orchestration logic.

How the `ai-proxy2` standard applies:

This is the direct coverage alignment task. The important principle is that the highest-risk modules must become measurable, not permanently exempt.

Validation:

Track omit-list reduction and coverage-threshold increases in the same change sets as the service extractions that make them feasible.

## Recommended Order

1. Strengthen guardrails first: lint, typing, code-size checks, and coverage policy.
2. Extract the shared conversation pipeline, because it removes the largest duplication and unlocks safer follow-on refactors.
3. Split `BotHandler` and adapter logic around the new services.
4. Replace weak typing and implicit config/runtime state once the service seams exist.
5. Normalize persistence and wiki-sync semantics.
6. Finish by tightening coverage and resolving the summarization subsystem boundary.

## Definition Of Done

- [ ] No production Python file exceeds the adopted line-limit policy unless there is a documented exception.
- [ ] No production Python function exceeds the adopted function-length limit unless there is a documented exception.
- [ ] Ruff and mypy settings are at least as explicit as the `ai-proxy2` backend baseline.
- [ ] Coverage omissions no longer include the main orchestration paths without a documented architectural reason.
- [x] Ordinary conversation and regenerate share one canonical execution pipeline.
- [ ] CLI and Telegram adapters call the same core services for shared behavior.
- [ ] Core workflows no longer depend on raw Telegram objects or `dict[str, Any]` payloads when a typed domain model is available.
- [ ] Wiki synchronization and summarization behavior are explicit, owned, and testable.