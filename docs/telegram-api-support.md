# Telegram Bot API support audit (TG-1)

Audited on 2026-11-22. The latest published `python-telegram-bot` (PTB) was **22.8**, which reports Bot API **10.0**. Telegram added rich messages and rich drafts in Bot API 10.1, the draft stop controls and update in 10.3, and selected quotes from rich messages in 10.4. [PyPI release](https://pypi.org/project/python-telegram-bot/22.8/) · [PTB releases](https://github.com/python-telegram-bot/python-telegram-bot/releases/tag/v22.8) · [Telegram Bot API changelog](https://core.telegram.org/bots/api-changelog)

Keep `python-telegram-bot>=21.0`. The existing delivery path uses ordinary sends and edits. PTB 21.0 passed the focused Telegram adapter and startup tests, and raising the floor to 22.8 would still leave the rich and stop features without native SDK support. The audit uses raw requests only in the offline probe. Keep each new capability disabled until its checks below pass.

## Tested environments

| Environment | Python | PTB | Result |
| --- | --- | --- | --- |
| Isolated development virtualenv | 3.12.10 | 22.8 | Offline audit passed. The combined TG-1/TG-2 worktree passed 1,292 local tests and 32 serial integration tests with `make test-local`. |
| Repository Dockerfile on Linux amd64 | 3.12.16 | 22.8 | `pip install .` resolved 22.8. The same audit passed with `--network none`, and `pip check` passed. |
| Separate virtualenv at the dependency floor | 3.12.10 | 21.0 | 37 tests passed with `pytest -n0 tests/test_messenger/test_telegram.py tests/test_main.py`. `TextQuote` is available and `send_message_draft` is absent. |

## Capability results

| Capability | Released SDK and offline result | Blocker to enabling it |
| --- | --- | --- |
| **TG-3 · Rich inbound message** | PTB 22.8 has no `RichMessage` or typed `Message.rich_message`. A private-chat table with no text or caption was preserved as a nested dictionary under `Message.api_kwargs["rich_message"]`. The project's handler registration produced zero callbacks for that message. | Rich dispatch and conversion into a model-readable turn need verification through prompt construction, including unsupported blocks and table row boundaries. [RichMessage](https://core.telegram.org/bots/api#richmessage) |
| **TG-3 · Selected quotes** | PTB 21.0 exposes `TextQuote`. In 22.8 a plain selected quote decoded to `message.quote.text`. The rich table-cell selection decoded with empty `quote.text` and the cell available under `quote.api_kwargs["rich_texts"]`. Both questions reached the adapter once with the question alone in `IncomingMessage.text`. The command, plain text and captioned photo each dispatched once. | Carry the plain or rich selection separately from the question through prompt construction, then verify the selection from a Telegram client. [TextQuote](https://core.telegram.org/bots/api#textquote) |
| **TG-4 · Plain draft baseline** | PTB added `Bot.send_message_draft` in 22.6. In 22.8 the typed method serialized a private-chat preview with a nonzero draft ID and decoded `true` from the mock. The method also passed raw stop options via `api_kwargs`. | Production uses the existing send/edit path. Plain draft support by itself leaves the rich-draft and stop checks open. [PTB draft method](https://docs.python-telegram-bot.org/en/stable/telegram.bot.html#telegram.Bot.send_message_draft) · [Telegram draft method](https://core.telegram.org/bots/api#sendmessagedraft) |
| **TG-4 · Rich draft** | PTB 22.8 lacks `Bot.send_rich_message_draft`. Its public `Bot.do_api_request` serialized two `sendRichMessageDraft` requests with the same nonzero draft ID, a thread ID and a Markdown table inside `rich_message`. One request included the stop options. Both mock responses decoded as `true`. | Confirm acceptance and rendering in a private chat, preview visibility during incremental generation, and fallback behavior when a draft is rejected. [sendRichMessageDraft](https://core.telegram.org/bots/api#sendrichmessagedraft) |
| **TG-4 · Publish and edit rich replies** | PTB 22.8 lacks `Bot.send_rich_message` and a typed `rich_message` parameter on `Bot.edit_message_text`. Raw `sendRichMessage` and `editMessageText` requests serialized Markdown and an inline keyboard with the expected IDs. Their mock responses decoded to `Message` objects with the returned ID and raw rich content. PTB emitted a warning recommending its typed edit method, whose signature requires `text`. The rich-only callback reached the app with its keyboard, but the existing restoration returned an empty body. Ordinary send and edit succeeded through the adapter. | Verify that cut, cancel and regen preserve content and controls on a published rich message. Verify one permanent reply and its recorded ID, including after ambiguous final-send timeouts. [sendRichMessage](https://core.telegram.org/bots/api#sendrichmessage) · [editMessageText](https://core.telegram.org/bots/api#editmessagetext) |
| **TG-5 · Stop generation** | PTB 22.8 has no `MessageGenerationStopped`, typed `Update.stopped_message_generation`, or matching entry in `Update.ALL_TYPES`. Explicit `allowed_updates` serialized successfully. PTB stored the mock stop update under `Update.api_kwargs` with its chat, optional thread and draft ID. Both `effective_chat` and `effective_message` were empty. A diagnostic `TypeHandler(Update, ...)` received the raw stop via `Application.process_update`. Production polling omits an explicit allowed_updates list; the probe supplied it separately. The adapter has no stop handler, and its builder uses the default single-update processing slot. | Verify delivery from a real client and route the stop while generation is active. Enable `can_stop` only after cancellation reaches the provider and the stop/publication race preserves the correct history. [MessageGenerationStopped](https://core.telegram.org/bots/api#messagegenerationstopped) · [Update](https://core.telegram.org/bots/api#update) |

## Documented chat scope and limits

- Rich drafts, plain drafts and their stop buttons target private chats. Both draft methods accept an optional thread ID. A draft is temporary, roughly 30 seconds. `keep_on_stop` extends the preview temporarily. Persistence requires publishing a regular message. [Rich drafts](https://core.telegram.org/bots/api#sendrichmessagedraft) · [Plain drafts](https://core.telegram.org/bots/api#sendmessagedraft)
- Ordinary rich publication accepts target chats, including private chats, groups, supergroups and channels, subject to the usual permissions. Forum thread IDs apply to forum supergroups and private chats with bot topic mode enabled. Channel direct messages require a direct-message topic ID. The offline fixtures cover the private-chat path. [Rich publication](https://core.telegram.org/bots/api#sendrichmessage)
- Telegram documents up to **32,768 UTF-8 characters** of rich text, including emoji alternative text and formula source, **500 blocks** including nested blocks, list items and table rows, **16 levels** of nesting, **50 media attachments** and **20 table columns**. Rich drafts disallow direct uploads and explicit URL uploads. These are documented limits. Server-side boundary behavior remains unverified. [Rich message limits](https://core.telegram.org/bots/api#rich-message-limits) · [Rich draft parameters](https://core.telegram.org/bots/api#sendrichmessagedraft)

## Private-chat smoke test

**Offline smoke passed** in both environments on synthetic private-chat updates and mocked HTTP responses. The probe exercised PTB deserialization, the app's registered handlers, and its HTTP request encoding. It injected a test-only handler to confirm that raw stop data can reach a handler. The transport has a reserved `.invalid` hostname and a fake token.

**Live private-chat delivery passed.** After verifying the test bot with getMe and the recipient with getChat, Telegram accepted sendRichMessageDraft and returned message 811 with two rich blocks from sendRichMessage. The recipient confirmed that the final message arrived. Draft animation, formatting fidelity, size boundaries and interactive stop delivery remain checks for TG-3 through TG-5.

## Reproduce the offline audit

Run the probe at the audited PTB version. Its assertions describe the 22.8 baseline and should be revisited when auditing a later release.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . python-telegram-bot==22.8
.venv/bin/python scripts/audit_telegram_sdk.py
```

The audited local image ID was `sha256:c5857a807a755fca154e8ae04a7223424cfa95b3f3ebab80d879b39d00b30c41`. A fresh build resolves the currently available dependencies. From a checkout containing the committed changes, build with the repository Dockerfile using tracked build inputs to exclude local configuration and credentials. The run passes the probe through standard input and uses the SDK installed by the Dockerfile.

```sh
git ls-files -z -- Dockerfile pyproject.toml README.md src config prompts |
  tar --null -T - -cf - | docker build --tag mai-telegram-tg1-audit -
docker run --rm -i --network none mai-telegram-tg1-audit python - < scripts/audit_telegram_sdk.py
docker run --rm --network none mai-telegram-tg1-audit python -m pip check
```
