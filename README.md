# @argszero/cordis-plugin-length-stop-overflow

**A truncation the harness reads as a normal turn end, restored to the overflow
recovery that already exists for it.**

A session can end turn after turn with **one** output token and never recover.
The one token is not the provider's choice — dsh asks for it.

## The defect (discussion #7214)

`llm-pi-ai` streams through pi-ai's *simple* entry point and hands it the
resolved model (`llm-pi-ai/src/adapter.ts:380`). pi-ai clamps the output budget
against that model's context window
(`@earendil-works/pi-ai/dist/api/simple-options.js:2-9`):

```js
const CONTEXT_SAFETY_TOKENS = 4096
const MIN_MAX_TOKENS = 1
const available = model.contextWindow - estimateContextTokens(context).tokens - CONTEXT_SAFETY_TOKENS
return Math.min(maxTokens, Math.max(MIN_MAX_TOKENS, available))
```

Once the conversation outgrows the window dsh *believes* in, the request goes
out as `max_tokens: 1`. The provider answers `finish_reason: "length"` after one
token — correctly. Three things then line up against recovery:

1. **pi-ai's own overflow sniff needs exactly zero output.** `isContextOverflow`
   (`dist/utils/overflow.js:149`) special-cases a length stop with no content;
   `MIN_MAX_TOKENS = 1` is precisely what keeps the count off zero.
2. **`llm-pi-ai` reads `length` as an ordinary cap**: `case 'length': return
   { kind: 'max-tokens' }` (`llm-pi-ai/src/stream.ts:109`).
3. **`max-tokens` is a sticky terminal turn end with no recovery hook.** The only
   overflow-recovery entry point is `agent/request-error` carrying
   `CONTEXT_WINDOW_EXCEEDED` (`compaction-basic/src/index.ts:180`), and nothing on
   this path produces it.

Each degenerate answer is a valid prefix estimate for the next request (it is
neither `aborted` nor `error`), so `continue` reproduces the same negative
headroom: a full request, one token back, no way out. Repetition-based loop
detectors see a *different* one-token answer each time; the pressure threshold
that would trigger compaction is computed from a window the provider does not
share.

## What the plugin does

Observes the public `llm/stream` waterfall and rewrites a truncating
`max-tokens` finish into the `CONTEXT_WINDOW_EXCEEDED` failure the harness
already knows how to recover from:

```
max-tokens finish  ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
```

`@deepseek-ai/dsh-compaction-basic` (mounted in `bundle/base` and
`bundle/web-app`) then compacts and returns `{ kind: 'retry' }`, bounded by its
own `maxOverflowRetries` (default 1) — so there is no retry loop.

Two properties make the rewrite exact rather than approximate:

- The synthesized failure carries the harness' **own**
  `CONTEXT_WINDOW_EXCEEDED_CODE`, imported by value from `@deepseek-ai/dsh-llm`,
  so plugin and core cannot disagree about which failure the recovery path keys
  on.
- On an `error` finish the loop settles only `assistant/attempt`
  (`agent-loop/src/agent.ts:445-448`) and never appends `assistant/message`, so
  the useless one-token answer does **not** enter the derived conversation. The
  compaction summarizes a history without it.

## The verdict is window-independent

The plugin fires only when a `max-tokens` finish arrives with a negligible
output count. A real output cap cannot look like this: the budget it hits is the
model's own — tens of thousands of tokens in the catalog — so a `length` stop
after one or two tokens is a truncation of *our request*, whoever truncated it.
That is why the rule needs no context-window number, which is what makes it work
on routes where the window is itself wrong (discussion #7213).

A finish that carries no `usage` chunk is left alone (fail-closed): without the
provider's own count there is nothing to judge.

## Install

```sh
npm i @argszero/cordis-plugin-length-stop-overflow
```

Mount through a bundle patch (`cordis.patch.yml` in this package, or your own
overlay):

```yaml
- insert:
    - id: length-stop-overflow
      name: '@argszero/cordis-plugin-length-stop-overflow'
```

## Configuration

```yaml
- set:
    - id: length-stop-overflow
      config:
        mode: error             # error | warn | off   (default: error)
        atMostOutputTokens: 2   # default: 2
```

| option | meaning |
|---|---|
| `mode: 'error'` | replace the truncating finish with `CONTEXT_WINDOW_EXCEEDED` so compaction runs (default) |
| `mode: 'warn'` | log the detection and stream unchanged — how to measure your own traffic before enabling |
| `mode: 'off'` | pure pass-through |
| `atMostOutputTokens` | output count at or below which a `max-tokens` finish is a truncation (default 2) |

Raise `atMostOutputTokens` when a backend clamps the output budget to a small
*positive* remainder instead of the floor: with a believed window of 262,144 and
pi-ai's 4,096-token reserve, a prompt anchored anywhere in
`[250_000, 258_047]` yields a budget in `[8_046, 1]`, all of it context
pressure. `atMostOutputTokens: 64` covers that band.

## Relationship to `@argszero/cordis-plugin-overflow-classifier-guard`

They cover the two halves of the same recovery path, and both are safe to mount
together:

| | what the provider said | what the harness heard |
|---|---|---|
| `overflow-classifier-guard` (#6361) | an **error** naming an input-token budget, in wording the classifier misses | `INVALID_REQUEST` |
| `length-stop-overflow` (#7214) | **no error at all** — a `length` stop after one token | `max-tokens` |

## Tests

```sh
npm test        # tsc, then 48 tests over a real Cordis context
```

Three layers, each with a control arm, so a green run is evidence rather than an
accident:

- **unit** — the pure generator and the classifier.
- **behaviour** — the real `llm/stream` waterfall and the harness' own stream
  invariant, mounted and unmounted.
- **end-to-end** — the real `LlmRuntime` (hop 1) and the real
  `@deepseek-ai/dsh-compaction-basic` mounted with the services it injects
  (hop 2): the plugin's failure must pass its `failure.code` gate, reach
  `compactIfNeeded`, and come back as `{ kind: 'retry' }` — while a foreign
  failure code compacts nothing, and a second overflow failure is terminal.

Plus a packaging guard asserting that the shipped artifact's bare imports and
its runtime dependency declarations match in **both** directions.
