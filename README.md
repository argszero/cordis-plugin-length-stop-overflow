# @argszero/cordis-plugin-length-stop-overflow

**Two ways a provider refuses an over-long request without naming the reason —
both read as something the harness does not recover from, both restored to the
overflow recovery that already exists.**

A session can end turn after turn with **one** output token and never recover.
The one token is not the provider's choice — dsh asks for it.

## The first defect (discussion #7214)

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

## The second defect (discussion #7626)

A session can also grow past the provider's **transport** limit, in bytes, while
every budget dsh computes still reports headroom — because dsh prices a request
in *tokens* and never in *bytes*. The only byte budget in the whole LLM layer
prices images (`LlmImageRequestBudget.maxBytes`); a text request has no byte
bound anywhere.

The provider then refuses the request outright, and the classification order
turns that into a code nothing recovers from
(`llm-deepseek/src/transport.ts:31-33`):

```ts
else if (isContextWindowExceededError(detail)) code = 'CONTEXT_WINDOW_EXCEEDED'
else if (status === 400 || status === 413 || type === 'invalid_request_error') code = 'INVALID_REQUEST'
```

A 413 with no JSON body carries no wording for `isContextWindowExceededError` to
read (`llm/src/error.ts:54-85`), so the adapter's fallback string —
`DeepSeek Messages request failed (413)` — reaches the `INVALID_REQUEST` branch.
Then:

- the recovery gate (`compaction-basic/src/index.ts:180`) reads
  `CONTEXT_WINDOW_EXCEEDED`, so the only thing that can shrink what is sent never
  runs;
- `INVALID_REQUEST` is not in `DEFAULT_RETRYABLE_CODES`
  (`llm/src/retry-policy.ts:18-24`), so the repeated 413s are re-issued turns,
  not a backoff that never gives up — hundreds of them, in the report;
- a fork is seeded with the parent's corpus, so it fails from its first turn.

The second trigger reads **size evidence** instead of the code, and rewrites the
same way:

```
error finish (INVALID_REQUEST + 413 / size wording)  ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
```

`INVALID_REQUEST` is a bag that holds malformed, forbidden and oversized
requests together, so the code alone is never sufficient — the wording has to
name size *and* name the request (not, say, an image). See
[What it will not claim](#what-it-will-not-claim).

## What the plugin does

Observes the public `llm/stream` waterfall and rewrites the terminal frame of
either defect into the `CONTEXT_WINDOW_EXCEEDED` failure the harness already
knows how to recover from:

```
max-tokens finish              ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
error finish (size refusal)    ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
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
- **One terminal frame in, one terminal frame out.** The upstream verdict is
  replaced, never followed by a second one, and the plugin never re-dispatches a
  request of its own: the retry decision stays with the loop and its
  `maxOverflowRetries` bound.

## The byte measurement

The harness has no byte accounting for a text request, so the plugin measures
what it can see where it can see it: `JSON.stringify(options.messages)` in
UTF-8, attached to the failure text and the log line, e.g. *"the request carried
59 MB (61,800,448 bytes) of message JSON (an estimate of this component — the
wire form belongs to the adapter)"*.

Two honesty notes:

- it is **one component of the request**, measured at the `llm/stream` seam.
  Adapters re-serialize into their own protocol and `dsh-attachment` may
  re-encode request images, so the wire payload is the adapter's and is not
  observable from here. It is labelled as an estimate everywhere it appears;
- it runs **only when a detection fires**, so a healthy request is never
  serialized.

## The verdict is window-independent

The plugin fires only when a `max-tokens` finish arrives with a negligible
output count. A real output cap cannot look like this: the budget it hits is the
model's own — tens of thousands of tokens in the catalog — so a `length` stop
after one or two tokens is a truncation of *our request*, whoever truncated it.
That is why the rule needs no context-window number, which is what makes it work
on routes where the window is itself wrong (discussion #7213).

A finish that carries no `usage` chunk is left alone (fail-closed): without the
provider's own count there is nothing to judge. That rule belongs to the
length-stop trigger only — a refused request may produce no chunk at all.

## What it will not claim

A guard that fires on everything is worse than one that fires on nothing, so the
size rule is narrow on purpose. It never reclassifies:

- a **caller cancellation** (`aborted`) — the caller stopped us, and that
  decision is not a provider verdict, even when the failure carries a
  request-rejection code;
- a failure whose code **another recovery already owns**: `CONTEXT_WINDOW_EXCEEDED`
  itself, and `IMAGE_OFFLOAD_REQUIRED` (the image path offloads what it named and
  retries — rewriting it would take the recovery from the thing that can fix it);
- a size complaint about **something other than the request** — `image too
  large`, `Failed to load image or audio file` — which belongs to
  `@deepseek-ai/dsh-compaction-image-offload`, not to compaction;
- a **token** bound (`input tokens exceeded max_prompt_tokens`) — a different
  budget, read by a different classifier;
- a rate limit, a transport failure, or a malformed request that merely mentions
  a size noun.

`classifyOversizeRequests: false` turns the second trigger off without touching
the first.

## What this does not fix

Where the refusal is *itself* larger than the provider will accept, no
classification can help: the summarization call replays the compacted region and
is refused too, so compaction fails (bounded by `maxOverflowRetries`, no loop).
What the plugin changes there is the **label and the message**: a turn now ends
with `CONTEXT_WINDOW_EXCEEDED` carrying the provider's own words, the status, the
request id and a measured size, instead of an opaque `DeepSeek Messages request
failed (413)` that names no cause. Where the region does fit — the session just
crossed the limit — the built-in recovery runs and the session continues.

It does **not** bound the summarization request by bytes. That is a core change
(`compaction-basic` builds one request from the replayed prefix, deliberately, so
the provider's warm prefix cache is reused; `summarizer.ts:110-162`), and no
plugin can chunk that region without losing content.

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
        mode: error                      # error | warn | off   (default: error)
        atMostOutputTokens: 2            # default: 2
        classifyOversizeRequests: true   # default: true
```

| option | meaning |
|---|---|
| `mode: 'error'` | replace the truncating finish with `CONTEXT_WINDOW_EXCEEDED` so compaction runs (default) |
| `mode: 'warn'` | log the detection and stream unchanged — how to measure your own traffic before enabling |
| `mode: 'off'` | pure pass-through |
| `atMostOutputTokens` | output count at or below which a `max-tokens` finish is a truncation (default 2) |
| `classifyOversizeRequests` | also reclassify a request refused for its size — `413`, or size wording on a request-rejection code (default `true`) |

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
| `length-stop-overflow` ≥0.2.0 (#7626) | an **error** refusing the request for its size (413, no body) | `INVALID_REQUEST` |

## Tests

```sh
npm test        # tsc, then 77 tests over a real Cordis context
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
  failure code compacts nothing, and a second overflow failure is terminal
  (hop 3: the same two arms for a size refusal, including the reported session's
  exact `INVALID_REQUEST` — which compacts nothing without the plugin).

Plus a packaging guard asserting that the shipped artifact's bare imports and
its runtime dependency declarations match in **both** directions.
