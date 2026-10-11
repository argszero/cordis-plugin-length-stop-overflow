# @argszero/cordis-plugin-length-stop-overflow

**Four ways a provider refuses an over-long request without naming the reason —
all read as something the harness does not recover from, all restored to the
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

## The third defect (discussion #7632)

Sometimes the provider says the right thing and the harness still cannot read
it. A `usage`-based overflow sniff exists upstream, and it is one conjunct short
of this case: pi-ai 0.85.1's `isContextOverflow` gates its usage branch on
`message.stopReason === "stop"` (`dist/utils/overflow.js:140-145`), while a
request refused for its length by a provider that does not name it answers
`stopReason: "error"`. `llm-pi-ai` then ends at `PI_AI_ERROR`
(`llm-pi-ai/src/stream.ts:42`), no wording matches the text fallback
(`llm/src/error.ts:51-85`), and the recovery gate
(`compaction-basic/src/index.ts:180`) never runs.

(The fix for exactly this was merged upstream in `4791b40b26`, "fix(llm-pi-ai):
classify usage-based context overflow" — it is an ancestor of the reporting
build `dsh-v0.1.5-rc.2`. The conjunct above is what remains.)

What is left, though, is the *provider's own arithmetic*: it counted this
request's prompt tokens and found them at (or past) the window the route
advertises. The plugin reads that count against that window:

```
error finish, nothing attributed, prompt >= saturationRatio x window
    ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
```

The failure it synthesizes keeps the provider's message verbatim and adds both
numbers, so the rewrite is checkable against the session log:

```
openai/gpt-x failed as PI_AI_ERROR after the provider counted 192749 prompt
token(s) against the 131072-token window that route advertises (1.47x): <the
provider's own words>. Reclassified as CONTEXT_WINDOW_EXCEEDED by
length-stop-overflow (discussion #7632): on a prompt this full the provider's
own numbers name the reason, and the code it arrived with is one no recovery
reads.
```

Two design notes make this rule safe to leave on by default:

- **It needs no wording.** The other two rules read the provider's *shape*
  (a `length` stop, a size noun); this one reads its *numbers*. That is why it
  works on the case the upstream conjunct misses — a refusal whose text names
  nothing at all.
- **The window is resolved lazily.** It is the only input that costs a call
  (`ctx.llm.resolveModelInfo`), so the wiring asks for it **only** after the
  finish has been declined by both other rules *and* the failure alone has
  passed every window-independent gate. A healthy turn never pays for it, and a
  resolver that throws leaves the stream untouched (fail-closed, on purpose: the
  guard may not become a new way for a turn to fail).

`classifySaturatedFailures: false` turns this trigger off without touching the
other two.

## The fourth defect (discussion #9434)

A provider can enforce a tighter budget than the window it declares, refuse the
request for exceeding it, and say so in a body that names no cause. The reported
route (`opencode-go` / `deepseek-v4.1-flash`) declares
`contextWindow: 1000000` and a 384,000-token output capability, and refuses
anything whose `prompt_tokens + max_tokens` exceeds `2^20`. Its refusals arrive
as **HTTP 400** with a 31-byte body (`{"model":"deepseek-v4.1-flash"}`, no
`error` field), so the SDK surfaces `400 status code (no body)`; the reporter's
session crossed the real wall at ~664,576 prompt tokens — about 63% of the
advertised window — and every later request failed, until a manual `/compact`
shrank it.

Nothing in the harness can read that refusal: the built-in recovery is gated on
the code alone (`compaction-basic/src/index.ts:194` —
`failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted`), there is no
wording in 31 bytes for the second rule to match, and no `usage` count arrives
for the third to compare — the prompt comes back `INVALID_REQUEST` with
`usage: 0`.

What is left is arithmetic the harness can do on its own side: how big is the
request it just sent? The plugin asks its **own meter** for that number and
compares it against the window the route advertises:

```
error finish, nothing attributed, harness meter >= fullRequestRatio x window
    ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
```

The synthesized failure carries both numbers, the line, and the words the
provider did send:

```
openrouter/deepseek/deepseek-v4.1-flash refused the request as INVALID_REQUEST
while this session's own meter priced it at 673267 token(s) against the
1000000-token window that route advertises (0.67x, at or above the 500000-token
line this rule acts on): 400 status code (no body). Reclassified as
CONTEXT_WINDOW_EXCEEDED by length-stop-overflow (discussion #9434): …
```

### The line is a substitute — the earlier arithmetic is not observable here

Earlier in this thread we stated the predicate as
`prompt_tokens + cap actually reserved on the wire > declared contextWindow`.
That is the provider's rule, and it is what the report measures — but **it
cannot be evaluated at this seam**, and this rule does not pretend to evaluate
it. The cap a request carries is decided below the plugin:

- `llm-pi-ai` forwards one only when the call names it (`adapter.ts:405`), and a
  loop-built call names one only when the session's effective envelope does
  (`llm/src/call-config.ts:23`); otherwise pi-ai sizes the request from the
  catalog's own `maxTokens`;
- `LlmResolvedModelInfo` (`llm/src/types.ts:410`) reports
  `context.contextWindow` and an adapter's `defaultMaxTokens`, and **no** catalog
  output capability — `maxTokens` appears in the type only on a *discovered*
  model (`types.ts:316`), which is a settings-time draft rather than the request
  path;
- and the number actually sent is produced after that, inside pi-ai's
  `clampMaxTokensToContext` on the clamped request path, from a local estimate
  that can sit tens of thousands of tokens below the provider's own count. It is
  not reported back to the harness at all.

So "the cap reserved on the wire" is not a number a plugin can read — nor one
the harness reads, which is precisely why the compaction threshold and the wire
cap are free to disagree. That disagreement is the *other* half of the report,
and it is a core question, not this plugin's: what this rule changes is only
what the **failure means**, so the tree can recover from it.

The two numbers it does own are the harness' own meter —
`tokenMeter.measure(session).totalTokens`, the same measurement
`compaction-basic/src/index.ts:278` compares against its own threshold — and the
resolved window. The line is therefore a **policy, not an identity**:
`fullRequestRatio` (default 0.5) of the declared window, deliberately **below**
the compaction policy's own threshold (default `thresholdRatio: 0.8`), because
the whole defect is that the provider's real input capacity can be smaller than
the window the harness believes in.

Three further notes:

- **It is the only rule that can act on a refusal carrying no count of any
  kind** — trigger 3 reads the provider's arithmetic, trigger 4 the harness'. On
  a real circuit the provider's count is the stronger evidence, so trigger 3 is
  evaluated first and trigger 4 only sees what it declined; the two share one
  window lookup.
- **It is fail-closed three times over**: no `sessions`/`tokenMeter` service, no
  session for the id the request carried, or a measurement that is not a finite
  non-negative number — and nothing happens. The rule is reached through
  cordis' untracked `ctx.get`, so a profile without a meter still gets triggers
  1–3, and a service whose shape drifts declines exactly as an absent one does.
- **It costs nothing on a healthy turn.** The gate is a failure whose code
  attributes nothing; a `max-tokens` finish and an attributed error never ask
  for a measurement or a window.

`classifyFullRequestRefusals: false` turns this trigger off without touching the
other three.

## What the plugin does

Observes the public `llm/stream` waterfall and rewrites the terminal frame of
any of the four defects into the `CONTEXT_WINDOW_EXCEEDED` failure the harness
already knows how to recover from:

```
max-tokens finish               ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
error finish (size refusal)     ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
error finish (saturated prompt) ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
error finish (full request)     ->  { kind: 'error', failure: { code: 'CONTEXT_WINDOW_EXCEEDED', … } }
```

`@deepseek-ai/dsh-compaction-basic` (mounted in `bundle/base` and
`bundle/web-app`) then compacts and returns `{ kind: 'retry' }`, bounded by its
own `maxOverflowRetries` (default 1) — so there is no retry loop.

Three properties make the rewrite exact rather than approximate:

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

The **truncation** rule fires only when a `max-tokens` finish arrives with a
negligible output count. A real output cap cannot look like this: the budget it hits is the
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
- an **unattributed failure on a prompt with room**: no `usage` chunk, a count
  below `saturationRatio` of the window, a route whose capacity cannot be
  resolved, or a capacity that is not a positive integer — all leave the
  failure exactly as it arrived;
- an **unattributed failure on a request the harness' own meter prices below
  the line**: no `sessions`/`tokenMeter` service, no session for the id the
  request carried, a request that carried no session id at all, a route whose
  capacity cannot be resolved, a measurement that is not a finite non-negative
  number, or a price below `fullRequestRatio` of that window — the same
  fail-closed answer, for the same reason: the plugin may not become a new way
  for a turn to fail;
- a rate limit, a transport failure, or a malformed request that merely mentions
  a size noun.

`classifyOversizeRequests: false` turns the second trigger off without touching
the others; `classifySaturatedFailures: false` does the same for the third, and
`classifyFullRequestRefusals: false` for the fourth.

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
        classifySaturatedFailures: true  # default: true
        saturationRatio: 0.99            # default: 0.99
        classifyFullRequestRefusals: true # default: true
        fullRequestRatio: 0.5            # default: 0.5
```

| option | meaning |
|---|---|
| `mode: 'error'` | replace the truncating finish with `CONTEXT_WINDOW_EXCEEDED` so compaction runs (default) |
| `mode: 'warn'` | log the detection and stream unchanged — how to measure your own traffic before enabling |
| `mode: 'off'` | pure pass-through |
| `atMostOutputTokens` | output count at or below which a `max-tokens` finish is a truncation (default 2) |
| `classifyOversizeRequests` | also reclassify a request refused for its size — `413`, or size wording on a request-rejection code (default `true`) |
| `classifySaturatedFailures` | also reclassify an unattributed failure whose prompt count has reached the route's window (default `true`) |
| `saturationRatio` | fraction of the resolved window at which that prompt count counts as saturated (default `0.99`) |
| `classifyFullRequestRefusals` | also reclassify an unattributed failure on a request the harness' own meter prices at `fullRequestRatio` of the window (default `true`) |
| `fullRequestRatio` | fraction of the declared window at which that measurement counts as a full request (default `0.5`) |

`fullRequestRatio` is the one knob set by policy rather than by measurement —
the window it is a fraction of is declared by the route, and the capacity the
provider actually enforces is not observable from here (see *The line is a
substitute* above). Lower it to catch a provider that refuses earlier than its
declared window; raise it if your routes are honest and you would rather the
alert came from trigger 3's provider-side count. A route that refuses earlier
than half its declared window is one whose declared window is wrong, and this
rule is not the right place to assert that: it decides what a refusal *means*,
never what a window *is*.

Raise `saturationRatio`'s bar (e.g. `1.0`) if your route advertises a window far
larger than it enforces; lower it if your provider refuses slightly *before* the
window. The default sits just under 1.0 because a provider that counted a prompt
this full and answered with an unattributed error has already told you what the
prompt was worth.

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
| `length-stop-overflow` ≥0.3.0 (#7632) | an **error** naming nothing, on a prompt the provider counted at the window | `PI_AI_ERROR` |
| `length-stop-overflow` ≥0.4.0 (#9434) | an **error** naming nothing, on a request the harness' own meter priced at half the declared window | `INVALID_REQUEST`, `400 status code (no body)` |

## Tests

```sh
npm test        # tsc, then 134 tests over a real Cordis context
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
  exact `INVALID_REQUEST` — which compacts nothing without the plugin; hop 4: the
  saturation rule through a real mounted context — the reported turn and its
  unmounted control, plus the arms where the route resolves no window, where the
  capacity lookup is refused, and where the prompt has room; hop 5: the
  full-request rule through the real waterfall — the reported turn, and the arms
  that keep it honest: an attributed code at the reported price, a `max-tokens`
  finish at that price, a healthy turn that must not pay for either a
  measurement or a window lookup, the saturation rule taking precedence when the
  provider's own count qualifies, and the two rules sharing one window lookup.
  Hops 4 and 5 stop at the waterfall on purpose: the failure they emit is the
  same `CONTEXT_WINDOW_EXCEEDED_CODE` by value that hop 2 already proved passes
  the recovery's gate, so a further compaction arm would re-measure hop 2's
  subject.)

Plus a packaging guard asserting that the shipped artifact's bare imports and
its runtime dependency declarations match in **both** directions.
