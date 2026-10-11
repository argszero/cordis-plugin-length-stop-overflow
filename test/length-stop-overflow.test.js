/**
 * Overflow guard — behaviour tests.
 *
 * The unit half drives the pure generator; the integration half drives the real
 * `@deepseek-ai/cordis` context and the real `LlmRuntime` from
 * `@deepseek-ai/dsh-llm`, so the assertion is about the shipped `llm/stream`
 * waterfall rather than about this module's own idea of it.
 *
 * Both triggers carry a control arm. A guard that fires on everything is worse
 * than one that fires on nothing, so most of what follows asserts what must be
 * left alone.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { CONTEXT_WINDOW_EXCEEDED_CODE, LlmAdapter } from '@deepseek-ai/dsh-llm'
import * as LlmInvariant from '@deepseek-ai/dsh-llm/invariant'
import InvariantRegistry from '@deepseek-ai/dsh-invariants'
import {
  apply,
  classifyFinish,
  classifyFullRequestRefusal,
  classifyLengthStop,
  classifyOversizeFailure,
  classifySaturatedFailure,
  Config,
  DEFAULT_AT_MOST_OUTPUT_TOKENS,
  DEFAULT_CLASSIFY_FULL_REQUEST_REFUSALS,
  DEFAULT_CLASSIFY_OVERSIZE_REQUESTS,
  DEFAULT_CLASSIFY_SATURATED_FAILURES,
  DEFAULT_FULL_REQUEST_RATIO,
  DEFAULT_MODE,
  DEFAULT_SATURATION_RATIO,
  estimateRequestBytes,
  formatBytes,
  fullRequestFailure,
  guardLengthStopStream,
  guardOverflowStream,
  inject,
  isAttributedStatus,
  isFullRequestCandidate,
  isMaxTokensFinish,
  isRequestTooLargeFailure,
  isSaturatedCandidate,
  measureRequestTokens,
  name,
  overflowFailure,
  oversizeFailure,
  promptTokensOf,
  rewriteFinish,
  saturationFailure,
} from '../lib/index.js'

/** Resolved-config shape the wiring always produces. */
function cfg(over = {}) {
  return {
    mode: 'error',
    atMostOutputTokens: DEFAULT_AT_MOST_OUTPUT_TOKENS,
    classifyOversizeRequests: DEFAULT_CLASSIFY_OVERSIZE_REQUESTS,
    classifySaturatedFailures: DEFAULT_CLASSIFY_SATURATED_FAILURES,
    saturationRatio: DEFAULT_SATURATION_RATIO,
    classifyFullRequestRefusals: DEFAULT_CLASSIFY_FULL_REQUEST_REFUSALS,
    fullRequestRatio: DEFAULT_FULL_REQUEST_RATIO,
    ...over,
  }
}
const WARN = cfg({ mode: 'warn' })
const OFF = cfg({ mode: 'off' })
const ROUTE = { provider: 'openrouter', model: 'deepseek/deepseek-v4.1-flash' }

/** The exact refusal the #7626 reporter's session answered every turn (log line). */
const REPORTED_413 = {
  type: 'finish',
  reason: {
    kind: 'error',
    failure: { code: 'INVALID_REQUEST', message: 'DeepSeek Messages request failed (413)', status: 413 },
  },
}

/** One `error` finish carrying an arbitrary failure. */
function errorFinish(failure) {
  return { type: 'finish', reason: { kind: 'error', failure } }
}

/** Collect an async iterable. */
async function drain(stream) {
  const out = []
  for await (const chunk of stream) out.push(chunk)
  return out
}

/** Build an async iterable from a plain array. */
async function* from(chunks) {
  for (const chunk of chunks) yield chunk
}

/** A `usage` chunk with the counts the report published. */
function usage(over = {}) {
  return { type: 'usage', usage: { inputTokens: 11, outputTokens: 1, totalTokens: 287116, cacheReadTokens: 287104, ...over } }
}

/** A terminal `max-tokens` finish. */
function maxTokensFinish(extra = {}) {
  return { type: 'finish', reason: { kind: 'max-tokens' }, ...extra }
}

/**
 * The reported turn, verbatim in shape: a one-token `length` stop on a 287k
 * prompt (discussion #7214, turn 23).
 */
const REPORTED_TURN = [
  { type: 'block-start', index: 0, blockType: 'reasoning' },
  { type: 'reasoning-delta', index: 0, text: 'The' },
  { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'The' } },
  usage(),
  maxTokensFinish(),
]

/* ------------------------------------------------------------------ *
 * The reported defect
 * ------------------------------------------------------------------ */

test('the reported one-token length stop becomes CONTEXT_WINDOW_EXCEEDED', async () => {
  const out = await drain(guardLengthStopStream(from(REPORTED_TURN), cfg(), ROUTE))
  const finish = out.at(-1)
  assert.equal(finish.type, 'finish')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the synthesized failure names the route and the provider\u2019s own numbers', async () => {
  const out = await drain(guardLengthStopStream(from(REPORTED_TURN), cfg(), ROUTE))
  const { failure } = out.at(-1).reason
  assert.match(failure.message, /openrouter\/deepseek\/deepseek-v4\.1-flash/)
  assert.match(failure.message, /1 output token/)
  // The prompt count is the disjoint sum the provider reported: input + cache.
  assert.match(failure.message, /287115-token prompt/)
  assert.match(failure.message, /length-stop-overflow/)
})

test('every non-terminal chunk is forwarded identically and in order', async () => {
  const out = await drain(guardLengthStopStream(from(REPORTED_TURN), cfg(), ROUTE))
  assert.deepEqual(out.slice(0, -1), REPORTED_TURN.slice(0, -1))
  assert.equal(out.length, REPORTED_TURN.length, 'no chunk may be added or dropped')
})

test('the rewritten finish drops replay metadata, which describes a successful response', async () => {
  // Replay state is only consumed on the success path (agent-loop builds the
  // assistant message from it); the error path settles an attempt and throws, so
  // forwarding a stale envelope would mislead a reader of the durable log.
  const chunks = [...REPORTED_TURN.slice(0, -1), maxTokensFinish({ replayState: { response: { native: 'length' } } })]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.equal('replayState' in out.at(-1), false)
})

test('a zero-output length stop is also reclassified', async () => {
  // pi-ai's own zero-output rule needs `input + cacheRead >= 0.99 * contextWindow`;
  // with a wrongly resolved window that check misses too, and the harness is left
  // with a max-tokens finish whose output is exactly zero.
  const chunks = [{ type: 'usage', usage: { inputTokens: 287115, outputTokens: 0 } }, maxTokensFinish()]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

/* ------------------------------------------------------------------ *
 * What must never be reclassified
 * ------------------------------------------------------------------ */

test('a genuine output-cap stop is left alone', async () => {
  // The budget a real cap hits is the model's own capability: tens of thousands
  // of tokens. Reclassifying one would compact a session for no reason.
  for (const outputTokens of [8192, 32768, 384000]) {
    const chunks = [usage({ outputTokens }), maxTokensFinish()]
    const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
    assert.deepEqual(out, chunks, `${outputTokens} output tokens is an output cap`)
  }
})

test('a successful stop is never reclassified, whatever the usage says', async () => {
  const chunks = [usage({ outputTokens: 1 }), { type: 'finish', reason: { kind: 'stop' } }]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('tool-call and aborted finishes are never reclassified', async () => {
  const chunks = [
    usage({ outputTokens: 1 }),
    { type: 'finish', reason: { kind: 'tool-calls' } },
    { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'cancelled' } } },
    { type: 'finish', reason: { kind: 'error', failure: { code: 'SERVER', message: 'HTTP 500' } } },
  ]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('an already-classified overflow passes through unchanged', async () => {
  const chunks = [
    usage({ outputTokens: 0 }),
    { type: 'finish', reason: { kind: 'error', failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'prompt is too long' } } },
  ]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('a max-tokens finish with no usage chunk fails closed', async () => {
  // Without the provider's own count there is nothing to judge, and a wrong
  // verdict costs a compaction that cannot help.
  const chunks = [maxTokensFinish()]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('a usage chunk arriving after the finish cannot resurrect the verdict', async () => {
  const chunks = [maxTokensFinish(), usage()]
  const out = await drain(guardLengthStopStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('a non-finite or negative output count is refused', () => {
  for (const outputTokens of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    assert.equal(
      classifyLengthStop({ kind: 'max-tokens' }, { inputTokens: 10, outputTokens }, cfg()),
      null,
      `${outputTokens} must not classify`,
    )
  }
})

/* ------------------------------------------------------------------ *
 * The threshold
 * ------------------------------------------------------------------ */

test('the default threshold is the clamp floor plus one', () => {
  assert.equal(DEFAULT_AT_MOST_OUTPUT_TOKENS, 2)
})

test('the threshold is inclusive and configurable', () => {
  const usageChunk = { inputTokens: 10, outputTokens: 3 }
  assert.equal(classifyLengthStop({ kind: 'max-tokens' }, usageChunk, cfg()), null)
  assert.notEqual(classifyLengthStop({ kind: 'max-tokens' }, usageChunk, cfg({ atMostOutputTokens: 3 })), null)
  assert.notEqual(classifyLengthStop({ kind: 'max-tokens' }, usageChunk, cfg({ atMostOutputTokens: 64 })), null)
})

test('a threshold of zero accepts only a zero-output stop', () => {
  const reason = { kind: 'max-tokens' }
  assert.notEqual(classifyLengthStop(reason, { inputTokens: 10, outputTokens: 0 }, cfg({ atMostOutputTokens: 0 })), null)
  assert.equal(classifyLengthStop(reason, { inputTokens: 10, outputTokens: 1 }, cfg({ atMostOutputTokens: 0 })), null)
})

/* ------------------------------------------------------------------ *
 * The prompt-token arithmetic
 * ------------------------------------------------------------------ */

test('the prompt count is the disjoint sum of input and both cache fields', () => {
  assert.equal(promptTokensOf({ inputTokens: 11, outputTokens: 1, cacheReadTokens: 287104 }), 287115)
  assert.equal(promptTokensOf({ inputTokens: 11, outputTokens: 1, cacheReadTokens: 200, cacheWriteTokens: 300 }), 511)
  assert.equal(promptTokensOf({ inputTokens: 7, outputTokens: 1 }), 7)
})

test('the reported turn\u2019s evidence matches the published numbers', () => {
  const evidence = classifyLengthStop(
    { kind: 'max-tokens' },
    { inputTokens: 11, outputTokens: 1, totalTokens: 287116, cacheReadTokens: 287104 },
    cfg(),
  )
  assert.deepEqual(evidence, { rule: 'length-stop', outputTokens: 1, promptTokens: 287115 })
})

test('prompt tokens are absent from the verdict', () => {
  // Deliberately: any rule comparing the prompt against the resolved window
  // inherits that window's error, and the reported session resolved it four
  // times too small. The clamp floor is the window-independent signal.
  const evidence = classifyLengthStop({ kind: 'max-tokens' }, { inputTokens: 3, outputTokens: 1 }, cfg())
  assert.deepEqual(evidence, { rule: 'length-stop', outputTokens: 1, promptTokens: 3 })
})

/* ------------------------------------------------------------------ *
 * Narrowing and failure shape
 * ------------------------------------------------------------------ */

test('isMaxTokensFinish narrows only the max-tokens member', () => {
  assert.equal(isMaxTokensFinish({ kind: 'max-tokens' }), true)
  for (const reason of [{ kind: 'stop' }, { kind: 'tool-calls' }, { kind: 'aborted', failure: { code: 'ABORTED', message: 'x' } }]) {
    assert.equal(isMaxTokensFinish(reason), false)
  }
})

test('the failure carries only code and message', () => {
  const failure = overflowFailure({ outputTokens: 1, promptTokens: 10 }, ROUTE)
  assert.deepEqual(Object.keys(failure).sort(), ['code', 'message'])
})

/* ------------------------------------------------------------------ *
 * Modes
 * ------------------------------------------------------------------ */

test('warn mode reports without changing the stream', async () => {
  const seen = []
  const out = await drain(guardLengthStopStream(from(REPORTED_TURN), WARN, ROUTE, (e) => seen.push(e)))
  assert.deepEqual(out, REPORTED_TURN)
  assert.deepEqual(seen, [{ rule: 'length-stop', outputTokens: 1, promptTokens: 287115 }])
})

test('error mode reports the evidence it reclassified', async () => {
  const seen = []
  await drain(guardLengthStopStream(from(REPORTED_TURN), cfg(), ROUTE, (e) => seen.push(e)))
  assert.deepEqual(seen, [{ rule: 'length-stop', outputTokens: 1, promptTokens: 287115 }])
})

test('off mode is a pure pass-through and does not even evaluate', async () => {
  const seen = []
  const out = await drain(guardLengthStopStream(from(REPORTED_TURN), OFF, ROUTE, (e) => seen.push(e)))
  assert.deepEqual(out, REPORTED_TURN)
  assert.deepEqual(seen, [])
})

test('the detection callback is not invoked for a finish that passes through', async () => {
  const seen = []
  await drain(guardLengthStopStream(from([usage({ outputTokens: 4096 }), maxTokensFinish()]), cfg(), ROUTE, (e) => seen.push(e)))
  assert.deepEqual(seen, [])
})

test('DEFAULT_MODE restores the classification the recovery path keys on', () => {
  assert.equal(DEFAULT_MODE, 'error')
})

/* ------------------------------------------------------------------ *
 * Streaming contract
 * ------------------------------------------------------------------ */

test('chunks are emitted as they arrive — the guard never buffers', async () => {
  const order = []
  async function* tracked() {
    for (let index = 0; index < 3; index += 1) {
      order.push(`src:${index}`)
      if (index === 0) yield usage()
      else if (index === 2) yield maxTokensFinish()
      else yield { type: 'reasoning-delta', index: 0, text: `r${index}` }
    }
  }
  let index = 0
  for await (const chunk of guardLengthStopStream(tracked(), cfg(), ROUTE)) {
    assert.ok(chunk)
    order.push(`out:${index}`)
    index += 1
  }
  assert.deepEqual(order, ['src:0', 'out:0', 'src:1', 'out:1', 'src:2', 'out:2'])
})

test('the guard is lazy: constructing it consumes nothing', async () => {
  let pulled = 0
  async function* counted() {
    pulled += 1
    yield maxTokensFinish()
  }
  const stream = guardLengthStopStream(counted(), cfg(), ROUTE)
  assert.equal(pulled, 0, 'no upstream pull before the first next()')
  await stream[Symbol.asyncIterator]().next()
  assert.equal(pulled, 1)
})

test('an empty upstream stream yields no chunks and does not throw', async () => {
  assert.deepEqual(await drain(guardLengthStopStream(from([]), cfg(), ROUTE)), [])
})

test('an upstream throw propagates rather than being swallowed', async () => {
  async function* boom() {
    yield usage()
    throw new Error('upstream exploded')
  }
  await assert.rejects(() => drain(guardLengthStopStream(boom(), cfg(), ROUTE)), /upstream exploded/)
})

/* ------------------------------------------------------------------ *
 * Config schema
 * ------------------------------------------------------------------ */

test('Config applies every default', () => {
  const resolved = Config({})
  assert.equal(resolved.mode, 'error')
  assert.equal(resolved.atMostOutputTokens, DEFAULT_AT_MOST_OUTPUT_TOKENS)
  assert.equal(resolved.classifyOversizeRequests, DEFAULT_CLASSIFY_OVERSIZE_REQUESTS)
  assert.equal(resolved.classifySaturatedFailures, DEFAULT_CLASSIFY_SATURATED_FAILURES)
  assert.equal(resolved.saturationRatio, DEFAULT_SATURATION_RATIO)
})

test('Config preserves explicit values', () => {
  const resolved = Config({
    mode: 'warn',
    atMostOutputTokens: 64,
    classifyOversizeRequests: false,
    classifySaturatedFailures: false,
    saturationRatio: 0.8,
  })
  assert.equal(resolved.mode, 'warn')
  assert.equal(resolved.atMostOutputTokens, 64)
  assert.equal(resolved.classifyOversizeRequests, false)
  assert.equal(resolved.classifySaturatedFailures, false)
  assert.equal(resolved.saturationRatio, 0.8)
})

test('Config rejects an unknown mode', () => {
  assert.throws(() => Config({ mode: 'silent' }))
})

test('Config rejects a non-integer or negative threshold', () => {
  assert.throws(() => Config({ atMostOutputTokens: 1.5 }))
  assert.throws(() => Config({ atMostOutputTokens: -1 }))
})

test('Config rejects a non-boolean oversize switch', () => {
  assert.throws(() => Config({ classifyOversizeRequests: 'yes' }))
})

test('Config rejects a switch that is not a boolean or a negative ratio', () => {
  assert.throws(() => Config({ classifySaturatedFailures: 'yes' }))
  assert.throws(() => Config({ saturationRatio: -0.1 }))
  assert.throws(() => Config({ saturationRatio: 'a lot' }))
})

/* ------------------------------------------------------------------ *
 * Integration: the real Cordis context and the real LlmRuntime
 * ------------------------------------------------------------------ */

/** An adapter that replays one scripted chunk list for any request. */
class ScriptedAdapter extends LlmAdapter {
  constructor(script) {
    super()
    this.script = script
  }

  async * stream(_options) {
    for (const chunk of this.script) yield chunk
  }
}

/**
 * A scripted adapter that also resolves a context window, which is what a real
 * provider adapter does (`resolveModel`) and what the saturation rule compares
 * the provider's own count against.
 */
class WindowedAdapter extends ScriptedAdapter {
  constructor(script, contextWindow = WINDOW) {
    super(script)
    this.contextWindow = contextWindow
  }

  resolveModel(provider, model) {
    return Promise.resolve({ provider, id: model, name: model, context: { contextWindow: this.contextWindow } })
  }
}

/**
 * Build the real service stack: a Cordis context, the real `LlmRuntime`, the
 * real `llm/stream` waterfall, and a scripted adapter registered on a route.
 * @param script - the chunks the adapter replays.
 * @param config - plugin config; the plugin is mounted when present.
 * @param adapter - the adapter to register; defaults to one that resolves no
 *   route metadata, which is also the arm that proves the saturation rule needs
 *   a window to act.
 * @returns the context.
 */
async function harness(script, config, adapter) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  if (config !== undefined) await ctx.plugin({ name, apply, inject }, config)
  ctx.llm.registerAdapter(['openrouter'], adapter ?? new ScriptedAdapter(script))
  return ctx
}

/** Drive one real stream through the real waterfall. */
async function streamOnce(ctx, extra = {}) {
  const chunks = []
  for await (const chunk of ctx.llm.stream({ provider: 'openrouter', model: ROUTE.model, messages: [], ...extra })) {
    chunks.push(chunk)
  }
  return chunks
}

test('mounted in a real context, the real waterfall rewrites the truncating finish', async () => {
  const ctx = await harness(REPORTED_TURN, {})
  const chunks = await streamOnce(ctx)
  const finish = chunks.at(-1)
  assert.equal(finish.type, 'finish')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.match(finish.reason.failure.message, /openrouter\/deepseek\/deepseek-v4\.1-flash/)
})

test('unmounted, the same stream ends as an ordinary max-tokens turn', async () => {
  // The control arm: without the plugin the reported defect is exactly what the
  // harness produces, which is why the session never recovered.
  const ctx = await harness(REPORTED_TURN, undefined)
  const chunks = await streamOnce(ctx)
  assert.deepEqual(chunks.at(-1).reason, { kind: 'max-tokens' })
})

test('a real output-cap stop is untouched through the waterfall', async () => {
  const ctx = await harness([usage({ outputTokens: 32768 }), maxTokensFinish()], {})
  const chunks = await streamOnce(ctx)
  assert.deepEqual(chunks.at(-1).reason, { kind: 'max-tokens' })
})

test('the plugin is a pass-through when disabled through a real mount', async () => {
  const ctx = await harness(REPORTED_TURN, { mode: 'off' })
  const chunks = await streamOnce(ctx)
  assert.deepEqual(chunks.at(-1).reason, { kind: 'max-tokens' })
})

test('the rewritten finish satisfies the harness\u2019 own stream invariant', async () => {
  // The in-tree invariant validator installs itself as a global, prepended
  // `llm/stream` listener (`llm/src/invariant.ts:88`), which puts it outside this
  // plugin, so it validates what the plugin emits. A finish rewritten to `error`
  // is legal even with an open block, but that has to be run, not assumed.
  const ctx = new Context()
  await ctx.plugin(InvariantRegistry)
  await ctx.plugin(LlmInvariant)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin({ name, apply, inject }, {})
  ctx.llm.registerAdapter(['openrouter'], new ScriptedAdapter(REPORTED_TURN))
  const chunks = await streamOnce(ctx)
  assert.equal(chunks.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the invariant still rejects a broken stream while the plugin is mounted', async () => {
  // The control arm for the test above: the validator really is running, so its
  // acceptance of the rewritten finish is evidence and not a no-op.
  const ctx = new Context()
  await ctx.plugin(InvariantRegistry)
  await ctx.plugin(LlmInvariant)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin({ name, apply, inject }, {})
  ctx.llm.registerAdapter(['openrouter'], new ScriptedAdapter([
    { type: 'text-delta', index: 0, text: 'x' },
    maxTokensFinish(),
  ]))
  await assert.rejects(() => streamOnce(ctx), /requires an open text block/)
})

/* ------------------------------------------------------------------ *
 * The second trigger: a request refused for its size (#7626)
 * ------------------------------------------------------------------ */

/**
 * Wordings that must classify, and the provider each comes from. Kept as a
 * table because this rule's whole risk lives in the wording, not the mechanism.
 */
const SIZE_REFUSALS = [
  ['the adapter\u2019s body-less fallback for a 413', { code: 'INVALID_REQUEST', message: 'DeepSeek Messages request failed (413)' }],
  ['a proxy\u2019s HTML title', { code: 'INVALID_REQUEST', message: '413 Request Entity Too Large', status: 413 }],
  ['a plain JSON payload refusal', { code: 'INVALID_REQUEST', message: 'Payload too large' }],
  ['Anthropic\u2019s prompt-length wording', { code: 'INVALID_REQUEST', message: 'prompt is too long: 210000 tokens > 200000 maximum' }],
  ['a size noun with a bound verb', { code: 'INVALID_REQUEST', message: 'request size exceeded' }],
  ['a reversed vLLM-style wording', { code: 'INVALID_REQUEST', message: 'The request exceeds the maximum size allowed by the model' }],
  ['a status carried by an adapter that maps nothing', { code: 'HTTP_413', message: 'upstream refused', status: 413 }],
  ['a gateway relaying the status in its own text', { code: 'SERVER', message: 'upstream returned 413' }],
]

test('every observed size wording classifies as a request-size refusal', () => {
  for (const [label, failure] of SIZE_REFUSALS) {
    assert.equal(isRequestTooLargeFailure(failure), true, `${label} must classify`)
    const detection = classifyOversizeFailure({ kind: 'error', failure }, cfg())
    assert.equal(detection?.rule, 'request-too-large', `${label} must produce evidence`)
    assert.equal(detection.matched, failure.status === 413 ? 'status' : 'wording')
  }
})

test('the reported 413 becomes CONTEXT_WINDOW_EXCEEDED', async () => {
  const out = await drain(guardOverflowStream(from([REPORTED_413]), cfg(), ROUTE))
  const finish = out.at(-1)
  assert.equal(finish.type, 'finish')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the rewrite keeps the provider\u2019s words, the status and the request id', async () => {
  const chunk = errorFinish({
    code: 'INVALID_REQUEST',
    message: 'DeepSeek Messages request failed (413)',
    status: 413,
    requestId: 'req_01J',
    providerRetryAfterMs: 30_000,
  })
  const out = await drain(guardOverflowStream(from([chunk]), cfg(), ROUTE))
  const { failure } = out.at(-1).reason
  assert.match(failure.message, /openrouter\/deepseek\/deepseek-v4\.1-flash refused the request as too large \(HTTP 413\)/)
  // The provider's own sentence is the only durable trace of what it said.
  assert.match(failure.message, /DeepSeek Messages request failed \(413\)/)
  assert.match(failure.message, /discussion #7626/)
  assert.equal(failure.status, 413)
  assert.equal(failure.requestId, 'req_01J')
  // Waiting does not make an oversized request smaller, and the code is not
  // retryable anyway, so the retry hint is deliberately not carried over.
  assert.equal('providerRetryAfterMs' in failure, false)
  assert.deepEqual(Object.keys(failure).sort(), ['code', 'message', 'requestId', 'status'])
})

test('a refusal with no reported status carries no status key', () => {
  const failure = oversizeFailure(
    { rule: 'request-too-large', providerMessage: 'Payload too large', matched: 'wording' },
    ROUTE,
  )
  assert.match(failure.message, /refused the request as too large:/)
  assert.deepEqual(Object.keys(failure).sort(), ['code', 'message'])
})

test('the measured size is named in the failure when it is known', async () => {
  const out = await drain(
    guardOverflowStream(from([REPORTED_413]), cfg(), ROUTE, undefined, () => 61_800_448),
  )
  const { failure } = out.at(-1).reason
  assert.match(failure.message, /carried 59 MB \(61,800,448 bytes\) of message JSON/)
  // The seam sees one component of a request, never the wire payload, and says so.
  assert.match(failure.message, /estimate of this component/)
})

test('a rewritten refusal still carries exactly one terminal frame', async () => {
  // The retry-wrapper law: a wrapper that can send the turn around again must
  // not leak the discarded attempt's terminal frame. Here the upstream failure
  // is replaced, never followed, and this plugin never re-dispatches a request
  // of its own — the loop and its maxOverflowRetries bound stay in charge.
  const out = await drain(guardOverflowStream(from([{ type: 'text-delta', index: 0, text: 'x' }, REPORTED_413]), cfg(), ROUTE))
  assert.equal(out.filter((chunk) => chunk.type === 'finish').length, 1)
  assert.equal(out.at(-1).type, 'finish')
})

/* ------------------------------------------------------------------ *
 * What the size rule must never claim
 * ------------------------------------------------------------------ */

const NOT_SIZE_REFUSALS = [
  // A cancellation that raced a refusal carries the refusal's own code, which is
  // the shape the `reason.kind` guard exists for: the code and the wording both
  // classify, so nothing but the guard stands between them and a rewrite.
  ['a caller cancellation', { kind: 'aborted', failure: { code: 'INVALID_REQUEST', message: 'payload too large' } }],
  ['an already-classified overflow', { kind: 'error', failure: { code: CONTEXT_WINDOW_EXCEEDED_CODE, message: 'prompt too large' } }],
  ['the image recovery\u2019s own failure', { kind: 'error', failure: { code: 'IMAGE_OFFLOAD_REQUIRED', message: 'request too large: image payload', status: 413 } }],
  ['an undecodable-image rejection', { kind: 'error', failure: { code: 'INVALID_REQUEST', message: 'Failed to load image or audio file' } }],
  ['a size complaint about an image, not the request', { kind: 'error', failure: { code: 'INVALID_REQUEST', message: 'image too large' } }],
  ['a malformed request', { kind: 'error', failure: { code: 'INVALID_REQUEST', message: 'Invalid request: unknown field "foo"' } }],
  ['a token bound, which is a different budget', { kind: 'error', failure: { code: 'INVALID_REQUEST', message: 'input tokens exceeded max_prompt_tokens' } }],
  ['a rate limit that mentions request size', { kind: 'error', failure: { code: 'RATE_LIMIT', message: 'request size limit exceeded', status: 429 } }],
  ['a transient transport failure', { kind: 'error', failure: { code: 'SERVER', message: 'upstream said payload too large', status: 502 } }],
]

test('nothing but a request-size refusal is claimed', () => {
  for (const [label, reason] of NOT_SIZE_REFUSALS) {
    assert.equal(classifyOversizeFailure(reason, cfg()), null, `${label} must pass through`)
  }
})

test('an aborted finish is never rewritten, whatever it says', async () => {
  const chunks = [{ type: 'finish', reason: { kind: 'aborted', failure: { code: 'INVALID_REQUEST', message: 'request too large' } } }]
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE))
  assert.deepEqual(out, chunks)
})

test('the size rule can be switched off entirely', async () => {
  const out = await drain(guardOverflowStream(from([REPORTED_413]), cfg({ classifyOversizeRequests: false }), ROUTE))
  assert.deepEqual(out, [REPORTED_413])
  assert.equal(classifyOversizeFailure(REPORTED_413.reason, cfg({ classifyOversizeRequests: false })), null)
})

test('a length stop is still a length stop while the size rule is off', async () => {
  const out = await drain(guardOverflowStream(from(REPORTED_TURN), cfg({ classifyOversizeRequests: false }), ROUTE))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('warn mode reports a size refusal without changing it', async () => {
  const seen = []
  const out = await drain(guardOverflowStream(from([REPORTED_413]), WARN, ROUTE, (d) => seen.push(d)))
  assert.deepEqual(out, [REPORTED_413])
  assert.deepEqual(seen, [{
    rule: 'request-too-large',
    providerMessage: 'DeepSeek Messages request failed (413)',
    matched: 'status',
    status: 413,
  }])
})

test('off mode does not even evaluate the size rule', async () => {
  const seen = []
  const measured = []
  const out = await drain(guardOverflowStream(from([REPORTED_413]), OFF, ROUTE, (d) => seen.push(d), () => {
    measured.push(1)
    return 1
  }))
  assert.deepEqual(out, [REPORTED_413])
  assert.deepEqual(seen, [])
  assert.deepEqual(measured, [])
})

/* ------------------------------------------------------------------ *
 * Trigger 3 — a saturated request failed as an unnamed error (#7632)
 * ------------------------------------------------------------------ */

/** The window the #7632 reporter configured on his route (`contextWindow: 131072`). */
const WINDOW = 131072

/**
 * The reporter's third sample, verbatim: `input 7277 / cacheRead 185472 /
 * output 16` is a 192,749-token prompt on a 131,072-token window — 1.47x — and
 * the failure he saw on every turn, including the 102 summarization attempts.
 */
const SATURATED_TURN = [
  { type: 'usage', usage: { inputTokens: 7277, outputTokens: 16, totalTokens: 192765, cacheReadTokens: 185472 } },
  errorFinish({ code: 'PI_AI_ERROR', message: 'Response incomplete: length' }),
]

/** The reporter's first sample: 5486 + 166528 = 172,014 tokens, 1.31x. */
const FIRST_SAMPLE_USAGE = {
  type: 'usage',
  usage: { inputTokens: 5486, outputTokens: 999, totalTokens: 173013, cacheReadTokens: 166528 },
}

/** A resolver that answers with the reported window, counting its calls. */
function windowResolver(value = WINDOW) {
  const calls = []
  const resolve = async (provider, model) => {
    calls.push(`${provider}/${model}`)
    return value
  }
  resolve.calls = calls
  return resolve
}

/** Drain one stream through the guard with a window resolver. */
async function drainSaturated(chunks, config = cfg(), resolveWindow = windowResolver()) {
  return drain(guardOverflowStream(from(chunks), config, ROUTE, undefined, undefined, resolveWindow))
}

test('the reported unnamed failure on a saturated prompt becomes CONTEXT_WINDOW_EXCEEDED', async () => {
  const out = await drainSaturated(SATURATED_TURN)
  const finish = out.at(-1)
  assert.equal(finish.type, 'finish')
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the saturation failure names both the provider\u2019s count and the route\u2019s window', async () => {
  const { failure } = (await drainSaturated(SATURATED_TURN)).at(-1).reason
  assert.match(failure.message, /openrouter\/deepseek\/deepseek-v4\.1-flash/)
  assert.match(failure.message, /192749 prompt token/)
  assert.match(failure.message, /131072-token window/)
  assert.match(failure.message, /1\.47x/)
  // The provider's words survive: on this path the harness' own code named
  // nothing, so the words are the only record of what the failing request said.
  assert.match(failure.message, /Response incomplete: length/)
  assert.match(failure.message, /PI_AI_ERROR/)
  assert.match(failure.message, /length-stop-overflow/)
  assert.match(failure.message, /7632/)
})

test('every reported sample classifies, including the least saturated one', async () => {
  for (const [usage_, ratio] of [
    [FIRST_SAMPLE_USAGE, 1.31],
    [{ type: 'usage', usage: { inputTokens: 13671, outputTokens: 16, cacheReadTokens: 171904 } }, 1.42],
    [SATURATED_TURN[0], 1.47],
  ]) {
    const out = await drainSaturated([usage_, errorFinish({ code: 'PI_AI_ERROR', message: 'Response incomplete: length' })])
    assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE, `${ratio}x must classify`)
    assert.match(out.at(-1).reason.failure.message, new RegExp(`${ratio}x`))
  }
})

test('every non-terminal chunk is forwarded identically and in order', async () => {
  const out = await drainSaturated(SATURATED_TURN)
  assert.deepEqual(out.slice(0, -1), SATURATED_TURN.slice(0, -1))
  assert.equal(out.length, SATURATED_TURN.length, 'no chunk may be added or dropped')
})

test('a prompt below the ratio is left alone', async () => {
  // 0.92x of the window: large, but a provider that refuses *this* is refusing
  // something other than the size, and compaction would spend a summarization
  // call proving it.
  const chunks = [{ type: 'usage', usage: { inputTokens: 100000, outputTokens: 16, cacheReadTokens: 20586 } }, ...SATURATED_TURN.slice(-1)]
  assert.equal(promptTokensOf(chunks[0].usage), 120586)
  assert.deepEqual(await drainSaturated(chunks), chunks)
})

test('the ratio is configurable and inclusive at its boundary', async () => {
  const atWindow = [{ type: 'usage', usage: { inputTokens: WINDOW, outputTokens: 16 } }, ...SATURATED_TURN.slice(-1)]
  const belowWindow = [{ type: 'usage', usage: { inputTokens: WINDOW - 1, outputTokens: 16 } }, ...SATURATED_TURN.slice(-1)]
  const ratioOne = cfg({ saturationRatio: 1 })
  assert.equal((await drainSaturated(atWindow, ratioOne)).at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.deepEqual(await drainSaturated(belowWindow, ratioOne), belowWindow)
  // Lowering it widens the rule deliberately: a gateway that refuses while a
  // tenth of the window is still reported as headroom.
  const halfFull = [{ type: 'usage', usage: { inputTokens: 65536, outputTokens: 16 } }, ...SATURATED_TURN.slice(-1)]
  assert.deepEqual(await drainSaturated(halfFull, cfg({ saturationRatio: 0.99 })), halfFull)
  assert.equal(
    (await drainSaturated(halfFull, cfg({ saturationRatio: 0.5 }))).at(-1).reason.failure.code,
    CONTEXT_WINDOW_EXCEEDED_CODE,
  )
})

test('an attributed failure is never reclassified, however full the prompt', async () => {
  // The rule's whole licence is that the code named nothing. A code that names a
  // cause rules a size refusal out (or owns its own recovery).
  const codes = [
    'AUTH', 'RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT', 'STREAM_CLOSED', 'ABORTED',
    'QUOTA', 'EMPTY_RESPONSE', 'INVALID_CREDENTIAL', 'INVARIANT', 'INVALID_ARGS', 'NO_ADAPTER',
    CONTEXT_WINDOW_EXCEEDED_CODE, 'IMAGE_OFFLOAD_REQUIRED',
  ]
  for (const code of codes) {
    const chunks = [SATURATED_TURN[0], errorFinish({ code, message: 'whatever the provider said' })]
    const out = await drainSaturated(chunks)
    assert.deepEqual(out, chunks, `${code} must be left alone`)
  }
})

test('a status that attributes the failure rules the rule out too', async () => {
  // Independent evidence: a provider refuses an oversized request with a 4xx
  // that says so, and a 5xx/429/408/401/403 is its health, its throttle, or a
  // credential. `PI_AI_ERROR` would otherwise be admissible.
  for (const status of [500, 502, 503, 504, 429, 408, 401, 403]) {
    const chunks = [SATURATED_TURN[0], errorFinish({ code: 'PI_AI_ERROR', message: 'Response incomplete: length', status })]
    assert.deepEqual(await drainSaturated(chunks), chunks, `HTTP ${status} must be left alone`)
    assert.equal(isAttributedStatus(status), true)
  }
  for (const status of [400, 404, 413, 422]) {
    assert.equal(isAttributedStatus(status), false, `HTTP ${status} does not attribute a cause`)
  }
  assert.equal(isAttributedStatus(undefined), false)
})

test('INVALID_REQUEST is admitted, because it attributes nothing', async () => {
  // The size rule needs wording on this code because the code is a bag; on a
  // saturated prompt the numbers are the evidence the wording was supposed to
  // provide, which is the whole point of reading usage.
  const failure = { code: 'INVALID_REQUEST', message: '400 Bad Request' }
  const chunks = [SATURATED_TURN[0], errorFinish(failure)]
  const out = await drainSaturated(chunks)
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  // Contrast: the same failure on a prompt with room is left alone, which is the
  // arm the size rule would also take (no size wording anywhere in the message).
  const roomy = [{ type: 'usage', usage: { inputTokens: 1024, outputTokens: 16 } }, errorFinish(failure)]
  assert.deepEqual(await drainSaturated(roomy), roomy)
})

test('a failure with no usage chunk fails closed', async () => {
  const chunks = [errorFinish({ code: 'PI_AI_ERROR', message: 'Response incomplete: length' })]
  assert.deepEqual(await drainSaturated(chunks), chunks)
})

test('an unresolvable window fails closed', async () => {
  // `GenerateOptions` carries no window, so a route whose adapter reports none
  // leaves the rule with nothing to compare against. The finished stream keeps
  // the code the adapter chose.
  assert.deepEqual(await drainSaturated(SATURATED_TURN, cfg(), async () => undefined), SATURATED_TURN)
})

test('a resolver that throws propagates — failing closed is the wiring\u2019s job', async () => {
  // The generator stays a pure function of its inputs; `apply` is where an
  // adapter lookup is caught and turned into `undefined`.
  await assert.rejects(
    drainSaturated(SATURATED_TURN, cfg(), async () => { throw new Error('NO_ADAPTER') }),
    /NO_ADAPTER/,
  )
})

test('a max-tokens finish never reaches the saturation rule', async () => {
  // The truncation rule owns that shape. If saturation could claim it, raising
  // `saturationRatio` would silently widen rule 1 to any output count.
  const resolve = windowResolver()
  const chunks = [SATURATED_TURN[0], maxTokensFinish()]
  assert.deepEqual(await drainSaturated(chunks, cfg(), resolve), chunks)
  assert.deepEqual(resolve.calls, [], 'the window must not even be resolved')
})

test('a successful stop is never reclassified, whatever the prompt says', async () => {
  const resolve = windowResolver()
  const chunks = [SATURATED_TURN[0], { type: 'finish', reason: { kind: 'stop' } }]
  assert.deepEqual(await drainSaturated(chunks, cfg(), resolve), chunks)
  assert.deepEqual(resolve.calls, [])
})

test('the window is resolved only when the cheap gates pass', async () => {
  // Resolving costs an adapter call, so the rule runs last and pays only when it
  // can still change the verdict.
  const attributed = windowResolver()
  await drainSaturated([SATURATED_TURN[0], errorFinish({ code: 'SERVER', message: 'HTTP 500' })], cfg(), attributed)
  assert.deepEqual(attributed.calls, [], 'an attributed code must not resolve a window')

  const noUsage = windowResolver()
  await drainSaturated([errorFinish({ code: 'PI_AI_ERROR', message: 'Response incomplete: length' })], cfg(), noUsage)
  assert.deepEqual(noUsage.calls, [], 'no provider count, nothing to compare')

  const declined = windowResolver()
  await drainSaturated([{ type: 'usage', usage: { inputTokens: 1024, outputTokens: 4 } }, ...SATURATED_TURN.slice(-1)], cfg(), declined)
  assert.deepEqual(declined.calls, [`${ROUTE.provider}/${ROUTE.model}`], 'resolved once, then declined on the comparison')

  const claimed = windowResolver()
  await drainSaturated([REPORTED_413], cfg(), claimed)
  assert.deepEqual(claimed.calls, [], 'a finish another rule already claimed never resolves one')

  const saturated = windowResolver()
  await drainSaturated(SATURATED_TURN, cfg(), saturated)
  assert.deepEqual(saturated.calls, [`${ROUTE.provider}/${ROUTE.model}`])
})

test('the saturation rule can be switched off entirely', async () => {
  const resolve = windowResolver()
  const out = await drainSaturated(SATURATED_TURN, cfg({ classifySaturatedFailures: false }), resolve)
  assert.deepEqual(out, SATURATED_TURN)
  assert.deepEqual(resolve.calls, [], 'switched off means not even evaluated')
})

test('warn mode reports the saturation evidence without changing the stream', async () => {
  const seen = []
  const out = await drain(guardOverflowStream(from(SATURATED_TURN), WARN, ROUTE, (d) => seen.push(d), undefined, windowResolver()))
  assert.deepEqual(out, SATURATED_TURN)
  assert.equal(seen.length, 1)
  assert.deepEqual(seen[0], {
    rule: 'context-saturated',
    promptTokens: 192749,
    contextWindow: WINDOW,
    code: 'PI_AI_ERROR',
    providerMessage: 'Response incomplete: length',
  })
})

test('off mode does not even resolve a window', async () => {
  const resolve = windowResolver()
  assert.deepEqual(await drainSaturated(SATURATED_TURN, OFF, resolve), SATURATED_TURN)
  assert.deepEqual(resolve.calls, [])
})

test('a rewritten saturation failure still carries exactly one terminal frame', async () => {
  const out = await drainSaturated(SATURATED_TURN)
  assert.equal(out.filter((chunk) => chunk.type === 'finish').length, 1)
  assert.equal(out.at(-1).type, 'finish')
})

test('rewriteFinish routes a saturation detection to its own failure text', () => {
  const detection = classifySaturatedFailure(
    SATURATED_TURN[1].reason.failure,
    SATURATED_TURN[0].usage,
    WINDOW,
    cfg(),
  )
  assert.equal(detection.rule, 'context-saturated')
  const rewritten = rewriteFinish(SATURATED_TURN[1], detection, ROUTE)
  assert.equal(rewritten.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.equal(rewritten.reason.failure.message, saturationFailure(detection, ROUTE).message)
})

/* ------------------------------------------------------------------ *
 * The saturation rule's own units
 * ------------------------------------------------------------------ */

test('isSaturatedCandidate answers the window-independent gates', () => {
  const usage = SATURATED_TURN[0].usage
  const unattributed = { code: 'PI_AI_ERROR', message: 'Response incomplete: length' }
  assert.equal(isSaturatedCandidate(unattributed, usage, cfg()), true)
  assert.equal(isSaturatedCandidate(unattributed, undefined, cfg()), false, 'no provider count')
  assert.equal(isSaturatedCandidate({ code: 'RATE_LIMIT', message: '429' }, usage, cfg()), false)
  assert.equal(isSaturatedCandidate({ ...unattributed, status: 503 }, usage, cfg()), false)
  assert.equal(isSaturatedCandidate(unattributed, usage, cfg({ classifySaturatedFailures: false })), false)
  // A usage whose counts are not numbers has nothing to compare.
  assert.equal(isSaturatedCandidate(unattributed, { inputTokens: Number.NaN, outputTokens: 0 }, cfg()), false)
})

test('classifySaturatedFailure refuses a window it cannot trust', () => {
  const failure = SATURATED_TURN[1].reason.failure
  const usage = SATURATED_TURN[0].usage
  for (const window of [undefined, 0, -1, 1.5, Number.NaN]) {
    assert.equal(classifySaturatedFailure(failure, usage, window, cfg()), null, `window ${window}`)
  }
  assert.notEqual(classifySaturatedFailure(failure, usage, WINDOW, cfg()), null)
})

test('the saturation evidence carries the status and the request id when there are any', () => {
  const detection = classifySaturatedFailure(
    { code: 'PI_AI_ERROR', message: 'Response incomplete: length', status: 400, requestId: 'req_123' },
    SATURATED_TURN[0].usage,
    WINDOW,
    cfg(),
  )
  assert.equal(detection.status, 400)
  assert.equal(detection.requestId, 'req_123')
  assert.equal('status' in saturationFailure({ ...detection, status: undefined }, ROUTE), false)
})

/* ------------------------------------------------------------------ *
 * The byte measurement
 * ------------------------------------------------------------------ */

test('the request is measured only when a detection fires', async () => {
  let calls = 0
  const measure = () => { calls += 1; return 4096 }
  // A healthy turn: the size rule never runs, so nothing is ever serialized.
  await drain(guardOverflowStream(from([usage({ outputTokens: 4096 }), maxTokensFinish()]), cfg(), ROUTE, undefined, measure))
  assert.equal(calls, 0, 'a passing request must not be measured')
  // A truncation is not a size question either — same rule, different budget.
  await drain(guardOverflowStream(from(REPORTED_TURN), cfg(), ROUTE, undefined, measure))
  assert.equal(calls, 0)
  await drain(guardOverflowStream(from([REPORTED_413]), cfg(), ROUTE, undefined, measure))
  assert.equal(calls, 1)
})

test('a request that cannot be measured still classifies', async () => {
  const out = await drain(guardOverflowStream(from([REPORTED_413]), cfg(), ROUTE, undefined, () => undefined))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.equal(/message JSON/.test(out.at(-1).reason.failure.message), false)
})

test('estimateRequestBytes measures the message array as UTF-8', () => {
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'continue — 続き' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'ok' }] },
  ]
  assert.equal(estimateRequestBytes({ messages }), new TextEncoder().encode(JSON.stringify(messages)).length)
})

test('estimateRequestBytes reports nothing rather than guessing', () => {
  assert.equal(estimateRequestBytes({ messages: 'not an array' }), undefined)
  assert.equal(estimateRequestBytes({}), undefined)
  const cyclic = {}
  cyclic.self = cyclic
  assert.equal(estimateRequestBytes({ messages: cyclic }), undefined)
})

test('formatBytes keeps the exact count next to the human scale', () => {
  assert.equal(formatBytes(61_800_448), '59 MB (61,800,448 bytes)')
  assert.equal(formatBytes(8_912_896), '8.5 MB (8,912,896 bytes)')
  assert.equal(formatBytes(20 * 1024 * 1024), '20 MB (20,971,520 bytes)')
  assert.equal(formatBytes(0), '0.0 MB (0 bytes)')
})

/* ------------------------------------------------------------------ *
 * Dispatch between the two rules
 * ------------------------------------------------------------------ */

test('classifyFinish sends each finish to its own rule', () => {
  assert.equal(classifyFinish({ kind: 'max-tokens' }, { inputTokens: 1, outputTokens: 1 }, cfg())?.rule, 'length-stop')
  assert.equal(classifyFinish(REPORTED_413.reason, undefined, cfg())?.rule, 'request-too-large')
  // An `error` finish that is not a size refusal, and a `stop` that is neither.
  assert.equal(classifyFinish(errorFinish({ code: 'SERVER', message: 'HTTP 500' }).reason, undefined, cfg()), null)
  assert.equal(classifyFinish({ kind: 'stop' }, undefined, cfg()), null)
})

test('rewriteFinish drops the replay metadata of the frame it replaces', () => {
  // An error finish should not carry replay state at all, but a hostile or
  // unusual adapter is not a reason to forward an envelope that describes a
  // successful response.
  const chunk = { ...errorFinish({ code: 'INVALID_REQUEST', message: 'payload too large' }), replayState: { response: { native: 'x' } } }
  const rewritten = rewriteFinish(chunk, classifyOversizeFailure(chunk.reason, cfg()), ROUTE)
  assert.equal('replayState' in rewritten, false)
})

test('the 0.1.0 generator name still points at the guard', () => {
  assert.equal(guardLengthStopStream, guardOverflowStream)
})

/* ------------------------------------------------------------------ *
 * Integration: the size refusal through the real waterfall
 * ------------------------------------------------------------------ */

test('mounted in a real context, a 413 turn reaches the caller as CONTEXT_WINDOW_EXCEEDED', async () => {
  const ctx = await harness([REPORTED_413], {})
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.equal(finish.reason.failure.status, 413)
})

test('control: unmounted, the same 413 turn reaches the caller as INVALID_REQUEST', async () => {
  // The reported defect, reproduced without the plugin: the code the recovery
  // path does not read, on a session that then cannot send anything.
  const ctx = await harness([REPORTED_413], undefined)
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, 'INVALID_REQUEST')
})

test('the real waterfall measures the request it failed on', async () => {
  const ctx = await harness([REPORTED_413], {})
  const finish = (await streamOnce(ctx, {
    messages: [{ role: 'user', content: [{ type: 'text', text: 'continue' }] }],
  })).at(-1)
  assert.match(finish.reason.failure.message, /of message JSON/)
})

test('a genuine server failure is untouched through the real waterfall', async () => {
  const ctx = await harness([errorFinish({ code: 'SERVER', message: 'HTTP 500' })], {})
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'SERVER')
})

test('the rewritten refusal satisfies the harness\u2019 own stream invariant', async () => {
  const ctx = new Context()
  await ctx.plugin(InvariantRegistry)
  await ctx.plugin(LlmInvariant)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin({ name, apply, inject }, {})
  ctx.llm.registerAdapter(['openrouter'], new ScriptedAdapter([REPORTED_413]))
  const chunks = await streamOnce(ctx)
  assert.equal(chunks.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the plugin is a pass-through for a 413 when disabled through a real mount', async () => {
  const ctx = await harness([REPORTED_413], { classifyOversizeRequests: false })
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'INVALID_REQUEST')
})

/* ------------------------------------------------------------------ *
 * Integration: the saturated, unnamed failure through the real waterfall
 * ------------------------------------------------------------------ */

test('mounted in a real context, the reported turn reaches the caller as CONTEXT_WINDOW_EXCEEDED', async () => {
  // Hop 1 of #7632 through the real runtime: the route resolves its window the
  // way a real adapter does, and the provider's own count saturates it.
  const ctx = await harness(SATURATED_TURN, {}, new WindowedAdapter(SATURATED_TURN))
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.match(finish.reason.failure.message, /192749 prompt token/)
  assert.match(finish.reason.failure.message, /discussion #7632/)
})

test('control: unmounted, the same turn is the opaque PI_AI_ERROR the report describes', async () => {
  const ctx = await harness(SATURATED_TURN, undefined, new WindowedAdapter(SATURATED_TURN))
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, 'PI_AI_ERROR')
  assert.equal(finish.reason.failure.message, 'Response incomplete: length')
})

test('control: mounted, a route that resolves no window leaves the code alone', async () => {
  // The base adapter's `resolveModel` answer (and a scripted adapter's), which is
  // also what an adapter that knows nothing about the route returns: no capacity,
  // no verdict. Fail-closed, measured through the real waterfall.
  const ctx = await harness(SATURATED_TURN, {})
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'PI_AI_ERROR')
})

test('mounted, a route whose capacity lookup fails leaves the code alone', async () => {
  // The wiring catches a rejecting lookup rather than letting it break the
  // stream. The runtime resolves the route before dispatching (`prepareCall`),
  // so only the second lookup — the guard's — fails here; the assertion on the
  // count keeps that ordering explicit instead of implied.
  class FlakyLookupAdapter extends ScriptedAdapter {
    calls = 0

    resolveModel(provider, model) {
      this.calls += 1
      if (this.calls > 1) return Promise.reject(new Error('NO_ADAPTER: unknown route'))
      return Promise.resolve({ provider, id: model, name: model, context: { contextWindow: WINDOW } })
    }
  }
  const adapter = new FlakyLookupAdapter(SATURATED_TURN)
  const ctx = await harness(SATURATED_TURN, {}, adapter)
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'PI_AI_ERROR')
  assert.equal(adapter.calls, 2, 'the guard must have asked, and been refused')
})

test('mounted, a failure on a prompt with room is untouched through the real waterfall', async () => {
  const roomy = [{ type: 'usage', usage: { inputTokens: 1024, outputTokens: 16 } }, ...SATURATED_TURN.slice(-1)]
  const ctx = await harness(roomy, {}, new WindowedAdapter(roomy))
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'PI_AI_ERROR')
})

test('a server failure on a saturated prompt is untouched through the real waterfall', async () => {
  // The adversarial shape for this rule: the numbers say saturated and the code
  // says the provider broke. The code wins, because its recovery is the retry
  // policy's and a compaction would not fix a 500.
  const chunks = [SATURATED_TURN[0], errorFinish({ code: 'SERVER', message: 'HTTP 500' })]
  const ctx = await harness(chunks, {}, new WindowedAdapter(chunks))
  const finish = (await streamOnce(ctx)).at(-1)
  assert.equal(finish.reason.failure.code, 'SERVER')
})

test('the rewritten saturation failure satisfies the harness\u2019 own stream invariant', async () => {
  const ctx = new Context()
  await ctx.plugin(InvariantRegistry)
  await ctx.plugin(LlmInvariant)
  await ctx.plugin(LlmRuntime)
  await ctx.plugin({ name, apply, inject }, {})
  ctx.llm.registerAdapter(['openrouter'], new WindowedAdapter(SATURATED_TURN))
  const chunks = await streamOnce(ctx)
  assert.equal(chunks.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('a mount that drops the module\u2019s inject declaration fails loudly', async () => {
  // Measured, not assumed. `inject` rides on the plugin *object* handed to
  // `ctx.plugin` — the module namespace the loader imports — so a hand-rolled
  // mount that spreads only `{ name, apply }` loses it, and cordis then refuses
  // the service read ("cannot get property 'llm' without inject"). Reading it at
  // wiring time turns that into a mount error; reading it lazily would have left
  // the saturation rule dead inside every stream, reporting nothing.
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await assert.rejects(async () => { await ctx.plugin({ name, apply }, {}) }, /without inject/)
})

/* ------------------------------------------------------------------ *
 * Trigger 4 — a full request refused with nothing to read (#9434)
 * ------------------------------------------------------------------ */

/**
 * The reported route: `opencode-go` / `deepseek-v4.1-flash` with a declared
 * window of 1,000,000, a catalog output capability of 384,000, and a provider
 * that refuses anything whose `prompt_tokens + max_tokens` exceeds `2^20`.
 */
const FULL_WINDOW = 1_000_000

/** The failing request's price, as the reporter measured it (~673,267 tokens). */
const REPORTED_PROMPT_TOKENS = 673_267

/**
 * The failure the reporter's session answered every turn with, verbatim in
 * shape: HTTP 400, a 31-byte body carrying no `error` field, `usage: 0`, and the
 * `INVALID_REQUEST` the adapter derives from a request rejection.
 */
const REPORTED_400 = {
  type: 'finish',
  reason: {
    kind: 'error',
    failure: { code: 'INVALID_REQUEST', message: '400 status code (no body)' },
  },
}

/** The `usage` chunk the reported failures carried: the provider counted nothing. */
const ZERO_USAGE = { type: 'usage', usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } }

/** A measurement delegate that answers with a fixed price and counts its calls. */
function measurement(value = REPORTED_PROMPT_TOKENS) {
  const calls = []
  const measure = () => {
    calls.push(value)
    return value
  }
  measure.calls = calls
  return measure
}

/** Drain one stream through the guard with a measurement delegate. */
async function drainFull(chunks, config = cfg(), resolveWindow = windowResolver(FULL_WINDOW), measureRequest = measurement()) {
  return drain(guardOverflowStream(from(chunks), config, ROUTE, undefined, undefined, resolveWindow, measureRequest))
}

test('the reported unattributed 400 on a full request becomes CONTEXT_WINDOW_EXCEEDED', async () => {
  const out = await drainFull([ZERO_USAGE, REPORTED_400])
  const finish = out.at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('the full-request failure names the harness\u2019 own numbers, the line and the provider\u2019s words', async () => {
  const { failure } = (await drainFull([ZERO_USAGE, REPORTED_400])).at(-1).reason
  assert.match(failure.message, /openrouter\/deepseek\/deepseek-v4\.1-flash/)
  assert.match(failure.message, /673267 token/)
  assert.match(failure.message, /1000000-token window/)
  assert.match(failure.message, /0\.67x/)
  // The line has to be in the text: this verdict is a policy, and a reader must
  // be able to see how far above the line the request sat.
  assert.match(failure.message, /500000-token line/)
  assert.match(failure.message, /400 status code \(no body\)/)
  assert.match(failure.message, /INVALID_REQUEST/)
  assert.match(failure.message, /length-stop-overflow/)
  assert.match(failure.message, /9434/)
})

test('every non-terminal chunk is forwarded identically and in order', async () => {
  const chunks = [ZERO_USAGE, REPORTED_400]
  const out = await drainFull(chunks)
  assert.deepEqual(out.slice(0, -1), chunks.slice(0, -1))
})

test('a rewritten full-request failure still carries exactly one terminal frame', async () => {
  const out = await drainFull([ZERO_USAGE, REPORTED_400])
  assert.equal(out.filter((chunk) => chunk.type === 'finish').length, 1)
  assert.equal(out.at(-1).type, 'finish')
})

test('the caller\u2019s status and request id survive the rewrite', async () => {
  const failure = {
    code: 'INVALID_REQUEST',
    message: '400 status code (no body)',
    status: 400,
    requestId: 'req_9434',
  }
  const { failure: rewritten } = (await drainFull([ZERO_USAGE, errorFinish(failure)])).at(-1).reason
  assert.equal(rewritten.status, 400)
  assert.equal(rewritten.requestId, 'req_9434')
  // Waiting does not make an oversized request smaller, and this code is not
  // retryable — the same reason the other two rewrites drop it.
  assert.equal(rewritten.providerRetryAfterMs, undefined)
})

test('rewriteFinish routes a full-request detection to its own failure text', () => {
  const detection = classifyFullRequestRefusal(
    { code: 'INVALID_REQUEST', message: '400 status code (no body)' },
    REPORTED_PROMPT_TOKENS,
    FULL_WINDOW,
    cfg(),
  )
  assert.equal(detection.rule, 'full-request-refused')
  const rewritten = rewriteFinish({ type: 'finish', reason: { kind: 'error', failure: { code: 'X', message: 'y' } } }, detection, ROUTE)
  assert.equal(rewritten.reason.failure.message, fullRequestFailure(detection, ROUTE).message)
})

/* ---- the gates ---- */

test('the full-request gates accept an unattributed request rejection', () => {
  assert.equal(isFullRequestCandidate({ code: 'INVALID_REQUEST', message: '400 status code (no body)' }, cfg()), true)
  assert.equal(isFullRequestCandidate({ code: 'PI_AI_ERROR', message: 'boom' }, cfg()), true)
})

test('the full-request gates refuse a failure another cause already owns', () => {
  for (const code of ['QUOTA', 'EMPTY_RESPONSE', 'AUTH', 'RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT', 'STREAM_CLOSED', 'INVARIANT', 'INVALID_ARGS', 'NO_ADAPTER', 'CONTEXT_WINDOW_EXCEEDED', 'IMAGE_OFFLOAD_REQUIRED']) {
    assert.equal(isFullRequestCandidate({ code, message: 'x' }, cfg()), false, `${code} must be left alone`)
  }
})

test('the full-request gates refuse a status that attributes the failure', () => {
  for (const status of [500, 503, 401, 403, 408, 429]) {
    assert.equal(isFullRequestCandidate({ code: 'INVALID_REQUEST', message: 'x', status }, cfg()), false)
  }
  // A 400 is the shape this rule exists for: the provider refused the request
  // itself, and nothing in the status says why.
  assert.equal(isFullRequestCandidate({ code: 'INVALID_REQUEST', message: 'x', status: 400 }, cfg()), true)
})

test('the full-request rule can be switched off entirely', () => {
  assert.equal(isFullRequestCandidate({ code: 'INVALID_REQUEST', message: 'x' }, cfg({ classifyFullRequestRefusals: false })), false)
  assert.equal(
    classifyFullRequestRefusal({ code: 'INVALID_REQUEST', message: 'x' }, REPORTED_PROMPT_TOKENS, FULL_WINDOW, cfg({ classifyFullRequestRefusals: false })),
    null,
  )
})

/* ---- the line ---- */

test('the reported price is above the line and a small request is below it', () => {
  const failure = { code: 'INVALID_REQUEST', message: '400 status code (no body)' }
  assert.notEqual(classifyFullRequestRefusal(failure, REPORTED_PROMPT_TOKENS, FULL_WINDOW, cfg()), null)
  assert.equal(classifyFullRequestRefusal(failure, 499_999, FULL_WINDOW, cfg()), null)
})

test('the line is inclusive and is floor(fullRequestRatio x window)', () => {
  const failure = { code: 'INVALID_REQUEST', message: 'x' }
  const atLine = classifyFullRequestRefusal(failure, 500_000, FULL_WINDOW, cfg())
  assert.equal(atLine.threshold, 500_000)
  assert.equal(atLine.measuredTokens, 500_000)
  assert.equal(classifyFullRequestRefusal(failure, 499_999, FULL_WINDOW, cfg()), null)
  // A non-integral line floors, so the threshold never claims more room than the
  // ratio granted: 0.4999 of a 1,000-token window is a 499-token line, and a
  // 499-token request is therefore *at* it while a 498-token one is not.
  assert.equal(classifyFullRequestRefusal(failure, 498, 1_000, cfg({ fullRequestRatio: 0.4999 })), null)
  assert.equal(classifyFullRequestRefusal(failure, 499, 1_000, cfg({ fullRequestRatio: 0.4999 }))?.threshold, 499)
})

test('raising fullRequestRatio narrows the rule without touching the others', async () => {
  const chunks = [ZERO_USAGE, REPORTED_400]
  const raised = cfg({ fullRequestRatio: 0.9 })
  // 673,267 of 1,000,000 is below a 0.9 line — the operator asked for a rule
  // that only acts on a request that has essentially filled the window.
  assert.deepEqual(await drainFull(chunks, raised), chunks)
  // And the same turn still classifies at the default line, so the knob moved
  // this rule and nothing else.
  assert.equal((await drainFull(chunks, cfg())).at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

/* ---- fail-closed arms ---- */

test('the full-request rule fails closed without a measurement', () => {
  const failure = { code: 'INVALID_REQUEST', message: 'x' }
  assert.equal(classifyFullRequestRefusal(failure, undefined, FULL_WINDOW, cfg()), null)
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    assert.equal(classifyFullRequestRefusal(failure, bad, FULL_WINDOW, cfg()), null, `${String(bad)} must not classify`)
  }
})

test('the full-request rule fails closed without a usable window', () => {
  const failure = { code: 'INVALID_REQUEST', message: 'x' }
  for (const bad of [undefined, 0, -1, 1.5]) {
    assert.equal(classifyFullRequestRefusal(failure, REPORTED_PROMPT_TOKENS, bad, cfg()), null, `${String(bad)} must not classify`)
  }
})

test('an unattributed refusal is left alone when no measurement can be read', async () => {
  const chunks = [ZERO_USAGE, REPORTED_400]
  const noMeasure = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, undefined, undefined, windowResolver(FULL_WINDOW)))
  assert.deepEqual(noMeasure, chunks)
})

test('an unattributed refusal is left alone when the route resolves no window', async () => {
  const chunks = [ZERO_USAGE, REPORTED_400]
  // Written out rather than reused: `windowResolver(value = WINDOW)` defaults an
  // explicit `undefined` back to the reported window, which is exactly the answer
  // this arm has to withhold.
  const resolve = async () => undefined
  const noWindow = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, undefined, undefined, resolve, measurement()))
  assert.deepEqual(noWindow, chunks)
})

test('an attributed failure is left alone even at the reported price', async () => {
  // The adversarial shape for this rule: the request is full by our own meter,
  // and the code says the provider timed out. The code wins, because a
  // compaction would not fix a timeout — and it wins *before* the measurement,
  // which is the other half of that claim: the window-free gate is what keeps a
  // failure another recovery owns from costing a pass over the session surface
  // at all.
  const measure = measurement()
  const resolve = windowResolver(FULL_WINDOW)
  const chunks = [ZERO_USAGE, errorFinish({ code: 'TIMEOUT', message: 'stream idle timeout' })]
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, undefined, undefined, resolve, measure))
  assert.deepEqual(out, chunks)
  assert.deepEqual(measure.calls, [])
  assert.deepEqual(resolve.calls, [])
})

test('a healthy finish costs neither a measurement nor a window lookup', async () => {
  const measure = measurement()
  const resolve = windowResolver(FULL_WINDOW)
  const done = [{ type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }, { type: 'finish', reason: { kind: 'stop' } }]
  const out = await drain(guardOverflowStream(from(done), cfg(), ROUTE, undefined, undefined, resolve, measure))
  assert.deepEqual(out, done)
  assert.deepEqual(measure.calls, [])
  assert.deepEqual(resolve.calls, [])
})

test('a max-tokens finish never reaches the full-request rule', async () => {
  // The truncation rule owns that shape, and this one must not widen it behind
  // an operator's back: a big prompt with a real output cap is not an overflow.
  const measure = measurement()
  const resolve = windowResolver(FULL_WINDOW)
  const chunks = [{ type: 'usage', usage: { inputTokens: 673_267, outputTokens: 9_000 } }, { type: 'finish', reason: { kind: 'max-tokens' } }]
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, undefined, undefined, resolve, measure))
  assert.deepEqual(out, chunks)
  assert.deepEqual(measure.calls, [])
  assert.deepEqual(resolve.calls, [])
})

/* ---- ordering and sharing with the saturation rule ---- */

test('the saturation rule acts first when the provider\u2019s own count qualifies', async () => {
  // Both rules would claim this finish: the provider counted 990,000 of a
  // 1,000,000-token window, and our own meter says the same. The provider's
  // count is the stronger evidence, so its evidence is what is reported — and
  // the measurement is never taken.
  const measure = measurement(990_000)
  const resolve = windowResolver(FULL_WINDOW)
  const chunks = [
    { type: 'usage', usage: { inputTokens: 990_000, outputTokens: 0 } },
    errorFinish({ code: 'PI_AI_ERROR', message: 'boom' }),
  ]
  const seen = []
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, (detection) => seen.push(detection.rule), undefined, resolve, measure))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.deepEqual(seen, ['context-saturated'])
  assert.deepEqual(measure.calls, [])
  assert.deepEqual(resolve.calls, [`${ROUTE.provider}/${ROUTE.model}`])
})

test('the measurement fails closed on every reading it does not own', () => {
  // The rule's evidence comes from two services this plugin does not own, so the
  // seam is exercised directly: an absent service, a store that does not hold the
  // id, a meter that throws, and a meter that does not answer with a number all
  // have to produce the same `undefined`, because `undefined` is what makes the
  // rule decline instead of guessing a price.
  const session = { id: 'session-1' }
  const ctxWith = (services) => ({ get: (name) => services[name] })
  const meter = (value) => ({ measure: () => ({ totalTokens: value }) })
  const held = { sessions: { get: () => session } }

  assert.equal(measureRequestTokens(ctxWith({}), 'session-1'), undefined, 'no services mounted')
  assert.equal(measureRequestTokens(ctxWith({ sessions: { get: () => undefined }, tokenMeter: meter(10) }), 'session-1'), undefined, 'store does not hold the id')
  assert.equal(measureRequestTokens(ctxWith({ sessions: held.sessions }), 'session-1'), undefined, 'no meter mounted')
  assert.equal(
    measureRequestTokens(ctxWith({ ...held, tokenMeter: { measure: () => { throw new Error('not priceable') } } }), 'session-1'),
    undefined,
    'a meter that throws is one this rule cannot judge',
  )
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -1, undefined]) {
    assert.equal(measureRequestTokens(ctxWith({ ...held, tokenMeter: meter(bad) }), 'session-1'), undefined, `${String(bad)} must not be a price`)
  }
  assert.equal(measureRequestTokens(ctxWith({ ...held, tokenMeter: meter(0) }), 'session-1'), 0, 'zero is a price')
  assert.equal(measureRequestTokens(ctxWith({ ...held, tokenMeter: meter(4508) }), 'session-1'), 4508)
  assert.equal(measureRequestTokens(ctxWith({ ...held, tokenMeter: meter(4508) }), undefined), undefined, 'a request that carried no session id')
})

test('an already-classified finish is never relabelled by a rule that runs later', async () => {
  // The size rule owns this turn, and the provider's own count is saturated on
  // top of it. Both would classify, and exactly one may: the first match wins,
  // which is what keeps a specific verdict from being overwritten by a general
  // one downstream of it.
  const chunks = [
    { type: 'usage', usage: { inputTokens: 990_000, outputTokens: 0 } },
    errorFinish({ code: 'INVALID_REQUEST', message: 'request entity too large', status: 413 }),
  ]
  const seen = []
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, (detection) => seen.push(detection), undefined, windowResolver(FULL_WINDOW), measurement()))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.deepEqual(seen.map((detection) => detection.rule), ['request-too-large'])
})

test('one window resolution serves both rules', async () => {
  // The saturation rule's gates pass (a count arrived) but its ratio does not,
  // so the full-request rule takes over. Both ask the same question about the
  // same route, and the answer is paid for once.
  const resolve = windowResolver(FULL_WINDOW)
  const measure = measurement()
  const chunks = [
    { type: 'usage', usage: { inputTokens: 1_000, outputTokens: 0 } },
    errorFinish({ code: 'PI_AI_ERROR', message: 'boom' }),
  ]
  const out = await drain(guardOverflowStream(from(chunks), cfg(), ROUTE, undefined, undefined, resolve, measure))
  assert.equal(out.at(-1).reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  assert.deepEqual(resolve.calls, [`${ROUTE.provider}/${ROUTE.model}`])
  assert.equal(measure.calls.length, 1)
})

test('the full-request rule carries the reported turn through the real waterfall', async () => {
  // The real `llm/stream` waterfall, with the measurement and the window
  // supplied by the same two delegates the plugin wires. What this arm adds to
  // the generator tests is that nothing between the guard and the caller (the
  // runtime's own finish assembly, the invariant checks) rejects the shape.
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin({ name, apply, inject }, {})
  ctx.llm.registerAdapter(['openrouter'], new WindowedAdapter([ZERO_USAGE, REPORTED_400], FULL_WINDOW))
  const chunks = []
  for await (const chunk of ctx.llm.stream({
    provider: 'openrouter',
    model: ROUTE.model,
    messages: [],
    // No session was mounted in this context, so the real `measureRequestTokens`
    // finds no store and fails closed — the control arm below.
    sessionId: 'session-absent',
  })) chunks.push(chunk)
  assert.equal(chunks.at(-1).reason.failure.code, 'INVALID_REQUEST')
})

test('warn mode reports the full-request evidence without changing the stream', async () => {
  const chunks = [ZERO_USAGE, REPORTED_400]
  const seen = []
  const out = await drain(guardOverflowStream(from(chunks), WARN, ROUTE, (detection) => seen.push(detection), undefined, windowResolver(FULL_WINDOW), measurement()))
  assert.deepEqual(out, chunks)
  assert.equal(seen.length, 1)
  assert.equal(seen[0].rule, 'full-request-refused')
  assert.equal(seen[0].measuredTokens, REPORTED_PROMPT_TOKENS)
  assert.equal(seen[0].contextWindow, FULL_WINDOW)
  assert.equal(seen[0].threshold, 500_000)
  assert.equal(seen[0].code, 'INVALID_REQUEST')
})

test('off mode never asks for a measurement', async () => {
  const measure = measurement()
  const chunks = [ZERO_USAGE, REPORTED_400]
  const out = await drain(guardOverflowStream(from(chunks), OFF, ROUTE, undefined, undefined, windowResolver(FULL_WINDOW), measure))
  assert.deepEqual(out, chunks)
  assert.deepEqual(measure.calls, [])
})
