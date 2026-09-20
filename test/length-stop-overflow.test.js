/**
 * Length-stop overflow guard — behaviour tests.
 *
 * The unit half drives the pure generator; the integration half drives the real
 * `@deepseek-ai/cordis` context and the real `LlmRuntime` from
 * `@deepseek-ai/dsh-llm`, so the assertion is about the shipped `llm/stream`
 * waterfall rather than about this module's own idea of it.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { CONTEXT_WINDOW_EXCEEDED_CODE, LlmAdapter } from '@deepseek-ai/dsh-llm'
import * as LlmInvariant from '@deepseek-ai/dsh-llm/invariant'
import InvariantRegistry from '@deepseek-ai/dsh-invariants'
import {
  apply,
  classifyLengthStop,
  Config,
  DEFAULT_AT_MOST_OUTPUT_TOKENS,
  DEFAULT_MODE,
  guardLengthStopStream,
  isMaxTokensFinish,
  name,
  overflowFailure,
  promptTokensOf,
} from '../lib/index.js'

/** Resolved-config shape the wiring always produces. */
function cfg(over = {}) {
  return { mode: 'error', atMostOutputTokens: DEFAULT_AT_MOST_OUTPUT_TOKENS, ...over }
}
const WARN = cfg({ mode: 'warn' })
const OFF = cfg({ mode: 'off' })
const ROUTE = { provider: 'openrouter', model: 'deepseek/deepseek-v4.1-flash' }

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
  assert.deepEqual(evidence, { outputTokens: 1, promptTokens: 287115 })
})

test('prompt tokens are absent from the verdict', () => {
  // Deliberately: any rule comparing the prompt against the resolved window
  // inherits that window's error, and the reported session resolved it four
  // times too small. The clamp floor is the window-independent signal.
  const evidence = classifyLengthStop({ kind: 'max-tokens' }, { inputTokens: 3, outputTokens: 1 }, cfg())
  assert.deepEqual(evidence, { outputTokens: 1, promptTokens: 3 })
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
  assert.deepEqual(seen, [{ outputTokens: 1, promptTokens: 287115 }])
})

test('error mode reports the evidence it reclassified', async () => {
  const seen = []
  await drain(guardLengthStopStream(from(REPORTED_TURN), cfg(), ROUTE, (e) => seen.push(e)))
  assert.deepEqual(seen, [{ outputTokens: 1, promptTokens: 287115 }])
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
})

test('Config preserves explicit values', () => {
  const resolved = Config({ mode: 'warn', atMostOutputTokens: 64 })
  assert.equal(resolved.mode, 'warn')
  assert.equal(resolved.atMostOutputTokens, 64)
})

test('Config rejects an unknown mode', () => {
  assert.throws(() => Config({ mode: 'silent' }))
})

test('Config rejects a non-integer or negative threshold', () => {
  assert.throws(() => Config({ atMostOutputTokens: 1.5 }))
  assert.throws(() => Config({ atMostOutputTokens: -1 }))
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
 * Build the real service stack: a Cordis context, the real `LlmRuntime`, the
 * real `llm/stream` waterfall, and a scripted adapter registered on a route.
 * @param script - the chunks the adapter replays.
 * @param config - plugin config; the plugin is mounted when present.
 * @returns the context.
 */
async function harness(script, config) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  if (config !== undefined) await ctx.plugin({ name, apply }, config)
  ctx.llm.registerAdapter(['openrouter'], new ScriptedAdapter(script))
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
  await ctx.plugin({ name, apply }, {})
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
  await ctx.plugin({ name, apply }, {})
  ctx.llm.registerAdapter(['openrouter'], new ScriptedAdapter([
    { type: 'text-delta', index: 0, text: 'x' },
    maxTokensFinish(),
  ]))
  await assert.rejects(() => streamOnce(ctx), /requires an open text block/)
})
