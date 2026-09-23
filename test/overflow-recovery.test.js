/**
 * End-to-end: does the rewritten finish reach the harness' own overflow
 * recovery, or only the plugin's idea of it?
 *
 * The behaviour suite asserts the shape of the rewritten chunk. This one runs
 * the **real** downstream: the real `LlmRuntime` for hop 1, and the real
 * `@deepseek-ai/dsh-compaction-basic` — mounted with the services it injects —
 * for hop 2, dispatched through `agent/request-error` exactly as
 * `agent-loop/src/agent.ts:445-464` dispatches it.
 *
 * Both hops carry a control arm, so a green run is evidence rather than an
 * accident:
 *
 *   * hop 1, unmounted: the same scripted turn is an ordinary `max-tokens` end —
 *     which is the defect the plugin exists to fix.
 *   * hop 2, a foreign failure code: nothing is compacted and no retry is
 *     requested. Without this arm, "the gate passed" could mean "there is no
 *     gate".
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { CONTEXT_WINDOW_EXCEEDED_CODE, LlmAdapter } from '@deepseek-ai/dsh-llm'
import SessionStore from '@deepseek-ai/dsh-session'
import SessionProjection from '@deepseek-ai/dsh-session-projection'
import TokenMeter from '@deepseek-ai/dsh-token-meter'
import BasicCompactionEngine from '@deepseek-ai/dsh-compaction-basic'

import * as plugin from '../lib/index.js'

const ROUTE = { provider: 'openrouter', model: 'deepseek/deepseek-v4.1-flash' }

/** The reported turn, verbatim in shape (discussion #7214). */
const REPORTED_TURN = [
  { type: 'usage', usage: { inputTokens: 11, outputTokens: 1, totalTokens: 287116, cacheReadTokens: 287104 } },
  { type: 'finish', reason: { kind: 'max-tokens' } },
]

/** The refusal the #7626 reporter's session answered every turn with, log line verbatim. */
const REPORTED_413 = {
  type: 'finish',
  reason: {
    kind: 'error',
    failure: { code: 'INVALID_REQUEST', message: 'DeepSeek Messages request failed (413)', status: 413 },
  },
}

class ScriptedAdapter extends LlmAdapter {
  constructor(chunks) {
    super()
    this.chunks = chunks
  }

  async *stream() {
    for (const chunk of this.chunks) yield chunk
  }
}

async function drain(stream) {
  const out = []
  for await (const chunk of stream) out.push(chunk)
  return out
}

/** Mount the harness pieces the plugin rides on; `withGuard = false` is a control arm. */
async function harness(withGuard, chunks = REPORTED_TURN) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  if (withGuard) await ctx.plugin({ name: plugin.name, apply: plugin.apply }, {})
  ctx.llm.registerAdapter([ROUTE.provider], new ScriptedAdapter(chunks))
  return ctx
}

const streamOnce = (ctx) => ctx.llm.stream({
  provider: ROUTE.provider,
  model: ROUTE.model,
  messages: [{ role: 'user', content: [{ type: 'text', text: 'continue' }] }],
  signal: new AbortController().signal,
})

/* ------------------------------------------------------------------ *
 * hop 1 — the real runtime hands the caller the overflow classification
 * ------------------------------------------------------------------ */

test('through the real LlmRuntime the caller sees CONTEXT_WINDOW_EXCEEDED', async () => {
  const finish = (await drain(streamOnce(await harness(true)))).at(-1)
  assert.equal(finish.reason.kind, 'error')
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
})

test('control: unmounted, the same turn is an ordinary max-tokens end', async () => {
  const finish = (await drain(streamOnce(await harness(false)))).at(-1)
  assert.deepEqual(finish.reason, { kind: 'max-tokens' })
})

/* ------------------------------------------------------------------ *
 * hop 2 — the real compaction backend accepts that failure
 * ------------------------------------------------------------------ */

/**
 * `compactIfNeeded` is the documented subclass hook, so overriding it keeps the
 * code gate and the retry bookkeeping real while removing the summarizer
 * round-trip. A durable compaction advances `surface.replaceGeneration`, and the
 * handler treats no progress as "no retry", so the override must do the same.
 */
class ProbeEngine extends BasicCompactionEngine {
  calls = []

  async compactIfNeeded(agent, trigger, signal) {
    this.calls.push({ trigger, aborted: signal?.aborted === true })
    agent.session.surface.replaceGeneration += 1
    return null
  }
}

/** A session stub exposing only what `agent/request-error` reads before `compactIfNeeded`. */
function sessionStub() {
  return { surface: { replaceGeneration: 0 }, requestHeader: () => ({ config: ROUTE }) }
}

/**
 * Mount the real recovery path and return the pieces a test needs to drive it.
 *
 * `dsh-session-projection` and `dsh-token-meter` are here because
 * `BasicCompactionEngine.inject` names `tokenMeter`, whose own `inject` names
 * `sessionProjections` — the real mount order of `bundle/base`, reduced to what
 * this handler touches.
 */
async function recoveryHarness() {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(SessionProjection)
  await ctx.plugin(TokenMeter)
  await ctx.plugin(SessionStore)
  await ctx.plugin({ name: plugin.name, apply: plugin.apply }, {})
  await ctx.plugin(ProbeEngine, { auto: true })
  const engine = ctx.get('compaction')
  assert.ok(engine instanceof ProbeEngine, 'the real compaction backend must be mounted')

  const agent = { session: sessionStub(), options: { provider: ROUTE.provider, model: ROUTE.model } }
  const signal = new AbortController().signal
  const dispatch = (failure) => ctx.waterfall(
    'agent/request-error',
    { agent, turn: 23, step: 0, provider: ROUTE.provider, failure, retryPolicy: undefined, signal },
    () => Promise.resolve(undefined),
  )
  return { ctx, engine, dispatch, agent }
}

test('the plugin\u2019s failure passes the real code gate and buys a retry', async () => {
  const { engine, dispatch, agent } = await recoveryHarness()
  // The failure object is the plugin's own, produced by the real waterfall.
  const finish = (await drain(streamOnce(await harness(true)))).at(-1)
  const action = await dispatch(finish.reason.failure)
  assert.equal(engine.calls.length, 1)
  assert.equal(engine.calls[0].trigger, 'context-overflow')
  assert.equal(action.kind, 'retry')
  assert.equal(agent.session.surface.replaceGeneration, 1)
})

test('control: a foreign failure code never reaches compaction', async () => {
  const { engine, dispatch } = await recoveryHarness()
  const action = await dispatch({ code: 'INVALID_REQUEST', message: 'input tokens exceeded max_prompt_tokens' })
  assert.equal(engine.calls.length, 0)
  assert.equal(action, undefined)
})

test('the recovery stays bounded by maxOverflowRetries', async () => {
  const { engine, dispatch } = await recoveryHarness()
  const finish = (await drain(streamOnce(await harness(true)))).at(-1)
  assert.equal((await dispatch(finish.reason.failure)).kind, 'retry')
  // Default maxOverflowRetries is 1, so the second overflow failure is terminal
  // rather than a retry loop.
  assert.equal(await dispatch(finish.reason.failure), undefined)
  assert.equal(engine.calls.length, 1)
})

/* ------------------------------------------------------------------ *
 * hop 3 — the same recovery, reached by a second route
 * ------------------------------------------------------------------ */

test('a size refusal reaches the recovery a truncation reaches', async () => {
  // The classification is only worth anything if the built-in gate accepts it:
  // this drives the real LlmRuntime and then the real compaction engine.
  const finish = (await drain(streamOnce(await harness(true, [REPORTED_413])))).at(-1)
  assert.equal(finish.reason.failure.code, CONTEXT_WINDOW_EXCEEDED_CODE)
  const { engine, dispatch, agent } = await recoveryHarness()
  const action = await dispatch(finish.reason.failure)
  assert.equal(engine.calls.length, 1)
  assert.equal(engine.calls[0].trigger, 'context-overflow')
  assert.equal(action.kind, 'retry')
  assert.equal(agent.session.surface.replaceGeneration, 1)
})

test('control: unmounted, the same refusal is the opaque INVALID_REQUEST the report describes', async () => {
  // Both arms of #7626 in one place: without the plugin the code is one the
  // recovery does not read, so nothing is compacted and no retry is requested —
  // which is why the reported session could not send anything, ever again.
  const finish = (await drain(streamOnce(await harness(false, [REPORTED_413])))).at(-1)
  assert.equal(finish.reason.failure.code, 'INVALID_REQUEST')
  const { engine, dispatch } = await recoveryHarness()
  assert.equal(await dispatch(finish.reason.failure), undefined)
  assert.equal(engine.calls.length, 0)
})
