/**
 * Length-stop overflow guard for the dsh harness.
 *
 * A provider that runs out of room answers with a **truncation**, not a
 * failure: `finish_reason: "length"` (pi-ai's `stopReason: 'length'`), which
 * `dsh-llm` maps to `{ kind: 'max-tokens' }`. The harness reads that as "the
 * model reached the output budget I asked for" — an ordinary turn end.
 *
 * It is not always that. The reported case (discussion #7214) is a session that
 * ended a turn with **one** output token, over and over, forever:
 *
 * 1. `llm-pi-ai` hands pi-ai's *simple* entry point the resolved model
 *    (`adapter.ts:380` → `streamSimple`), and pi-ai clamps the output budget
 *    against that model's context window
 *    (`@earendil-works/pi-ai/dist/api/simple-options.js:2-9`):
 *
 *    ```js
 *    const CONTEXT_SAFETY_TOKENS = 4096
 *    const MIN_MAX_TOKENS = 1
 *    const available = model.contextWindow - estimateContextTokens(context).tokens - CONTEXT_SAFETY_TOKENS
 *    return Math.min(maxTokens, Math.max(MIN_MAX_TOKENS, available))
 *    ```
 *
 *    Once the conversation outgrows the window the harness *believes* in, the
 *    request goes out as `max_tokens: 1`. The provider is right to answer
 *    `length` after one token.
 * 2. pi-ai's own overflow sniff requires **exactly zero** output for its
 *    length-stop case (`isContextOverflow`, `dist/utils/overflow.js:149`), and
 *    `MIN_MAX_TOKENS = 1` is precisely what keeps the count off zero.
 * 3. `llm-pi-ai` falls through to `case 'length': return { kind: 'max-tokens' }`
 *    (`stream.ts:109`), so the loop settles the turn normally.
 * 4. `max-tokens` is a sticky terminal `TurnEndReason` with no recovery hook.
 *    The only overflow-recovery entry point is `agent/request-error` carrying
 *    `CONTEXT_WINDOW_EXCEEDED` (`compaction-basic/src/index.ts:180`), and
 *    nothing on this path produces it.
 *
 * Each degenerate answer is a valid prefix estimate for the next request (it is
 * neither `aborted` nor `error`), so `continue` reproduces the same negative
 * headroom: a full request, one token back, no way out.
 *
 * ## What it does
 *
 * Observes the public `llm/stream` waterfall and rewrites the terminal
 * `max-tokens` finish into the `CONTEXT_WINDOW_EXCEEDED` error the harness
 * already knows how to recover from. `dsh-compaction-basic` (mounted in
 * `bundle/base` and `bundle/web-app`) then compacts and returns
 * `{ kind: 'retry' }`, bounded by its own `maxOverflowRetries`.
 *
 * Two properties make the rewrite exact rather than approximate:
 *
 * - The synthesized failure carries the harness' own
 *   `CONTEXT_WINDOW_EXCEEDED_CODE`, so it is indistinguishable from an
 *   adapter's own verdict.
 * - On an `error` finish the loop settles only `assistant/attempt`
 *   (`agent-loop/src/agent.ts:444-448`) and never appends an
 *   `assistant/message`, so the degenerate one-token answer does **not** enter
 *   the derived conversation. The compaction then summarizes a history without
 *   it — which matters here, because the alternative is retrying forever with
 *   a useless answer appended each time.
 *
 * ## Deliberate narrowness
 *
 * The plugin fires only when a `max-tokens` finish arrives with a negligible
 * output count. A real output cap cannot look like this: the budget it hits is
 * the model's own (tens of thousands of tokens in the catalog), so a `length`
 * stop after one or two tokens is a truncation of *our* request, whoever
 * truncated it — pi-ai's clamp, or a backend that serves less than it
 * advertises. `MIN_MAX_TOKENS = 1` is why the count is one and not zero.
 *
 * A finish with no `usage` chunk is left alone (fail-closed): without the
 * provider's own count there is nothing to judge.
 *
 * @module @argszero/cordis-plugin-length-stop-overflow
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
// Type-only: loads the @deepseek-ai/dsh-llm declaration merging that adds the
// `llm/stream` event to Cordis' Context.Events, and provides the FinishReason /
// GenerateOptions / StreamChunk / TokenUsage types.
import type { FinishReason, GenerateOptions, LlmFailure, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
// Value import: the harness' own code, so this plugin and the core can never
// disagree about which failure the recovery path keys on.
import { CONTEXT_WINDOW_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'length-stop-overflow'

/** The LLM service this plugin wraps (`llm/stream`). */
export const inject = ['llm']

/**
 * Largest output count that still reads as a truncation rather than an output
 * cap.
 *
 * pi-ai's clamp collapses to its `MIN_MAX_TOKENS = 1` floor as soon as the
 * prompt reaches the believed window, so the reported (and unrecoverable) case
 * is exactly one token. Two rather than one costs nothing — no real output cap
 * is two tokens — and absorbs an off-by-one in a provider's own accounting.
 *
 * Raise it when the provider clamps the output budget to a small *positive*
 * remainder instead of the floor: with a believed window of 262,144 and the
 * clamp's 4,096-token reserve, a prompt anchored anywhere in
 * `[250_000, 258_047]` yields a budget in `[8_046, 1]`, all of it context
 * pressure. `atMostOutputTokens: 64` covers that band; the summary in
 * discussion #7212 is why a prompt can sit there at all.
 */
export const DEFAULT_AT_MOST_OUTPUT_TOKENS = 2

/** Default mode: restore the classification the recovery path keys on. */
export const DEFAULT_MODE = 'error'

/** Plugin configuration. */
export interface Config {
  /**
   * What to do when a `max-tokens` finish carries a negligible output count.
   * One of `'error'` (default), `'warn'`, or `'off'`.
   *
   * `'error'` replaces the finish with a `CONTEXT_WINDOW_EXCEEDED` failure, so
   * the loop routes it to `agent/request-error`
   * (`agent-loop/src/agent.ts:444-464`), where `dsh-compaction-basic` compacts
   * and returns `{ kind: 'retry' }` — bounded by its `maxOverflowRetries`
   * (default 1), so there is no retry loop. `'warn'` reports the detection
   * without changing behaviour, which is how an operator validates the
   * threshold against their own traffic before enabling it. `'off'` makes the
   * plugin a pure pass-through.
   */
  mode?: 'error' | 'warn' | 'off'
  /** Output count at or below which a `max-tokens` finish is a truncation. Default 2. */
  atMostOutputTokens?: number
}

/** Resolved config: every field carries its validated default. */
export type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  mode: z.union(['error', 'warn', 'off']).default(DEFAULT_MODE),
  atMostOutputTokens: z.natural().default(DEFAULT_AT_MOST_OUTPUT_TOKENS),
})

/**
 * The `max-tokens` member of the merge-extensible {@link FinishReason} union.
 *
 * Declared explicitly because `FinishReasonMap` is designed to widen as
 * adapters add provider-specific reasons, so a plain discriminant check does
 * not narrow the union for the compiler.
 */
export interface MaxTokensFinishReason {
  kind: 'max-tokens'
}

/**
 * Narrow a finish reason to its `max-tokens` member.
 *
 * `aborted` and `error` are excluded by construction: they already reach the
 * recovery path, and reclassifying them here would only misreport why.
 *
 * @param reason - any terminal finish reason.
 * @returns `true` when the reason is a `max-tokens` stop.
 */
export function isMaxTokensFinish(reason: FinishReason): reason is MaxTokensFinishReason {
  return reason.kind === 'max-tokens'
}

/**
 * Prompt-side tokens from the provider's own count.
 *
 * `TokenUsage` counts are disjoint: `inputTokens` is uncached input only and
 * cached input is reported separately (`llm/src/types.ts:154-176`), so the
 * prompt is their sum — the same quantity pi-ai calls
 * `usage.input + usage.cacheRead`.
 *
 * @param usage - the terminal usage chunk's counts.
 * @returns the prompt token count.
 */
export function promptTokensOf(usage: TokenUsage): number {
  return (usage.inputTokens ?? 0) + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0)
}

/** What a truncating `max-tokens` finish looked like, for the log line. */
export interface TruncationEvidence {
  /** Output tokens the provider reported for the truncated answer. */
  outputTokens: number
  /** Prompt tokens the provider reported for the same request. */
  promptTokens: number
}

/**
 * Whether one terminal finish is a truncation rather than an output cap.
 *
 * @param reason - the stream's terminal finish reason.
 * @param usage - the last `usage` chunk seen, or `undefined` when none arrived.
 * @param config - resolved plugin config.
 * @returns the evidence when the finish should be reclassified, else `null`.
 */
export function classifyLengthStop(
  reason: FinishReason,
  usage: TokenUsage | undefined,
  config: ResolvedConfig,
): TruncationEvidence | null {
  if (!isMaxTokensFinish(reason)) return null
  // No provider count, no verdict: guessing here would turn a legitimate
  // output cap into a compaction that cannot help.
  if (usage === undefined) return null
  const outputTokens = usage.outputTokens
  if (typeof outputTokens !== 'number' || !Number.isFinite(outputTokens) || outputTokens < 0) return null
  if (outputTokens > config.atMostOutputTokens) return null
  return { outputTokens, promptTokens: promptTokensOf(usage) }
}

/**
 * The `CONTEXT_WINDOW_EXCEEDED` failure that replaces the truncating stop.
 *
 * The message keeps the numbers the provider reported, because they are the
 * only durable trace of the request that was cut: `request/context` records the
 * resolved window but no output budget, so a clamped request leaves no other
 * evidence in the session log.
 *
 * @param evidence - what the truncated finish reported.
 * @param route - the provider/model the request went to.
 * @returns the failure the loop routes to `agent/request-error`.
 */
export function overflowFailure(evidence: TruncationEvidence, route: { provider: string; model: string }): LlmFailure {
  return {
    code: CONTEXT_WINDOW_EXCEEDED_CODE,
    message: `${route.provider}/${route.model} stopped for length after ${evidence.outputTokens} output token(s)`
      + ` on a ${evidence.promptTokens}-token prompt: the response was truncated, not capped`
      + ` (reclassified as ${CONTEXT_WINDOW_EXCEEDED_CODE} by length-stop-overflow, discussion #7214).`,
  }
}

/**
 * The guard: forward every chunk the moment it arrives, and reclassify a
 * truncating `max-tokens` finish.
 *
 * **Streaming-preserving by construction.** The verdict needs only the terminal
 * chunk plus the `usage` chunk that precedes it, so nothing is buffered: every
 * chunk is yielded as it arrives and at most the last one is rewritten.
 *
 * @param source - the upstream chunk stream.
 * @param config - resolved plugin config.
 * @param route - the provider/model the request went to, for the failure text.
 * @param onDetect - called once when a finish is reclassified. Kept as a
 *   parameter so the generator stays pure and log-free for tests; the plugin
 *   passes a `ctx.logger.warn` delegate.
 * @returns the stream, with a truncating stop reported as
 *   `CONTEXT_WINDOW_EXCEEDED`.
 */
export async function* guardLengthStopStream(
  source: AsyncIterable<StreamChunk>,
  config: ResolvedConfig,
  route: { provider: string; model: string },
  onDetect?: (evidence: TruncationEvidence) => void,
): AsyncIterable<StreamChunk> {
  if (config.mode === 'off') {
    for await (const chunk of source) yield chunk
    return
  }

  let usage: TokenUsage | undefined

  for await (const chunk of source) {
    if (chunk.type === 'usage') {
      usage = chunk.usage
      yield chunk
      continue
    }
    if (chunk.type === 'finish') {
      const evidence = classifyLengthStop(chunk.reason, usage, config)
      if (evidence !== null) {
        onDetect?.(evidence)
        if (config.mode === 'error') {
          // Replay metadata describes a successful response (`llm/src/types.ts`),
          // and this one is not replayed: the loop settles an attempt and
          // throws, so keeping the envelope would only mislead a reader.
          const { replayState: _discarded, ...rest } = chunk
          yield { ...rest, reason: { kind: 'error', failure: overflowFailure(evidence, route) } }
          continue
        }
      }
      yield chunk
      continue
    }
    yield chunk
  }
}

/**
 * Register the guard.
 *
 * @param ctx - the Cordis context.
 * @param config - plugin config (defaults applied here so a caller that builds
 *   its own config object — a custom profile layer, a test — gets the same
 *   behaviour as one that went through schemastery).
 */
export function apply(ctx: Context, config: Config = {}): void {
  const resolved: ResolvedConfig = {
    mode: config.mode ?? DEFAULT_MODE,
    atMostOutputTokens: config.atMostOutputTokens ?? DEFAULT_AT_MOST_OUTPUT_TOKENS,
  }

  ctx.on('llm/stream', (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => {
    const upstream = next()
    if (resolved.mode === 'off') return upstream
    return guardLengthStopStream(
      upstream,
      resolved,
      { provider: options.provider, model: options.model },
      (evidence) => {
        ctx.logger.warn(
          'length-stop-overflow: %s/%s stopped for length after %d output token(s) on a %d-token prompt; %s (discussion #7214)',
          options.provider,
          options.model,
          evidence.outputTokens,
          evidence.promptTokens,
          resolved.mode === 'error'
            ? `reported as ${CONTEXT_WINDOW_EXCEEDED_CODE} so context-overflow compaction can reduce the request`
            : 'left unchanged (warn mode)',
        )
      },
    )
  })
}
