/**
 * Length-stop overflow guard for the dsh harness.
 *
 * Two ways a provider can reject an over-long request without ever saying
 * "context window exceeded". Both leave the session unable to make progress,
 * and both are repaired here by restoring the one classification the built-in
 * recovery keys on.
 *
 * ## Trigger 1 — a truncation reported as an ordinary output cap (#7214)
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
 * ## Trigger 2 — a request refused for its size (#7626)
 *
 * A session can also grow past the provider's **transport** limit, in bytes,
 * while every budget the harness computes still reports headroom, because the
 * harness prices a request in tokens and never in bytes. The only byte budget
 * in the whole LLM layer prices images (`LlmImageRequestBudget.maxBytes`); a
 * text request has no byte bound anywhere.
 *
 * The provider then refuses the request outright — HTTP 413 — and the
 * classification order turns that into the wrong code
 * (`llm-deepseek/src/transport.ts:31-33`):
 *
 * ```ts
 * else if (isContextWindowExceededError(detail)) code = 'CONTEXT_WINDOW_EXCEEDED'
 * else if (status === 400 || status === 413 || type === 'invalid_request_error') code = 'INVALID_REQUEST'
 * ```
 *
 * A 413 with no JSON body carries no wording for `isContextWindowExceededError`
 * to read (`llm/src/error.ts:54-85`), so the fallback string
 * (`DeepSeek Messages request failed (413)`, `transport.ts:25`) reaches the
 * `INVALID_REQUEST` branch. `CONTEXT_WINDOW_EXCEEDED` is what the recovery path
 * keys on, so the recovery — the only thing that can shrink what is sent —
 * never runs, and the reporter's session answered 413 to every turn, including
 * the summarization call that was supposed to reduce it. Forking inherits the
 * corpus, so a fork fails from its first turn too.
 *
 * This trigger reads size evidence instead: a request-rejection code with size
 * wording, or the literal status. `INVALID_REQUEST` covers malformed, oversized
 * and forbidden requests alike, so the code alone is never sufficient — the
 * wording has to name size, and the wording has to name *the request* (not, say,
 * an image) — see {@link isRequestTooLargeFailure}.
 *
 * ## What it does
 *
 * Observes the public `llm/stream` waterfall and rewrites the terminal frame
 * into the `CONTEXT_WINDOW_EXCEEDED` error the harness already knows how to
 * recover from. `dsh-compaction-basic` (mounted in `bundle/base` and
 * `bundle/web-app`) then compacts and returns `{ kind: 'retry' }`, bounded by
 * its own `maxOverflowRetries`.
 *
 * Three properties make the rewrite exact rather than approximate:
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
 * - One terminal frame in, one terminal frame out: the upstream failure is
 *   **replaced**, never followed by a second verdict, and this plugin never
 *   re-dispatches a request itself. The retry decision stays with the loop and
 *   its `maxOverflowRetries` bound.
 *
 * ## Deliberate narrowness
 *
 * The length-stop rule fires only when a `max-tokens` finish arrives with a
 * negligible output count. A real output cap cannot look like this: the budget
 * it hits is the model's own (tens of thousands of tokens in the catalog), so a
 * `length` stop after one or two tokens is a truncation of *our* request,
 * whoever truncated it — pi-ai's clamp, or a backend that serves less than it
 * advertises. `MIN_MAX_TOKENS = 1` is why the count is one and not zero.
 *
 * A finish with no `usage` chunk is left alone (fail-closed): without the
 * provider's own count there is nothing to judge. That rule applies to the
 * length-stop rule only — a request rejection carries no usage chunk at all.
 *
 * The size rule deliberately does not touch `aborted` finishes (the caller
 * stopped us), failures whose code another recovery already owns
 * (`CONTEXT_WINDOW_EXCEEDED` itself, `IMAGE_OFFLOAD_REQUIRED`), or size
 * complaints aimed at something other than the request.
 *
 * @module @argszero/cordis-plugin-length-stop-overflow
 */

import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
// Type-only: loads the @deepseek-ai/dsh-llm declaration merging that adds the
// `llm/stream` event to Cordis' Context.Events, and provides the FinishReason /
// GenerateOptions / StreamChunk / TokenUsage types.
import type {
  FinishReason,
  GenerateOptions,
  LlmFailure,
  ProviderRequestId,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm'
// Value imports: the harness' own codes, so this plugin and the core can never
// disagree about which failure the recovery path keys on, nor about which code
// already owns a recovery of its own.
import { CONTEXT_WINDOW_EXCEEDED_CODE, IMAGE_OFFLOAD_REQUIRED_CODE } from '@deepseek-ai/dsh-llm'

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

/**
 * Whether a request rejected for its size is reclassified too (default).
 *
 * On by default because the alternative is a session that cannot send anything
 * and reports a code no recovery reads. Turn it off to keep the plugin exactly
 * as it was before 0.2.0.
 */
export const DEFAULT_CLASSIFY_OVERSIZE_REQUESTS = true

/** Plugin configuration. */
export interface Config {
  /**
   * What to do when a `max-tokens` finish carries a negligible output count, or
   * a request is refused for its size. One of `'error'` (default), `'warn'`, or
   * `'off'`.
   *
   * `'error'` replaces the terminal frame with a `CONTEXT_WINDOW_EXCEEDED`
   * failure, so the loop routes it to `agent/request-error`
   * (`agent-loop/src/agent.ts:444-464`), where `dsh-compaction-basic` compacts
   * and returns `{ kind: 'retry' }` — bounded by its `maxOverflowRetries`
   * (default 1), so there is no retry loop. `'warn'` reports the detection
   * without changing behaviour, which is how an operator validates the
   * thresholds against their own traffic before enabling them. `'off'` makes
   * the plugin a pure pass-through.
   */
  mode?: 'error' | 'warn' | 'off'
  /** Output count at or below which a `max-tokens` finish is a truncation. Default 2. */
  atMostOutputTokens?: number
  /**
   * Whether a request refused for its size (`413`, or size wording on a
   * request-rejection code) is reclassified as `CONTEXT_WINDOW_EXCEEDED`.
   * Default `true`.
   */
  classifyOversizeRequests?: boolean
}

/** Resolved config: every field carries its validated default. */
export type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  mode: z.union(['error', 'warn', 'off']).default(DEFAULT_MODE),
  atMostOutputTokens: z.natural().default(DEFAULT_AT_MOST_OUTPUT_TOKENS),
  classifyOversizeRequests: z.boolean().default(DEFAULT_CLASSIFY_OVERSIZE_REQUESTS),
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
  /** Discriminant: this detection came from a negligible-output `max-tokens` finish. */
  rule: 'length-stop'
  /** Output tokens the provider reported for the truncated answer. */
  outputTokens: number
  /** Prompt tokens the provider reported for the same request. */
  promptTokens: number
}

/** What a size-rejected request looked like, for the log line and the failure text. */
export interface OversizeEvidence {
  /** Discriminant: this detection came from a request the provider refused for its size. */
  rule: 'request-too-large'
  /** The provider's own message, verbatim — the only durable trace of the refusal. */
  providerMessage: string
  /** Which evidence classified it: the status itself, or the wording. */
  matched: 'status' | 'wording'
  /** HTTP status the adapter reported, when it reported one. */
  status?: number
  /** Request identifier the adapter reported, when it reported one. */
  requestId?: ProviderRequestId
  /**
   * Estimated bytes of the request's message array, measured only when a
   * detection fires. See {@link estimateRequestBytes} for what it is and is not.
   */
  requestBytes?: number
}

/** One reclassification this guard performed (or, in `warn` mode, would have performed). */
export type OverflowDetection = TruncationEvidence | OversizeEvidence

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
  return { rule: 'length-stop', outputTokens, promptTokens: promptTokensOf(usage) }
}

/**
 * Adapter codes that mean "the provider refused the request itself".
 *
 * A rejection code is **necessary but never sufficient**: `INVALID_REQUEST`
 * covers malformed, oversized and forbidden requests alike (`llm-deepseek` maps
 * 400/413/`invalid_request_error` onto it wholesale), so the wording still has
 * to name size. `HTTP_400`/`HTTP_413` are the fall-through codes an adapter
 * produces when it does not map the status at all.
 */
const REQUEST_REJECTION_CODES: ReadonlySet<string> = new Set(['INVALID_REQUEST', 'HTTP_400', 'HTTP_413'])

/**
 * Codes whose recovery is already owned by someone else.
 *
 * `CONTEXT_WINDOW_EXCEEDED` is the outcome this plugin wants; `IMAGE_OFFLOAD_REQUIRED`
 * is the image path (`dsh-compaction-image-offload` offloads the named count and
 * retries). A size complaint that reached either code is not a misclassification,
 * and rewriting it would take the recovery away from the thing that can fix it.
 */
const RECOVERY_OWNED_CODES: ReadonlySet<string> = new Set([
  CONTEXT_WINDOW_EXCEEDED_CODE,
  IMAGE_OFFLOAD_REQUIRED_CODE,
])

/**
 * Nouns that name what we send.
 *
 * A size complaint that mentions none of these is about something inside the
 * payload — an image, an attachment — and belongs to the recovery that owns
 * that thing, not to compaction.
 */
const REQUEST_BEARER = /\b(?:request|payload|body|content|prompt|input|messages?|conversation|transcript)\b/iu

/**
 * Phrases that state a size bound was hit.
 *
 * Three families, all observed in the wild: `<bearer> … too large|long|big`
 * (`413 Request Entity Too Large`, a gateway's plain `payload too large`,
 * Anthropic's `prompt is too long: 210000 tokens > 200000 maximum`); a size noun
 * with a bound verb (`request size exceeded`, `length limit exceeded`); and
 * `<bearer> exceeds the maximum|allowed|permitted …` (vLLM's *"The request
 * exceeds the maximum size allowed by the model"*).
 */
const SIZE_PHRASE = /\btoo[\s_-]+(?:large|long|big)\b|\b(?:size|length|bytes?)\b[\s_-]*(?:limit[\s_-]+)?(?:exceed\w*|over\w*|too\b)|\bexceed\w*[\s_-]+(?:the[\s_-]+)?(?:maximum|allowed|permitted)\b/iu

/** The status, spelled out in text. An adapter's body-less fallback is the common case. */
const STATUS_TOKEN = /\b413\b/u

/**
 * Whether one failure reads as "the request was refused because of its size".
 *
 * Deliberately **not** a code check: `INVALID_REQUEST` is a bag that holds
 * malformed, forbidden and oversized requests together, so the reading has to
 * come from the evidence. Accepted evidence, in order of strength:
 *
 * - the adapter reported status 413 — `Content Too Large` by definition, so no
 *   reading is needed;
 * - the failure text spells `413` — this is how a body-less refusal survives
 *   (`DeepSeek Messages request failed (413)`) and how a proxy in front of the
 *   provider reports it (`413 Request Entity Too Large`);
 * - a request-rejection code **and** a bearer **and** a size phrase.
 *
 * @param failure - the terminal failure the adapter produced.
 * @returns `true` when the failure should be read as a request-size refusal.
 */
export function isRequestTooLargeFailure(failure: LlmFailure): boolean {
  if (failure.status === 413) return true
  if (STATUS_TOKEN.test(failure.message)) return true
  if (!REQUEST_REJECTION_CODES.has(failure.code)) return false
  return REQUEST_BEARER.test(failure.message) && SIZE_PHRASE.test(failure.message)
}

/**
 * Whether one terminal finish is a request the provider refused for its size.
 *
 * @param reason - the stream's terminal finish reason.
 * @param config - resolved plugin config.
 * @returns the evidence when the finish should be reclassified, else `null`.
 */
export function classifyOversizeFailure(reason: FinishReason, config: ResolvedConfig): OversizeEvidence | null {
  if (!config.classifyOversizeRequests) return null
  // `aborted` means the caller stopped us — never rewrite the caller's own
  // decision into a provider verdict. `max-tokens` is the other trigger's.
  if (reason.kind !== 'error') return null
  const { failure } = reason
  if (RECOVERY_OWNED_CODES.has(failure.code)) return null
  if (!isRequestTooLargeFailure(failure)) return null
  return {
    rule: 'request-too-large',
    providerMessage: failure.message,
    matched: failure.status === 413 ? 'status' : 'wording',
    ...failure.status === undefined ? {} : { status: failure.status },
    ...failure.requestId === undefined ? {} : { requestId: failure.requestId },
  }
}

/**
 * Classify one terminal finish against both rules.
 *
 * The cheap discriminant first: a `max-tokens` finish is a truncation question,
 * an `error` finish is a refusal question, and everything else passes through.
 *
 * @param reason - the stream's terminal finish reason.
 * @param usage - the last `usage` chunk seen, or `undefined` when none arrived.
 * @param config - resolved plugin config.
 * @returns the detection, or `null` when the finish must pass through unchanged.
 */
export function classifyFinish(
  reason: FinishReason,
  usage: TokenUsage | undefined,
  config: ResolvedConfig,
): OverflowDetection | null {
  return classifyLengthStop(reason, usage, config) ?? classifyOversizeFailure(reason, config)
}

/**
 * Estimated UTF-8 bytes of one request's message array, as this seam sees it.
 *
 * The harness prices a request in **tokens** and never in bytes — the only byte
 * budget in the LLM layer prices images — so a session can grow past a
 * provider's transport limit while every in-tree check still reports headroom
 * (discussion #7626). This is the missing number, measured where a plugin can
 * measure it, and attached to the failure text so the refusal that ends a turn
 * carries a size instead of only a status.
 *
 * Read it as **"how much JSON this request carried"**, never as "what the
 * provider counted": adapters re-serialize into their own protocol, and
 * `dsh-attachment` may re-encode request images, so the wire payload is the
 * adapter's and is not observable from here. It is an estimate of one
 * component, and it is labelled as one everywhere it is reported.
 *
 * Called only when a detection fires, so a healthy request pays nothing for it.
 *
 * @param options - the request as the waterfall received it.
 * @returns the byte count, or `undefined` when the request cannot be measured.
 */
export function estimateRequestBytes(options: GenerateOptions): number | undefined {
  try {
    const { messages } = options
    if (!Array.isArray(messages)) return undefined
    return Buffer.byteLength(JSON.stringify(messages), 'utf8')
  } catch {
    // An unserializable payload is the caller's problem; the turn already has a
    // failure to report, and this plugin must not replace it with a different one.
    return undefined
  }
}

/**
 * Render a byte count for a log line or a failure message.
 *
 * @param bytes - a non-negative byte count.
 * @returns a human-scale rendering that keeps the exact count.
 */
export function formatBytes(bytes: number): string {
  const megabytes = bytes / (1024 * 1024)
  return `${megabytes >= 10 ? megabytes.toFixed(0) : megabytes.toFixed(1)} MB (${bytes.toLocaleString('en-US')} bytes)`
}

/**
 * The `CONTEXT_WINDOW_EXCEEDED` failure that replaces a truncating stop.
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
 * The `CONTEXT_WINDOW_EXCEEDED` failure that replaces a size refusal.
 *
 * Three facts are carried over deliberately:
 *
 * - the provider's own message, verbatim, because a body-less refusal
 *   (`DeepSeek Messages request failed (413)`) is the only record of what was
 *   said, and the operator's next question is which limit was hit;
 * - the status and the request id, because they are what a provider-side log
 *   search keys on;
 * - the measured size, when the request could be measured.
 *
 * `providerRetryAfterMs` is dropped: waiting does not make an oversized request
 * smaller, and this code is not retryable anyway.
 *
 * @param evidence - what the refused request looked like.
 * @param route - the provider/model the request went to.
 * @returns the failure the loop routes to `agent/request-error`.
 */
export function oversizeFailure(evidence: OversizeEvidence, route: { provider: string; model: string }): LlmFailure {
  const measured = evidence.requestBytes === undefined
    ? ''
    : `; the request carried ${formatBytes(evidence.requestBytes)} of message JSON (an estimate of this component — the wire form belongs to the adapter)`
  return {
    code: CONTEXT_WINDOW_EXCEEDED_CODE,
    message: `${route.provider}/${route.model} refused the request as too large`
      + `${evidence.status === undefined ? '' : ` (HTTP ${evidence.status})`}: ${evidence.providerMessage}${measured}.`
      + ` Reclassified as ${CONTEXT_WINDOW_EXCEEDED_CODE} by length-stop-overflow (discussion #7626): the harness`
      + ' prices a request in tokens and never in bytes, so context-overflow compaction is the only built-in'
      + ' recovery that can shrink what is sent.',
    ...evidence.status === undefined ? {} : { status: evidence.status },
    ...evidence.requestId === undefined ? {} : { requestId: evidence.requestId },
  }
}

/**
 * Replace one terminal finish with the overflow failure the recovery path reads.
 *
 * Replay metadata is dropped on both paths: it describes a successful response
 * (`llm/src/types.ts`), and neither rewrite describes one — the loop settles the
 * attempt and throws, so keeping the envelope would only mislead a reader of the
 * durable log.
 *
 * @param chunk - the terminal finish chunk as the adapter emitted it.
 * @param detection - the classification that justifies the rewrite.
 * @param route - the provider/model the request went to.
 * @returns a finish carrying the synthesized failure and nothing else.
 */
export function rewriteFinish(
  chunk: Extract<StreamChunk, { type: 'finish' }>,
  detection: OverflowDetection,
  route: { provider: string; model: string },
): StreamChunk {
  const { replayState: _discarded, ...rest } = chunk
  const failure = detection.rule === 'length-stop'
    ? overflowFailure(detection, route)
    : oversizeFailure(detection, route)
  return { ...rest, reason: { kind: 'error', failure } }
}

/**
 * The guard: forward every chunk the moment it arrives, and reclassify a
 * terminal frame that is an overflow the harness could not name.
 *
 * **Streaming-preserving by construction.** Every verdict needs only the
 * terminal chunk (plus the `usage` chunk that precedes a length stop), so
 * nothing is buffered: every chunk is yielded as it arrives and at most the last
 * one is rewritten.
 *
 * @param source - the upstream chunk stream.
 * @param config - resolved plugin config.
 * @param route - the provider/model the request went to, for the failure text.
 * @param onDetect - called once when a finish is reclassified. Kept as a
 *   parameter so the generator stays pure and log-free for tests; the plugin
 *   passes a `ctx.logger.warn` delegate.
 * @param requestBytes - measures the request on demand. Called only when the
 *   size rule detects, so a healthy request is never serialized.
 * @returns the stream, with an unnameable overflow reported as
 *   `CONTEXT_WINDOW_EXCEEDED`.
 */
export async function* guardOverflowStream(
  source: AsyncIterable<StreamChunk>,
  config: ResolvedConfig,
  route: { provider: string; model: string },
  onDetect?: (detection: OverflowDetection) => void,
  requestBytes?: () => number | undefined,
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
      const detection = classifyFinish(chunk.reason, usage, config)
      if (detection !== null) {
        const measured = detection.rule === 'request-too-large' ? requestBytes?.() : undefined
        const evidence: OverflowDetection = detection.rule === 'request-too-large' && measured !== undefined
          ? { ...detection, requestBytes: measured }
          : detection
        onDetect?.(evidence)
        if (config.mode === 'error') {
          yield rewriteFinish(chunk, evidence, route)
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
 * The v0.1.0 name of {@link guardOverflowStream}.
 *
 * Kept so a wire written against 0.1.0 keeps working; new code should use the
 * name that says what the guard now does.
 */
export const guardLengthStopStream = guardOverflowStream

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
    classifyOversizeRequests: config.classifyOversizeRequests ?? DEFAULT_CLASSIFY_OVERSIZE_REQUESTS,
  }

  ctx.on('llm/stream', (options: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => {
    const upstream = next()
    if (resolved.mode === 'off') return upstream
    return guardOverflowStream(
      upstream,
      resolved,
      { provider: options.provider, model: options.model },
      (detection) => {
        const verdict = resolved.mode === 'error'
          ? `reported as ${CONTEXT_WINDOW_EXCEEDED_CODE} so context-overflow compaction can reduce the request`
          : 'left unchanged (warn mode)'
        if (detection.rule === 'length-stop') {
          ctx.logger.warn(
            'length-stop-overflow: %s/%s stopped for length after %d output token(s) on a %d-token prompt; %s (discussion #7214)',
            options.provider,
            options.model,
            detection.outputTokens,
            detection.promptTokens,
            verdict,
          )
          return
        }
        ctx.logger.warn(
          'length-stop-overflow: %s/%s refused the request as too large (%s: %s)%s; %s (discussion #7626)',
          options.provider,
          options.model,
          detection.matched,
          detection.providerMessage,
          detection.requestBytes === undefined ? '' : ` carrying ${formatBytes(detection.requestBytes)} of message JSON`,
          verdict,
        )
      },
      () => estimateRequestBytes(options),
    )
  })
}
