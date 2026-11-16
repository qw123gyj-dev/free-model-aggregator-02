// The gateway: one Chat Completions call per model dialled, and what came back.
//
// Every free model is dialled on the gateway's Chat Completions route, never
// on Messages: `POST $OPENAI_BASE_URL/chat/completions` with the dialled id
// in `model` and the trial's dummy key as `Authorization: Bearer ...`; the
// gateway swaps it for the entrant's own OpenRouter key, which never enters
// the sandbox. A request names `max_tokens` (or `max_completion_tokens`):
// the gateway refuses a call with no output cap, or one over 8192, with 400.
//
// The upstream answer is read whole before anything is sent to the client:
// a stream that breaks halfway cannot be failed over once its first chunk
// has gone out, and the usage the provider sends in its last chunk is needed
// before the client's first event is written (the Anthropic wire's
// message_start carries the input count). Completions here are short — a
// token, or one JSON object — so buffering costs nothing and the env waits
// up to 30 seconds per answer.
//
// A refusal comes back as the provider sent it: its status and its own error
// body, no `retry-after` header. `errorOf` reads the provider's code out of
// it for the failover trail; where the body names none, the code is derived
// from the status, so the trail always carries something truthful.

import { isRecord, type Json } from './brief.ts';

/** The gateway's Chat Completions route for this trial's base URL. */
export function chatCompletionsUrl(baseUrl: string): string {
  const base = baseUrl.replace(/\/+$/, '');
  if (base === '') return '';
  return base.endsWith('/v1') ? `${base}/chat/completions` : `${base}/v1/chat/completions`;
}

/** One tool call the provider made, in OpenAI's shape. */
export interface UpstreamToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A provider answer, read whole: text, tool calls, stop reason, usage. */
export interface UpstreamCompletion {
  /** The text, exactly as the provider wrote it: never edited, never added to. */
  text: string;
  toolCalls: UpstreamToolCall[];
  /** OpenAI finish_reason: stop, length, tool_calls, function_call, content_filter. */
  finishReason: string;
  /** The provider's usage, or null when it sent none. */
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  /** The model the provider's own answer names (without the gateway prefix). */
  providerModel: string | null;
}

/** One gateway refusal, for the failover trail. */
export interface GatewayRefusal {
  status: number;
  code: string;
}

/** The provider's error code out of a refused body, or one derived from the status. */
export function errorCodeOf(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code;
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type;
    }
  } catch {
    // Not JSON: fall through to the status-derived code.
  }
  if (status === 400) return 'invalid_request_error';
  if (status === 401) return 'authentication_error';
  if (status === 403) return 'permission_error';
  if (status === 404) return 'not_found_error';
  if (status === 429) return 'rate_limited';
  if (status === 413) return 'request_too_large';
  if (status >= 500) return 'upstream_error';
  return 'request_error';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

function toolCallsOf(message: Json): UpstreamToolCall[] {
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const out: UpstreamToolCall[] = [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    out.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '',
    });
  }
  return out;
}

function usageOf(value: Json): { prompt_tokens: number; completion_tokens: number } | null {
  const usage = isRecord(value.usage) ? value.usage : null;
  if (usage === null) return null;
  const prompt = numberOrNull(usage.prompt_tokens);
  const completion = numberOrNull(usage.completion_tokens);
  if (prompt === null || completion === null) return null;
  return { prompt_tokens: prompt, completion_tokens: completion };
}

/** A Chat Completions JSON answer as an UpstreamCompletion. */
function completionOfJson(body: Json): UpstreamCompletion {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  return {
    text: textOf(message.content),
    toolCalls: toolCallsOf(message),
    finishReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    usage: usageOf(body),
    providerModel: typeof body.model === 'string' ? body.model : null,
  };
}

/** A Chat Completions SSE stream, read whole, as an UpstreamCompletion. */
async function completionOfStream(stream: ReadableStream<Uint8Array>): Promise<UpstreamCompletion> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let finishReason = 'stop';
  let usage: { prompt_tokens: number; completion_tokens: number } | null = null;
  let providerModel: string | null = null;
  const callsByIndex = new Map<number, UpstreamToolCall>();

  const frame = (payload: string): void => {
    if (payload === '[DONE]') return;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return;
    }
    if (!isRecord(chunk)) return;
    if (typeof chunk.model === 'string' && chunk.model !== '') providerModel = chunk.model;
    const chunkUsage = usageOf(chunk);
    if (chunkUsage !== null) usage = chunkUsage;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : callsByIndex.size;
      const fn = isRecord(call.function) ? call.function : {};
      const existing = callsByIndex.get(index) ?? { id: '', name: '', arguments: '' };
      if (typeof call.id === 'string' && call.id !== '') existing.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') existing.name = fn.name;
      if (typeof fn.arguments === 'string') existing.arguments += fn.arguments;
      callsByIndex.set(index, existing);
    }
    if (typeof choice.finish_reason === 'string') finishReason = choice.finish_reason;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const rawFrame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of rawFrame.split('\n')) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) frame(trimmed.slice(5).trim());
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    for (const line of buffer.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) frame(trimmed.slice(5).trim());
    }
  } finally {
    reader.releaseLock();
  }
  return {
    text,
    // Map iteration order is insertion order, and tool calls arrive in index
    // order, so this is the calls in the order the provider streamed them.
    toolCalls: [...callsByIndex.values()],
    finishReason,
    usage,
    providerModel,
  };
}

/**
 * Dial one model on the gateway's Chat Completions route. Returns the
 * response unread: the caller decides, from its status, whether to read it
 * as a completion or as a refusal. `timeoutMs` bounds one attempt; the env
 * waits 30 seconds per answer, so an attempt that hangs is abandoned rather
 * than taking the prompt down with it.
 */
export async function dialGateway(
  url: string,
  key: string,
  body: Json,
  timeoutMs: number,
): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${key}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

/**
 * Read a successful gateway response whole, JSON or streamed, as one
 * completion. A streamed answer that ends without a usage chunk keeps
 * usage null; the wire layer fills it from the texts instead.
 */
export async function readCompletion(response: Response, streamed: boolean): Promise<UpstreamCompletion> {
  if (!streamed) {
    const answer: unknown = await response.json();
    return completionOfJson(isRecord(answer) ? answer : {});
  }
  if (response.body === null) {
    return { text: '', toolCalls: [], finishReason: 'stop', usage: null, providerModel: null };
  }
  return completionOfStream(response.body);
}
