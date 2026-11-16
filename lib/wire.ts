// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. Every prompt goes upstream as a Chat
// Completions request — the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and renders the gateway's answer back, with every `model`
//     field set to the id this server dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this server dialled.
//   OpenRouter's own answer names a free model without the gateway's
//   `openrouter/` prefix; these functions put the dialled id back on every
//   answer and every streamed chunk. They never make a model up: the id came
//   from the brief and went to the gateway in the request's `model`.
//
// Answers are rendered from a completion read whole (see gateway.ts), so the
// event order is fully under this file's control: content before the finish
// chunk, the usage chunk after it, `[DONE]` last on the OpenAI wire; and on
// the Anthropic wire message_start first, every content block closed before
// message_delta, message_stop last, nothing after it. Usage is the
// provider's where it sent one, else an estimate from the texts (see
// route.ts): the wire checks need a whole input count of at least 0 and an
// output count of at least 1, and a missing usage must never fail a prompt
// that was otherwise served.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same gateway answer gives the same bytes on a replay.
// Response ids are the request's arrival number, and the model listings use
// fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';
import type { UpstreamCompletion, UpstreamToolCall } from './gateway.ts';

/** One model this server serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The output cap a request gets when it names none: the gateway refuses a call with no cap. */
const DEFAULT_MAX_TOKENS = 1024;

/** GET /v1/models on an OpenAI-shaped request: the models this server serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** The output cap a request carries, or the default when it names none. */
function outputCap(body: Json): number {
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (max === null || !Number.isFinite(max) || max <= 0) return DEFAULT_MAX_TOKENS;
  return Math.min(Math.floor(max), 8192);
}

/**
 * An OpenAI-shaped client body as the gateway request that names `model`:
 * the body as sent, the dialled id in `model`, an output cap always present
 * (the gateway refuses a call with none), and the usage chunk asked for on a
 * stream (the gateway forwards it, so a streamed answer carries its usage).
 */
export function openAIRequestOf(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (body.stream === true) {
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages, max_tokens: outputCap(body) };
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Usage as the client wire reads it: the provider's, or the estimate handed in. */
export function usagePair(
  completion: UpstreamCompletion,
  fallback: { prompt_tokens: number; completion_tokens: number },
): { prompt_tokens: number; completion_tokens: number } {
  return completion.usage ?? fallback;
}

/** A provider answer as a Chat Completions object. `model` is the id this server dialled. */
export function openAIJsonResponse(
  completion: UpstreamCompletion,
  model: string,
  requestId: string,
  fallback: { prompt_tokens: number; completion_tokens: number },
): Json {
  const usage = usagePair(completion, fallback);
  const message: Json = { role: 'assistant', content: completion.text };
  if (completion.toolCalls.length > 0) {
    message.tool_calls = completion.toolCalls.map((call) => ({
      id: call.id !== '' ? call.id : `call_${requestId}`,
      type: 'function',
      function: { name: call.name, arguments: call.arguments },
    }));
  }
  return {
    id: requestId,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: completion.finishReason }],
    usage: {
      prompt_tokens: usage.prompt_tokens,
      completion_tokens: usage.completion_tokens,
      total_tokens: usage.prompt_tokens + usage.completion_tokens,
    },
  };
}

/** A provider answer as an Anthropic Messages object. `model` is the id this server dialled. */
export function anthropicJsonResponse(
  completion: UpstreamCompletion,
  model: string,
  requestId: string,
  fallback: { prompt_tokens: number; completion_tokens: number },
): Json {
  const usage = usagePair(completion, fallback);
  const content: Json[] = [];
  if (completion.text !== '') content.push({ type: 'text', text: completion.text });
  for (const call of completion.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : `toolu_${requestId}`,
      name: call.name,
      input: argumentsOf(call.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: `msg_${requestId}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(completion.finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A provider answer as Chat Completions SSE frames: content, finish, usage, [DONE]. */
export function openAIStreamFrames(
  completion: UpstreamCompletion,
  model: string,
  requestId: string,
  fallback: { prompt_tokens: number; completion_tokens: number },
): string[] {
  const usage = usagePair(completion, fallback);
  const base = { id: requestId, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model };
  const frames: string[] = [];
  frames.push(
    sseEvent(null, {
      ...base,
      choices: [{ index: 0, delta: { role: 'assistant', content: completion.text }, finish_reason: null }],
    }),
  );
  frames.push(
    sseEvent(null, {
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: completion.finishReason }],
    }),
  );
  frames.push(
    sseEvent(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.prompt_tokens + usage.completion_tokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames;
}

/** One tool call as Anthropic stream events: start, input deltas, stop. */
function anthropicToolEvents(call: UpstreamToolCall, index: number): string[] {
  const events: string[] = [
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
    }),
  ];
  if (call.arguments !== '') {
    events.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: call.arguments },
      }),
    );
  }
  events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  return events;
}

/** A provider answer as Anthropic SSE frames, in the wire's event order. */
export function anthropicStreamFrames(
  completion: UpstreamCompletion,
  model: string,
  requestId: string,
  fallback: { prompt_tokens: number; completion_tokens: number },
): string[] {
  const usage = usagePair(completion, fallback);
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${requestId}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (completion.text !== '') {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'text', text: '' },
      }),
    );
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'text_delta', text: completion.text },
      }),
    );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of completion.toolCalls) {
    frames.push(...anthropicToolEvents(call, index));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(completion.finishReason), stop_sequence: null },
      usage: { output_tokens: usage.completion_tokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames;
}
