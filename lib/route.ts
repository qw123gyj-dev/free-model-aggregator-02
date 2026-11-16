// Routing: which free model serves a prompt, and what happens when it refuses.
//
// Quality is 60% of the score and it is the house's published rating of the
// model that really served each prompt, over the best rating for a free model
// meeting that prompt's requirements. So a prompt is always served by the
// best-rated free model that meets its requirements, and failover walks down
// that same order: the next-best model that meets them, never a weaker one
// while a stronger one is still untried.
//
// A refusal is never an answer. The gateway relays a provider's 429 or 5xx as
// the provider sent it, and this file turns each refusal into a cooldown so
// one rate-limited model cannot take every later prompt down with it:
//
//   - 429: the model is cooling down for the rest of the trial; later prompts
//     never spend a call on it.
//   - 5xx: one retry on the same model (a transient failure), then it cools
//     down too.
//   - 4xx other than 401: the request itself is wrong for that model; it cools
//     down at once.
//   - 401: there is no OpenRouter key behind the gateway at all, so no model
//     can serve anything; the dial loop stops at once instead of burning calls.
//
// Every choice here is a pure function of the brief, the request and the
// gateway answers seen so far, so a live pass and its replay make the same
// calls in the same order. No clock and no randomness anywhere in this file.

import {
  capabilitiesOf,
  freeModels,
  meetsRequirements,
  type Brief,
  type Json,
  type ModelRow,
  type Requirement,
} from './brief.ts';

/** One refusal, as the `x-makefun-upstream-failures` trail header carries it. */
export interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** Per-model refusal state for one trial. */
export class Cooldowns {
  private readonly cool = new Set<string>();
  private readonly strikes = new Map<string, number>();

  /** Whether the model is cooling down and must not be dialled again this trial. */
  isCool(modelId: string): boolean {
    return this.cool.has(modelId);
  }

  /**
   * Record a refusal. Returns true when the dial loop should stop trying
   * altogether (a 401: no key behind the gateway, so nothing can serve).
   */
  note(modelId: string, status: number): boolean {
    if (status === 401) {
      this.cool.add(modelId);
      return true;
    }
    if (status === 429) {
      this.cool.add(modelId);
      return false;
    }
    if (status >= 500) {
      const strikes = (this.strikes.get(modelId) ?? 0) + 1;
      this.strikes.set(modelId, strikes);
      if (strikes >= 2) this.cool.add(modelId);
      return false;
    }
    // Any other 4xx: the request is wrong for this model; do not retry it.
    this.cool.add(modelId);
    return false;
  }
}

/**
 * The free models that meet `requires`, best rating first. Ties keep the
 * brief's order (a stable sort), so the order is the same on every pass.
 */
export function rankedCandidates(brief: Brief, requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/**
 * The next model a prompt tries: the best-rated candidate that is not
 * cooling down and has not already been tried for this prompt. A model a
 * prompt already tried is never dialled again for that prompt — except the
 * one same-model retry after a 5xx, which the dial loop schedules itself
 * (see main.ts), so a retry never leaks into the next prompt's ranking.
 */
export function nextCandidate(
  candidates: readonly ModelRow[],
  cooldowns: Cooldowns,
  tried: ReadonlySet<string>,
): ModelRow | null {
  for (const model of candidates) {
    if (cooldowns.isCool(model.id)) continue;
    if (tried.has(model.id)) continue;
    return model;
  }
  return null;
}

/** The requirements a request body itself declares, for when the brief has no row for it. */
export function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/** Merge two requirement sets: a requirement either source names must be met. */
export function mergeRequirements(a: Requirement, b: Requirement): Requirement {
  const out: Requirement = {};
  if (a.tools === true || b.tools === true) out.tools = true;
  if (a.vision === true || b.vision === true) out.vision = true;
  if (a.json === true || b.json === true) out.json = true;
  const min = Math.max(a.min_context ?? 0, b.min_context ?? 0);
  if (min > 0) out.min_context = min;
  return out;
}

/**
 * A token-count estimate, used only when the provider sent no usage at all:
 * roughly one token per four characters of text, a whole count, at least one
 * output token. The wire checks need a whole input count of at least 0 and an
 * output count of at least 1; a missing usage must never fail a prompt that
 * was otherwise served. Deterministic in the texts it is computed from.
 */
export function estimateUsage(requestText: string, responseText: string): { prompt_tokens: number; completion_tokens: number } {
  return {
    prompt_tokens: Math.max(0, Math.ceil(requestText.length / 4)),
    completion_tokens: Math.max(1, Math.ceil(responseText.length / 4)),
  };
}

/** Every text the client sent, for the usage estimate: message texts and the system prompt. */
export function requestTextOf(body: Json): string {
  const parts: string[] = [];
  const system = body.system;
  if (typeof system === 'string') parts.push(system);
  else if (Array.isArray(system)) {
    for (const block of system) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
    }
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (typeof content === 'string') parts.push(content);
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
      }
    }
  }
  return parts.join('\n');
}
