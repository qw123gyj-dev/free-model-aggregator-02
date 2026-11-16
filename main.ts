// free-model-aggregator-02: one server, two client wires, and a real free model
// behind every answer.
//
// The client points at this server, and every completion it answers comes
// from a real free OpenRouter model dialled through the grader's gateway on
// the entrant's own pushed key — never an answer of this server's own. It
// serves both wires the trial's client speaks:
//
//   POST /v1/chat/completions   OpenAI Chat Completions, JSON or server-sent events
//   POST /v1/messages           Anthropic Messages, JSON or server-sent events
//   GET  /v1/models             the model list, in the shape of the auth header
//                               it got: Authorization: Bearer means OpenAI,
//                               x-api-key means Anthropic
//
// A trial runs this server as `node main.ts` from the repository's root, with
// Node 22's standard library alone (nothing is installed; TypeScript runs
// directly, and a relative import needs its .ts extension). The env is at
// $MAKEFUN_ENV_URL; model calls go through the grader's gateway at
// $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
// the key pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE carries
// two documents: the free model list (each free OpenRouter model's gateway id,
// its published rating, what it can do, and its free flag) and the six
// prompts' manifest, in the order the env sends them. The server:
//
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there
//      on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. answers each prompt by dialling ONE free model through the gateway.
//      OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//      OpenRouter key pushed to the competition; the real key never enters
//      the sandbox;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so it never waits for anything else.
//
// `node main.ts --list` prints the same catalog JSON the register body
// carries, byte for byte, and exits 0 before any server starts.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and
//   scores the model that really answered. This server sends the brief's
//   model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it
//   without the gateway's `openrouter/` prefix), so its claim is the call it
//   made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// ROUTING (quality, 60% of the score)
//   A prompt is served by the best-rated free model that meets its
//   requirements (the brief's rating_milli, ties in the brief's order), so a
//   JSON-mode prompt goes to the best JSON-capable free model and every other
//   prompt to the best free model overall. The requirements come from the
//   brief's manifest row for the prompt now arriving, merged with what the
//   request body itself declares (tools, JSON mode, images), so either source
//   alone still routes correctly.
//
// FAILOVER (10% of the score) AND RATE LIMITS
//   A gateway 4xx or 5xx is never an answer: the prompt is retried on the
//   next free model that meets it, and every refusal is reported on the
//   response in the `x-makefun-upstream-failures` trail header (at most 8
//   rows of {"status", "code", "model"}), the only way the failover column
//   can see a refusal the server recovered from. Per-model cooldowns keep
//   one rate-limited model from taking every later prompt down with it: a 429
//   cools its model down for the rest of the trial, a 5xx earns one same-model
//   retry and then cools it down too, and a 401 stops the dial loop at once
//   (no key behind the gateway: nothing can serve). At most three attempts
//   are spent on one prompt, so the trial's 60-call cap is never near.
//
// DETERMINISM
//   Every final trial runs twice, live and then replayed, and the replay must
//   leave the env's state export byte for byte. So every choice is a function
//   of the brief, the requests and the gateway's answers: no wall clock, no
//   unseeded randomness, response ids from the request's arrival number, and
//   fixed timestamps in the model listings. Waiting is fine; a time is never
//   written into anything sent.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  Cooldowns,
  estimateUsage,
  mergeRequirements,
  nextCandidate,
  rankedCandidates,
  requestTextOf,
  requirementsFromBody,
  type UpstreamFailure,
} from './lib/route.ts';
import {
  chatCompletionsUrl,
  dialGateway,
  errorCodeOf,
  readCompletion,
  type UpstreamCompletion,
} from './lib/gateway.ts';
import {
  anthropicError,
  anthropicJsonResponse,
  anthropicModelList,
  anthropicStreamFrames,
  anthropicToOpenAI,
  openAIError,
  openAIJsonResponse,
  openAIModelList,
  openAIRequestOf,
  openAIStreamFrames,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most attempts spent on one prompt: the best model, then its failovers. */
const MAX_ATTEMPTS = 3;
/** One gateway attempt's bound, in milliseconds: the env waits 30 s per answer. */
const DIAL_TIMEOUT_MS = 15000;
/** How often POST /register is retried while the env is not answering. */
const REGISTER_ATTEMPTS = 5;
/** The pause between register retries, in milliseconds. */
const REGISTER_RETRY_MS = 500;
/** The trail header's name, as the rules spell it. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the completion request now arriving is: the brief's prompts are in arrival order. */
let manifestIndex = 0;
/** Completion requests served so far: response ids count from it, so they are deterministic. */
let requestNumber = 0;
/** Per-model refusal state for this trial. */
const cooldowns = new Cooldowns();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/**
 * The requirements of the prompt now arriving: the brief's row by arrival
 * order, merged with what the body itself declares. Either source alone
 * still routes the prompt to a model that meets it.
 */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return mergeRequirements(entry.requires, fromBody);
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: UpstreamFailure[]): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: UpstreamFailure[]): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

function sendStream(res: ServerResponse, frames: string[], failures: UpstreamFailure[]): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(200, headers);
  for (const frame of frames) res.write(frame);
  res.end();
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list), []);
  else sendJson(res, 200, openAIModelList(list), []);
}

/** One dialled model, answered: the completion it served, or null when it refused. */
interface Dialled {
  completion: UpstreamCompletion;
  model: ModelRow;
}

/**
 * Dial the prompt's models in ranked order until one serves it. Each refusal
 * is recorded for the trail header and cools its model down (see route.ts);
 * a 401 stops the loop at once. A 5xx earns one same-model retry, scheduled
 * for this prompt only — a retry never leaks into the next prompt's ranking.
 * At most MAX_ATTEMPTS calls are spent.
 */
async function serveWithFailover(
  candidates: ModelRow[],
  upstreamBody: Json,
  streamed: boolean,
): Promise<{ dialled: Dialled | null; failures: UpstreamFailure[] }> {
  const failures: UpstreamFailure[] = [];
  const tried = new Set<string>();
  let retryModel: ModelRow | null = null;
  const url = chatCompletionsUrl(GATEWAY_URL);
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    let model: ModelRow | null = null;
    if (retryModel !== null && !cooldowns.isCool(retryModel.id)) {
      model = retryModel;
      retryModel = null;
    } else {
      model = nextCandidate(candidates, cooldowns, tried);
      if (model === null) break;
    }
    tried.add(model.id);
    let response: Response;
    try {
      response = await dialGateway(url, GATEWAY_KEY, { ...upstreamBody, model: model.id }, DIAL_TIMEOUT_MS);
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      log(`dial ${model.id} failed to reach the gateway: ${(err as Error).message}`);
      cooldowns.note(model.id, 502);
      continue;
    }
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      const code = errorCodeOf(response.status, text);
      failures.push({ status: response.status, code, model: model.id });
      log(`dial ${model.id} refused HTTP ${response.status} (${code})`);
      if (cooldowns.note(model.id, response.status)) break;
      if (response.status >= 500 && !cooldowns.isCool(model.id)) retryModel = model;
      continue;
    }
    if (response.body === null && streamed) {
      failures.push({ status: 502, code: 'upstream_error', model: model.id });
      cooldowns.note(model.id, 502);
      continue;
    }
    const completion = await readCompletion(response, streamed);
    return { dialled: { completion, model }, failures };
  }
  return { dialled: null, failures };
}

/** A refused prompt as a 502 in the client's wire: the prompt is lost, the server stands. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail = last === undefined ? '' : ` (the gateway answered HTTP ${last.status})`;
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway refused every free model that meets this request${detail}`, failures);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway refused every free model that meets this request${detail}`, failures);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
    return;
  }
  requestNumber += 1;
  const requestId = `chatcmpl-agg-${requestNumber}`;
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) {
    sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request', []);
    return;
  }
  const streamed = body.stream === true;
  const { dialled, failures } = await serveWithFailover(candidates, openAIRequestOf(body, ''), streamed);
  if (dialled === null) {
    refused(res, 'openai', failures);
    return;
  }
  const fallback = estimateUsage(requestTextOf(body), dialled.completion.text);
  if (streamed) {
    sendStream(res, openAIStreamFrames(dialled.completion, dialled.model.id, requestId, fallback), failures);
    return;
  }
  sendJson(res, 200, openAIJsonResponse(dialled.completion, dialled.model.id, requestId, fallback), failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
    return;
  }
  requestNumber += 1;
  const requestId = `msgagg-${requestNumber}`;
  const candidates = rankedCandidates(brief, requirementsFor(body));
  if (candidates.length === 0) {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request', []);
    return;
  }
  const streamed = body.stream === true;
  const { dialled, failures } = await serveWithFailover(candidates, anthropicToOpenAI(body, ''), streamed);
  if (dialled === null) {
    refused(res, 'anthropic', failures);
    return;
  }
  const fallback = estimateUsage(requestTextOf(body), dialled.completion.text);
  if (streamed) {
    sendStream(res, anthropicStreamFrames(dialled.completion, dialled.model.id, requestId, fallback), failures);
    return;
  }
  sendJson(res, 200, anthropicJsonResponse(dialled.completion, dialled.model.id, requestId, fallback), failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
    [],
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: string = '';
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = (err as Error).message;
      if (attempt < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  throw new Error(`POST ${ENV_URL}/register failed after ${REGISTER_ATTEMPTS} attempts: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl(GATEWAY_URL) === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
          [],
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
