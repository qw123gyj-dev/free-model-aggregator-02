# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/qw123gyj-dev/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/qw123gyj-dev/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two API wires, fourteen free models behind them. Point any
OpenAI- or Anthropic-compatible client at it and every completion it answers
comes from a real free OpenRouter model, dialled through the grader's
gateway — never an answer of its own.

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or server-sent events
- `POST /v1/messages` — Anthropic Messages, JSON or server-sent events
- `GET /v1/models` — the free-model list, in the shape of the auth header it
  got (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the
  Anthropic one)
- `node main.ts --list` — prints the catalog document it serves, byte for
  byte as it registers it

Routing is by the house's published quality rating: each prompt is served by
the best-rated free model that meets its requirements — the best JSON-capable
free model for a JSON-mode prompt, the best free model overall for the rest.
A gateway refusal is never an answer: the prompt is retried on the next
free model that meets it, per-model cooldowns keep one rate-limited model
from taking later prompts down with it, and every refusal the server
recovers from is reported on the response in the
`x-makefun-upstream-failures` trail header. Everything is deterministic —
no wall clock, no unseeded randomness — so a trial's replay leaves the same
state as its live pass.

## Run it

```sh
node main.ts
```

A trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL`
and the trial's dummy `OPENAI_API_KEY` for it; the server picks a free port,
registers it with the env, serves the trial's six prompts, and exits 0 when
the env says the work is done. It needs Node 22 and nothing else — no
dependencies, standard library only.

```sh
node main.ts --list
```

prints the catalog and exits 0, with no brief and no env needed.

## Layout

- `main.ts` — the server: routes, register, work polling, the dial loop
- `lib/brief.ts` — the brief's two documents (free model list, prompt manifest) and the catalog
- `lib/route.ts` — rating-ordered routing, per-model cooldowns, usage estimates
- `lib/gateway.ts` — the gateway's Chat Completions calls and their answers, read whole
- `lib/wire.ts` — the two client wires, their translation, and their SSE event order

---

<sub>Entered on [make.fun](https://make.fun/?ref=ec3esm0k&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
