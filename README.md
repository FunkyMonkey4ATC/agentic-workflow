# Ground Truth — GB10 local inference stack

Local, offline-first runway-safety pipeline for the DGX Spark (GB10, 128 GB unified memory).
Three resident models, one capability gateway, deterministic rules on the hot path, agents that only
ever name a *capability*. Everything in this repo runs on the box with the network off.

## One endpoint for the team

```
base URL:  http://<gb10>:4000/v1          (OpenAI-compatible)
model=fast    -> Qwen3-4B-Instruct-2507 (Q8_0, llama-server :8002)   structured extraction / classification
model=reason  -> gpt-oss-20b (llama-server :8001) when present, else falls back to `fast`
POST /v1/audio/transcriptions  model=asr -> whisper.cpp ATC fine-tune (:9000)
GET  /health                              per-backend + per-capability availability, scheduler stats
```

Headers: `X-Priority: P1|P2|P3` (default P2), `X-Coalesce-Key: <incident_id>` (newer `reason`
request replaces a queued one), `X-Timeout-S`. Overflow -> `429`, superseded -> `409`, no healthy
backend -> `503 {"degraded": true}` (caller renders its template). JSON-schema output:
`response_format: {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}`.

## Services and ports

| Port | Service | Role |
|---|---|---|
| 9000 | `whisper-server` (whisper.cpp, CUDA) | M1 — `jacktol/whisper-medium.en-fine-tuned-for-ATC`, ggml f16 |
| 8002 | `llama-server` (llama.cpp, CUDA) | M2 — `fast`: Qwen3-4B-Instruct-2507 Q8_0, 4 slots |
| 8001 | `llama-server` | M3 — `reason`: gpt-oss-20b MXFP4 (optional) |
| 4000 | `services/gateway` | capability resolution + health (app.py) and workload scheduler (scheduler.py) |
| 7000 | `services/hub` | event bus: `POST /events`, `GET /events?since=&types=`, `WS /ws?types=` ; durable log `data/events.jsonl` |
| 9100 | `services/asr` | audio -> provisional/final `transcript.segment`; keyword `candidate`; `POST /replay`, `/transcribe`, `/inject` (typed transcript) |
| — | `services/parser` | final segment -> FMK-7 `clearance` JSON via `fast` (schema-constrained) |
| — | `services/decision` | deterministic state machine: R1 occupied-runway clearance, R2 unauthorized entry (2.5 s grace), R3 weather minima; template alert first, `fast` enrichment after (may raise, never suppress) |
| 9200 | `services/agent` | incident/reporting: outbox -> Slack (at-least-once + duplicate suppression), `reason` write-up, `POST /ask` radio-log Q&A |

Start/stop everything (each component in a tmux session, logs in `$HF_HOME/logs`):

```bash
scripts/start_stack.sh            # models + services (idempotent)
scripts/start_stack.sh stop
tmux attach -t llama-fast          # inspect a component
```

Secrets go in `.env` (not committed): `SLACK_WEBHOOK_URL=...` or `SLACK_BOT_TOKEN=... SLACK_CHANNEL=...`.
Without either, alerts stay in `data/outbox/` (the offline demo) and flush when configured/online.

## Event flow

```
audio -> ASR (:9100) --provisional/final--> hub --final--> parser (fast) --clearance--> decision
                     \--keyword detector--> candidate ---------------------------------> decision
connectors (tracks / METAR, simulated or live) ----------------------------------------> decision
decision: rule hit -> alert(template, P0) -> fast severity (P1, bounded) -> alert.update
agent:    alert(confirmed) -> outbox -> Slack when online ; reason write-up (P2) upgrades the message
```

Alert states: `candidate` (keyword path, not confirmed), `confirmed` (rule), `uncertain` (stale/missing data).
Only `confirmed` is posted to Slack; everything is on the bus for the dashboard.

## Scenarios, metrics, tests

```bash
scripts/scenario.py lga        # Jazz 646 cleared to land rwy 4; Truck 1 cleared to cross at D -> R1 critical (SIMULATED)
scripts/scenario.py normal     # routine traffic -> expect 0 alerts (false-alert baseline)
scripts/scenario.py incursion  # vehicle on rwy 4 with no clearance -> R2
scripts/scenario.py weather    # METAR below minima with landing active -> R3
scripts/metrics.py --since <seq>   # latencies from the hub log (uses hub recv_ms)
scripts/mock_slack.py              # local webhook sink; stop/start it to simulate network loss
curl -s localhost:9100/metrics     # ASR: asr_ms, audio_end -> provisional / final
curl -s localhost:9200/health      # outbox + agent metrics
```

Measured on the GB10 (2026-10-25, scripted scenarios, Q8_0 4B as both `fast` and `reason`):

| Metric | Value |
|---|---|
| whisper ATC per clip (6–11 s audio) | 200–550 ms |
| transcript segment -> keyword candidate | 2 ms |
| final transcript -> clearance JSON (parse) | p50 1.25 s |
| clearance -> alert (rules + template) | 26 ms |
| final transcript -> confirmed alert | ~1.2 s |
| alert -> reason write-up (Qwen fallback) | 3.3 s |
| normal-traffic scenario | 0 alerts |
| outbox offline->online | all flushed, 0 duplicates |

## Build notes (why llama.cpp, not vLLM)

Venue uplink measured 1.3 MB/s per connection / ~4.7 MB/s aggregate. A vLLM container or wheel set
(several GB of CUDA libs) was not reachable in time; `llama.cpp` and `whisper.cpp` build from
source in minutes with the on-box CUDA 13 (`-DCMAKE_CUDA_ARCHITECTURES=121`), and GGUF weights are
half the size. `scripts/convert_whisper_hf_to_ggml.py` converts the HF safetensors without torch.

Weights live under `$HF_HOME` (`/home/dell/hf`): `hub/` (HF cache), `ggml/whisper-atc/ggml-model.bin`.
`HF_HUB_OFFLINE=1` is set for every service by the start script.

This is a hackathon prototype decision-support demo, not a validated ATC safety system.
