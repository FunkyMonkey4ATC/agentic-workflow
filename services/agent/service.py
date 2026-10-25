"""Incident / reporting agent on :9200 — plain Python, no agent framework (FMK-14 core).

On `alert` (state=confirmed): the template text goes to the durable outbox immediately (P0 text,
no model). Then an async `reason` call (P2, coalesced per incident) writes the grounded
what/where/why with the audio clip and radio-log evidence; if the first message is still
pending it is upgraded in place, otherwise a threaded follow-up is queued. Uncertain/candidate
alerts are recorded for the dashboard but not auto-posted as confirmed.

Slack delivery is handled by the outbox loop: posts when online, keeps everything when offline,
flushes on reconnect (at-least-once + duplicate suppression).

POST /ask {"question": "When was the truck cleared to cross runway 4?"}  -> P3 `reason` over the radio log
GET  /outbox   GET /health   POST /outbox/flush
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

from services.common.bus import Bus
from services.common.events import make_event
from services.outbox.store import Outbox

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:4000")
HUB_URL = os.environ.get("HUB_URL", "http://127.0.0.1:7000")
FLUSH_EVERY_S = float(os.environ.get("OUTBOX_FLUSH_EVERY_S", "5"))

app = FastAPI(title="Ground Truth incident agent")
bus = Bus()
outbox = Outbox()
http = httpx.AsyncClient(timeout=120.0)
alerts: Dict[str, Dict[str, Any]] = {}
metrics = {"alerts_seen": 0, "confirmed": 0, "writeups": 0, "writeup_degraded": 0, "writeup_ms": [], "questions": 0}

WRITER_SYSTEM = """You write short, grounded runway-safety alerts for an airport tower controller and the ops team.
Format exactly three labelled lines: WHAT: ... WHERE: ... WHY: ...  then one line SOURCE: with the radio call timestamps and audio clip file.
Use only the evidence given. Never invent callsigns, runways or times. Plain language, no markdown headers, under 90 words."""


def slack_text(a: Dict[str, Any], body: str, enriched: bool) -> str:
    tag = {"critical": ":rotating_light:", "high": ":warning:", "uncertain": ":grey_question:"}.get(a.get("severity"), ":information_source:")
    sim = " _(simulated)_" if a.get("evidence", {}).get("simulated") else ""
    head = f"{tag} *{a['title']}*{sim} — {a.get('severity', '').upper()} · {a.get('state')}"
    src = f"\nclip: `{a['audio']}`" if a.get("audio") else ""
    foot = "" if enriched else "\n_initial alert from rules; explanation follows_"
    return f"{head}\n{body}{src}\n`{a['incident_id']}`{foot}"


async def write_up(a: Dict[str, Any]) -> None:
    iid = a["incident_id"]
    evidence = a.get("evidence", {})
    calls = evidence.get("calls") or []
    prompt = (f"Alert title: {a['title']}\nRule: {a['rule']}\nRunway: {a.get('runway')}\nCallsigns: {a.get('callsigns')}\n"
              f"Radio calls (evidence):\n" + "\n".join(f"- {c.get('t')}: \"{c.get('text')}\" (clip {c.get('audio')})" for c in calls) +
              f"\nPositions: {json.dumps(evidence.get('positions'))[:400]}\nWeather: {json.dumps(evidence.get('weather'))[:300]}\n"
              f"Audio clip for this alert: {a.get('audio')}")
    body = {"model": "reason", "temperature": 0.2, "max_tokens": 220,
            "messages": [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}]}
    t0 = time.monotonic()
    try:
        r = await http.post(GATEWAY_URL + "/v1/chat/completions", json=body,
                            headers={"X-Priority": "P2", "X-Coalesce-Key": iid, "X-Timeout-S": "60"})
        if r.status_code >= 400:
            raise RuntimeError(f"reason unavailable ({r.status_code})")
        text = r.json()["choices"][0]["message"]["content"].strip()
        backend = r.json().get("gateway", {}).get("backend")
    except Exception as e:  # noqa: BLE001
        metrics["writeup_degraded"] += 1
        await bus.publish(make_event("alert.update", "agent", {"incident_id": iid, "degraded": True, "error": str(e)[:160]}))
        return
    ms = round((time.monotonic() - t0) * 1000)
    metrics["writeup_ms"].append(ms); metrics["writeups"] += 1
    a["explanation"] = text
    await bus.publish(make_event("alert.update", "agent", {"incident_id": iid, "body": text, "enriched_by": "reason", "backend": backend, "writeup_ms": ms}))
    upgraded = await outbox.replace_pending_body(iid, "alert", slack_text(a, text, True))
    if not upgraded:
        await outbox.enqueue(iid, f"*Explanation*\n{text}", kind="followup", dedupe_key=f"{iid}:writeup")


async def on_alert(ev: Dict[str, Any]) -> None:
    a = ev["data"]
    metrics["alerts_seen"] += 1
    alerts[a["incident_id"]] = a
    if a.get("state") != "confirmed":
        return  # candidates / uncertain: dashboard only, never auto-posted as confirmed
    metrics["confirmed"] += 1
    await outbox.enqueue(a["incident_id"], slack_text(a, a["body"], False), kind="alert", dedupe_key=f"{a['incident_id']}:alert")
    asyncio.create_task(write_up(a))


async def on_alert_update(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    a = alerts.get(d.get("incident_id"))
    if a and d.get("severity"):
        a["severity"] = d["severity"]


async def consume() -> None:
    async for ev in bus.subscribe(types=["alert", "alert.update"]):
        try:
            await (on_alert(ev) if ev["type"] == "alert" else on_alert_update(ev))
        except Exception as e:  # noqa: BLE001
            print("[agent] error", e, flush=True)


async def flush_loop() -> None:
    await outbox.reconcile()
    while True:
        res = await outbox.flush_once()
        if res.get("delivered"):
            await bus.publish(make_event("outbox.flushed", "agent", res))
        await asyncio.sleep(FLUSH_EVERY_S)


@app.on_event("startup")
async def _start() -> None:
    asyncio.create_task(consume())
    asyncio.create_task(flush_loop())
    print(f"[agent] gateway={GATEWAY_URL} hub={HUB_URL} slack_sink={outbox.sink()}", flush=True)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "outbox": outbox.snapshot(), "metrics": {k: (v if not isinstance(v, list) else len(v)) for k, v in metrics.items()}}


@app.get("/outbox")
async def get_outbox() -> List[Dict[str, Any]]:
    return outbox.all_records()


@app.post("/outbox/flush")
async def flush_now() -> Dict[str, Any]:
    return await outbox.flush_once()


@app.get("/alerts")
async def get_alerts() -> List[Dict[str, Any]]:
    return list(alerts.values())


@app.post("/ask")
async def ask(body: Dict[str, Any]) -> Dict[str, Any]:
    """Radio-log Q&A (P3). Grounded on clearance events from the hub."""
    q = body.get("question", "").strip()
    if not q:
        return {"error": "question required"}
    metrics["questions"] += 1
    r = await http.get(HUB_URL + "/events", params={"types": "clearance,clearance.flagged,alert", "limit": 300})
    log = [e["data"] for e in r.json()]
    lines = []
    for d in log[-150:]:
        if "action" in d:
            lines.append(f"{d.get('t')} {d.get('speaker')}: {d.get('callsign')} {d.get('action')} rwy={d.get('runway') or '-'} twy={d.get('taxiway') or '-'} conf={d.get('confidence')} :: \"{d.get('text', '')}\"")
        else:
            lines.append(f"{d.get('incident_id')} ALERT {d.get('severity')} {d.get('title')}")
    msgs = [{"role": "system", "content": "You answer questions about a tower radio log for LaGuardia. Quote timestamps and callsigns from the log. If the log does not contain the answer, say so."},
            {"role": "user", "content": "Radio log:\n" + "\n".join(lines) + f"\n\nQuestion: {q}"}]
    t0 = time.monotonic()
    rr = await http.post(GATEWAY_URL + "/v1/chat/completions", json={"model": "reason", "messages": msgs, "temperature": 0.1, "max_tokens": 250},
                         headers={"X-Priority": "P3", "X-Timeout-S": "60"})
    if rr.status_code >= 400:
        return {"degraded": True, "status": rr.status_code, "detail": rr.text[:200]}
    return {"answer": rr.json()["choices"][0]["message"]["content"].strip(), "ms": round((time.monotonic() - t0) * 1000),
            "backend": rr.json().get("gateway", {}).get("backend"), "log_lines": len(lines)}
