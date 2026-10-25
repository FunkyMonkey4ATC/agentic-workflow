"""Incident / reporting agent on :9200 — plain Python, no agent framework (FMK-14 core).

On `alert` (state=confirmed): the template text is recorded in the durable local outbox
immediately (P0 text, no model). Then an async `reason` call (P2, coalesced per incident)
writes the grounded what/where/why with the audio clip and radio-log evidence and upgrades
the outbox record in place. Uncertain/candidate alerts are kept for the dashboard but never
recorded as confirmed. The outbox is the authoritative alert log; the dashboard reads it
via GET /outbox. No external delivery (Slack was removed by team decision 2026-10-25).

POST /ask {"question": "When was the truck cleared to cross runway 4?"}  -> P3 `reason` over the radio log
GET  /outbox   GET /alerts   GET /health
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

app = FastAPI(title="Ground Truth incident agent")
bus = Bus()
outbox = Outbox()
http = httpx.AsyncClient(timeout=120.0)
alerts: Dict[str, Dict[str, Any]] = {}
metrics = {"alerts_seen": 0, "confirmed": 0, "writeups": 0, "writeup_degraded": 0, "writeup_ms": [], "questions": 0}

WRITER_SYSTEM = """You write short, grounded runway-safety alerts for an airport tower controller and the ops team.
Format exactly three labelled lines: WHAT: ... WHERE: ... WHY: ...  then one line SOURCE: with the radio call timestamps and audio clip file.
Use only the evidence given. Never invent callsigns, runways or times. Plain language, no markdown headers, under 90 words."""


def alert_text(a: Dict[str, Any], body: str, enriched: bool) -> str:
    sim = " (simulated)" if a.get("evidence", {}).get("simulated") else ""
    head = f"{a['title']}{sim} — {a.get('severity', '').upper()} · {a.get('state')}"
    src = f"\nclip: {a['audio']}" if a.get("audio") else ""
    foot = "" if enriched else "\n[initial alert from rules; explanation follows]"
    return f"{head}\n{body}{src}\n{a['incident_id']}{foot}"


async def write_up(a: Dict[str, Any]) -> None:
    iid = a["incident_id"]
    evidence = a.get("evidence", {})
    calls = evidence.get("calls") or []
    prompt = (f"Alert title: {a['title']}\nRule: {a['rule']}\nRunway: {a.get('runway')}\nCallsigns: {a.get('callsigns')}\n"
              f"Radio calls (evidence):\n" + "\n".join(f"- {c.get('t')}: \"{c.get('text')}\" (clip {c.get('audio')})" for c in calls) +
              f"\nPositions: {json.dumps(evidence.get('positions'))[:400]}\nWeather: {json.dumps(evidence.get('weather'))[:300]}\n"
              f"Audio clip for this alert: {a.get('audio')}")
    body = {"model": "reason", "temperature": 0.2, "max_tokens": 1200,
            # gpt-oss (harmony) spends tokens in a reasoning channel first; keep effort low and
            # the budget generous or the final channel comes back empty. Ignored by non-harmony models.
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "messages": [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}]}
    t0 = time.monotonic()
    try:
        r = await http.post(GATEWAY_URL + "/v1/chat/completions", json=body,
                            headers={"X-Priority": "P2", "X-Coalesce-Key": iid, "X-Timeout-S": "60"})
        if r.status_code >= 400:
            raise RuntimeError(f"reason unavailable ({r.status_code})")
        msg = r.json()["choices"][0]["message"]
        text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()[:600]
        if not text:
            raise RuntimeError("empty completion")
        backend = r.json().get("gateway", {}).get("backend")
    except Exception as e:  # noqa: BLE001
        metrics["writeup_degraded"] += 1
        await bus.publish(make_event("alert.update", "agent", {"incident_id": iid, "degraded": True, "error": str(e)[:160]}))
        return
    ms = round((time.monotonic() - t0) * 1000)
    metrics["writeup_ms"].append(ms); metrics["writeups"] += 1
    a["explanation"] = text
    await bus.publish(make_event("alert.update", "agent", {"incident_id": iid, "body": text, "enriched_by": "reason", "backend": backend, "writeup_ms": ms}))
    upgraded = await outbox.replace_pending_body(iid, "alert", alert_text(a, text, True))
    if not upgraded:
        await outbox.enqueue(iid, f"Explanation\n{text}", kind="followup", dedupe_key=f"{iid}:writeup")


async def on_alert(ev: Dict[str, Any]) -> None:
    a = ev["data"]
    metrics["alerts_seen"] += 1
    alerts[a["incident_id"]] = a
    if a.get("state") != "confirmed":
        return  # candidates / uncertain: dashboard only, never auto-posted as confirmed
    metrics["confirmed"] += 1
    await outbox.enqueue(a["incident_id"], alert_text(a, a["body"], False), kind="alert", dedupe_key=f"{a['incident_id']}:alert")
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


@app.on_event("startup")
async def _start() -> None:
    asyncio.create_task(consume())
    print(f"[agent] gateway={GATEWAY_URL} hub={HUB_URL} outbox=local-only", flush=True)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "outbox": outbox.snapshot(), "metrics": {k: (v if not isinstance(v, list) else len(v)) for k, v in metrics.items()}}


@app.get("/outbox")
async def get_outbox() -> List[Dict[str, Any]]:
    return outbox.all_records()


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
    rr = await http.post(GATEWAY_URL + "/v1/chat/completions",
                         json={"model": "reason", "messages": msgs, "temperature": 0.1, "max_tokens": 1200,
                               "chat_template_kwargs": {"reasoning_effort": "low"}},
                         headers={"X-Priority": "P3", "X-Timeout-S": "60"})
    if rr.status_code >= 400:
        return {"degraded": True, "status": rr.status_code, "detail": rr.text[:200]}
    m = rr.json()["choices"][0]["message"]
    return {"answer": ((m.get("content") or "").strip() or (m.get("reasoning_content") or "").strip()), "ms": round((time.monotonic() - t0) * 1000),
            "backend": rr.json().get("gateway", {}).get("backend"), "log_lines": len(lines)}
