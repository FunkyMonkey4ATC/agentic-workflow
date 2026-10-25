"""Decision engine — deterministic criticality state machine. No model on the hot path.

Consumes:  clearance, track, weather, candidate (keyword path), conflict (external checker, FMK-12)
Produces:  alert (state=candidate|confirmed|uncertain), alert.update (enrichment), conflict (own rules)

Rules (minimal local implementation of FMK-12 so this box demos standalone; Vi's checker can
also feed `conflict` events and they take the same path):
  R1 runway_occupied_clearance : cross_runway / cleared_for_takeoff / line_up_and_wait on runway R
                                 while an aircraft holds cleared_to_land on R and is not yet clear.
  R2 unauthorized_runway_entry : a track reports on_runway=R with no matching clearance for that callsign.
  R3 weather_below_minima      : weather below landing minima while a landing clearance is active.
Stale/missing weather -> uncertain, never silently non-critical.

The immediate alert is a template from the rule evidence (P0). Enrichment via `fast` (P1, bounded
wait) may RAISE severity or add context; it can never suppress a rule-fired alert. `uncertain` is a
first-class state shown on the dashboard and not auto-posted as confirmed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from services.common.bus import Bus
from services.common.events import make_event, now_iso

ROOT = Path(__file__).resolve().parents[2]
SEVERITY_SCHEMA = json.loads((ROOT / "config" / "severity_schema.json").read_text())
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:4000")
LANDING_WINDOW_S = float(os.environ.get("DECISION_LANDING_WINDOW_S", "150"))   # a landing clearance stays "active" this long unless cleared
DEDUP_WINDOW_S = float(os.environ.get("DECISION_DEDUP_WINDOW_S", "90"))
WEATHER_STALE_S = float(os.environ.get("DECISION_WEATHER_STALE_S", "600"))
ENRICH_TIMEOUT_S = float(os.environ.get("DECISION_ENRICH_TIMEOUT_S", "4"))
MINIMA = {"visibility_sm": 0.5, "ceiling_ft": 200}
RULE_SEVERITY = {"runway_occupied_clearance": "critical", "unauthorized_runway_entry": "critical",
                 "weather_below_minima": "high", "external": "critical"}
SEV_RANK = {"non_critical": 0, "uncertain": 1, "high": 2, "critical": 3}

bus = Bus()
http = httpx.AsyncClient(timeout=10.0)


@dataclass
class Landing:
    callsign: str
    runway: str
    t_wall: float
    t_iso: str
    audio: Optional[str]
    text: str
    cleared_of_runway: bool = False


@dataclass
class World:
    landings: Dict[str, Landing] = field(default_factory=dict)          # callsign -> active landing clearance
    runway_entries: List[Dict[str, Any]] = field(default_factory=list)  # recent cross/takeoff/luaw clearances
    tracks: Dict[str, Dict[str, Any]] = field(default_factory=dict)     # callsign -> last track
    weather: Optional[Dict[str, Any]] = None
    weather_wall: float = 0.0
    recent_alerts: Dict[str, float] = field(default_factory=dict)       # incident_id -> wall time
    calls: List[Dict[str, Any]] = field(default_factory=list)           # rolling radio log (for evidence/Q&A)

    def active_landings(self, runway: str) -> List[Landing]:
        now = time.monotonic()
        out = []
        for cs, l in list(self.landings.items()):
            if now - l.t_wall > LANDING_WINDOW_S or l.cleared_of_runway:
                self.landings.pop(cs, None)
                continue
            if l.runway == runway:
                out.append(l)
        return out

    def weather_state(self) -> str:
        if not self.weather:
            return "missing"
        if time.monotonic() - self.weather_wall > WEATHER_STALE_S or self.weather.get("stale"):
            return "stale"
        return "fresh"


world = World()
metrics = {"clearances": 0, "rule_hits": 0, "alerts": 0, "candidates": 0, "deduped": 0, "enriched": 0,
           "enrich_timeouts": 0, "uncertain": 0, "final_to_alert_ms": []}


def incident_id(rule: str, runway: str, callsigns: List[str]) -> str:
    bucket = int(time.time() // DEDUP_WINDOW_S)
    key = f"{rule}|{runway}|{'+'.join(sorted(callsigns))}|{bucket}"
    return "inc_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def template(rule: str, runway: str, callsigns: List[str], evidence: Dict[str, Any], state: str) -> Dict[str, str]:
    cs = " / ".join(callsigns) if callsigns else "unknown traffic"
    if rule == "runway_occupied_clearance":
        title = f"Runway {runway} conflict: {cs}"
        body = (f"{evidence.get('entry_callsign')} was cleared to {evidence.get('entry_action', '').replace('_', ' ')} runway {runway}"
                f"{' at ' + evidence['taxiway'] if evidence.get('taxiway') else ''} while {evidence.get('landing_callsign')} "
                f"holds a landing clearance for runway {runway}.")
    elif rule == "unauthorized_runway_entry":
        title = f"Runway {runway} incursion: {cs}"
        body = f"{cs} is on runway {runway} with no matching clearance."
    elif rule == "weather_below_minima":
        title = f"Weather below minima on runway {runway}"
        body = f"Landing clearance active for {cs} with {evidence.get('weather_summary', 'weather below limits')}."
    elif rule == "keyword_candidate":
        title = f"Possible emergency phrase: {evidence.get('match', '')}"
        body = f"Heard \"{evidence.get('text', '')}\" — candidate from phrase detector, awaiting confirmation."
    else:
        title = f"Conflict ({rule}) runway {runway}: {cs}"
        body = evidence.get("summary", "Conflict reported by checker.")
    if state == "uncertain":
        body += " Data is stale or incomplete; treat as uncertain."
    return {"title": title, "body": body}


async def publish_alert(rule: str, runway: str, callsigns: List[str], evidence: Dict[str, Any],
                        state: str, severity: str, audio: Optional[str], t0_wall: Optional[float] = None) -> Optional[str]:
    iid = incident_id(rule, runway, callsigns)
    now = time.monotonic()
    last = world.recent_alerts.get(iid)
    if last and now - last < DEDUP_WINDOW_S:
        metrics["deduped"] += 1
        return None
    world.recent_alerts[iid] = now
    tpl = template(rule, runway, callsigns, evidence, state)
    if t0_wall is not None:
        metrics["final_to_alert_ms"].append(round((now - t0_wall) * 1000))
    metrics["alerts"] += 1
    if state == "uncertain":
        metrics["uncertain"] += 1
    await bus.publish(make_event("alert", "decision", {
        "incident_id": iid, "rule": rule, "runway": runway, "callsigns": callsigns, "state": state,
        "severity": severity, "title": tpl["title"], "body": tpl["body"], "audio": audio,
        "evidence": evidence, "template": True, "degraded": False,
    }))
    if state == "confirmed":
        asyncio.create_task(enrich(iid, rule, severity, tpl, evidence))
    return iid


async def enrich(iid: str, rule: str, base_severity: str, tpl: Dict[str, str], evidence: Dict[str, Any]) -> None:
    """Bounded `fast` call. May raise severity / add why; cannot lower a rule-fired alert."""
    prompt = (f"Rule fired: {rule}. Alert: {tpl['title']}. {tpl['body']}\n"
              f"Evidence: {json.dumps(evidence)[:1200]}\n"
              "Assess severity for a tower controller. Return JSON.")
    body = {"model": "fast", "temperature": 0.0, "max_tokens": 160,
            "messages": [{"role": "system", "content": "You assess runway safety conflicts at LaGuardia for a tower controller. Be terse and concrete."},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "SeverityAssessment", "schema": SEVERITY_SCHEMA, "strict": True}}}
    try:
        r = await asyncio.wait_for(http.post(GATEWAY_URL + "/v1/chat/completions", json=body,
                                             headers={"X-Priority": "P1", "X-Timeout-S": str(ENRICH_TIMEOUT_S)}), ENRICH_TIMEOUT_S + 1)
        if r.status_code >= 400:
            return
        out = json.loads(r.json()["choices"][0]["message"]["content"])
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        metrics["enrich_timeouts"] += 1
        return
    model_sev = out.get("severity", "uncertain")
    final_sev = base_severity if SEV_RANK.get(model_sev, 0) <= SEV_RANK[base_severity] else model_sev
    metrics["enriched"] += 1
    await bus.publish(make_event("alert.update", "decision", {
        "incident_id": iid, "severity": final_sev, "model_severity": model_sev, "why": out.get("why", ""),
        "confidence": out.get("confidence"), "enriched_by": "fast", "overrode_rule": False,
    }))


# --- rules -----------------------------------------------------------------------
async def on_clearance(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    metrics["clearances"] += 1
    world.calls.append({"t": d.get("t"), "speaker": d.get("speaker"), "callsign": d.get("callsign"), "action": d.get("action"),
                        "runway": d.get("runway"), "taxiway": d.get("taxiway"), "text": d.get("text"), "audio": d.get("audio")})
    world.calls = world.calls[-500:]
    cs, action, rwy = d.get("callsign", ""), d.get("action"), d.get("runway", "")
    t0 = time.monotonic()
    if action == "cleared_to_land" and rwy:
        world.landings[cs] = Landing(cs, rwy, t0, d.get("t"), d.get("audio"), d.get("text", ""))
        st = world.weather_state()
        if st == "fresh" and below_minima(world.weather):
            await publish_alert("weather_below_minima", rwy, [cs], {"weather_summary": summarize_weather(world.weather), "weather": world.weather},
                                "confirmed", "high", d.get("audio"), t0)
        elif st in ("stale", "missing") and os.environ.get("DECISION_WEATHER_REQUIRED", "0") == "1":
            await publish_alert("weather_below_minima", rwy, [cs], {"weather_summary": f"weather {st}"}, "uncertain", "uncertain", d.get("audio"), t0)
        return
    if action in ("cross_runway", "cleared_for_takeoff", "line_up_and_wait") and rwy:
        world.runway_entries.append({"callsign": cs, "action": action, "runway": rwy, "taxiway": d.get("taxiway"), "t_wall": t0, "audio": d.get("audio"), "text": d.get("text")})
        world.runway_entries = world.runway_entries[-50:]
        for landing in world.active_landings(rwy):
            if landing.callsign == cs:
                continue
            metrics["rule_hits"] += 1
            evidence = {"entry_callsign": cs, "entry_action": action, "taxiway": d.get("taxiway"), "landing_callsign": landing.callsign,
                        "calls": [{"t": landing.t_iso, "text": landing.text, "audio": landing.audio}, {"t": d.get("t"), "text": d.get("text"), "audio": d.get("audio")}],
                        "weather": world.weather, "positions": [world.tracks.get(cs), world.tracks.get(landing.callsign)]}
            await publish_alert("runway_occupied_clearance", rwy, [cs, landing.callsign], evidence, "confirmed", "critical", d.get("audio"), t0)
    # action == "stop" is surfaced by the keyword path (candidate); no second alert here.


def below_minima(w: Optional[Dict[str, Any]]) -> bool:
    if not w:
        return False
    vis, ceil = w.get("visibility_sm"), w.get("ceiling_ft")
    return (vis is not None and vis < MINIMA["visibility_sm"]) or (ceil is not None and ceil < MINIMA["ceiling_ft"])


def summarize_weather(w: Dict[str, Any]) -> str:
    return f"visibility {w.get('visibility_sm', '?')} sm, ceiling {w.get('ceiling_ft', '?')} ft ({w.get('raw', '')[:40]})"


async def on_track(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    cs = d.get("callsign", "")
    world.tracks[cs] = d
    if d.get("clear_of_runway") and cs in world.landings:
        world.landings[cs].cleared_of_runway = True
    rwy = d.get("on_runway")
    if rwy and not d.get("simulated_noise"):
        ok = cs in world.landings and world.landings[cs].runway == rwy
        ok = ok or any(e["callsign"] == cs and e["runway"] == rwy and time.monotonic() - e["t_wall"] < LANDING_WINDOW_S for e in world.runway_entries)
        if not ok:
            metrics["rule_hits"] += 1
            await publish_alert("unauthorized_runway_entry", rwy, [cs], {"position": d, "simulated": d.get("simulated", False)}, "confirmed", "critical", None, time.monotonic())


async def on_weather(ev: Dict[str, Any]) -> None:
    world.weather = ev["data"]
    world.weather_wall = time.monotonic()
    if below_minima(world.weather) and not world.weather.get("stale"):
        for cs, l in list(world.landings.items()):
            await publish_alert("weather_below_minima", l.runway, [cs], {"weather_summary": summarize_weather(world.weather), "weather": world.weather}, "confirmed", "high", l.audio)


async def on_candidate(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    metrics["candidates"] += 1
    if d.get("severity_hint") != "critical":
        return
    match = ", ".join(p["match"] for p in d.get("phrases", []) if p["severity_hint"] == "critical")
    await publish_alert("keyword_candidate", "", [], {"match": match, "text": d.get("text"), "utterance_id": d.get("utterance_id"), "final": d.get("final")},
                        "candidate", "high", None)


async def on_external_conflict(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    rule = d.get("rule", "external")
    await publish_alert(rule, str(d.get("runway", "")), list(d.get("callsigns", [])), {**d.get("evidence", {}), "summary": d.get("summary", "")},
                        "uncertain" if d.get("uncertain") else "confirmed", RULE_SEVERITY.get(rule, "critical"), d.get("audio"), time.monotonic())


HANDLERS = {"clearance": on_clearance, "track": on_track, "weather": on_weather, "candidate": on_candidate, "conflict": on_external_conflict}


async def main() -> None:
    print(f"[decision] hub={bus.hub_url} gateway={GATEWAY_URL}", flush=True)
    async for ev in bus.subscribe(types=list(HANDLERS)):
        h = HANDLERS.get(ev["type"])
        if h:
            try:
                await h(ev)
            except Exception as e:  # noqa: BLE001
                await bus.publish(make_event("decision.error", "decision", {"error": str(e)[:200], "event": ev.get("type")}))


if __name__ == "__main__":
    asyncio.run(main())
