"""Parser: finalized transcript segment -> FMK-7 clearance JSON via capability `fast`.

Subscribes to `transcript.segment` (final only by default), asks the gateway for a
schema-constrained completion, and publishes one `clearance` event per clearance.
Low-confidence or partial results are published as `clearance.flagged` and must never
feed a rule. Corrections are resolved by the model into the corrected clearance.

Run:  python -m services.parser.service        (env: HUB_URL, GATEWAY_URL, PARSER_ON_PROVISIONAL=0)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

from services.common.bus import Bus
from services.common.events import make_event

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "config" / "clearance_schema.json").read_text())
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:4000")
ON_PROVISIONAL = os.environ.get("PARSER_ON_PROVISIONAL", "0") == "1"
LOW_CONF = float(os.environ.get("PARSER_LOW_CONF", "0.6"))

SYSTEM = """You convert one air traffic control radio transmission into structured clearances.
Airport context: LaGuardia (KLGA). Runways 4, 22, 13, 31. Taxiways are letters (e.g. D = "delta").
Speakers: tower/ground issue clearances; pilots and vehicles (e.g. "Truck 1", "Rescue 35") read them back.

Rules:
- Output only clearances that are actually in the transmission. If none, return an empty "clearances" list and add the flag "non_clearance".
- action values: cleared_to_land, cleared_for_takeoff, line_up_and_wait, cross_runway, hold_short, taxi, stop, readback.
- A pilot/vehicle repeating a clearance is action "readback" with the runway/taxiway they read back.
- "stop", "stop stop", "hold position", "go around", "cancel takeoff clearance" -> action "stop" with the addressed callsign.
- Handle corrections: "cross runway 2-2, correction runway 4" -> output ONLY the corrected clearance (runway "4") and add flag "correction". Never output the superseded value.
- Normalize callsigns to upper case, airline/vehicle name then digits exactly as spoken: "jazz six forty six" -> "JAZZ 646", "delta four five two" -> "DELTA 452", "truck one" -> "TRUCK 1", "rescue three five" -> "RESCUE 35". Never invent digits.
- If the callsign comes LAST in the transmission (e.g. "cross runway four at delta, truck one"), it is a readback by the pilot/vehicle: speaker "pilot" or "vehicle", action "readback".
- Runway is digits plus optional L/R/C, e.g. "4", "22", "13", "31". Spoken "two two" -> "22", "four" -> "4". Use "" if none.
- Taxiway is a single letter (spoken NATO: delta -> "D"). Use "" if none.
- confidence is your confidence in that clearance (0-1). If the text is garbled or partial, lower it and add "low_confidence" or "partial".
Return JSON only."""

FEWSHOT: List[Dict[str, str]] = [
    {"role": "user", "content": "Jazz six forty six runway four cleared to land wind zero four zero at one two"},
    {"role": "assistant", "content": json.dumps({"speaker": "tower", "clearances": [{"callsign": "JAZZ 646", "action": "cleared_to_land", "runway": "4", "taxiway": "", "confidence": 0.95}], "flags": []})},
    {"role": "user", "content": "Truck one cross runway four at delta"},
    {"role": "assistant", "content": json.dumps({"speaker": "tower", "clearances": [{"callsign": "TRUCK 1", "action": "cross_runway", "runway": "4", "taxiway": "D", "confidence": 0.93}], "flags": []})},
    {"role": "user", "content": "cross runway four at delta truck one"},
    {"role": "assistant", "content": json.dumps({"speaker": "vehicle", "clearances": [{"callsign": "TRUCK 1", "action": "readback", "runway": "4", "taxiway": "D", "confidence": 0.9}], "flags": []})},
    {"role": "user", "content": "truck one stop stop stop"},
    {"role": "assistant", "content": json.dumps({"speaker": "tower", "clearances": [{"callsign": "TRUCK 1", "action": "stop", "runway": "", "taxiway": "", "confidence": 0.9}], "flags": ["emergency_phrase"]})},
]

bus = Bus()
http = httpx.AsyncClient(timeout=30.0)
metrics = {"parsed": 0, "clearances": 0, "flagged": 0, "errors": 0, "degraded": 0, "latency_ms": []}


async def parse_text(text: str, priority: str = "P1") -> Dict[str, Any]:
    body = {
        "model": "fast",
        "messages": [{"role": "system", "content": SYSTEM}, *FEWSHOT, {"role": "user", "content": text}],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_schema", "json_schema": {"name": "ParsedRadioCall", "schema": SCHEMA, "strict": True}},
    }
    r = await http.post(GATEWAY_URL + "/v1/chat/completions", json=body, headers={"X-Priority": priority})
    if r.status_code == 503:
        metrics["degraded"] += 1
        raise RuntimeError("fast capability degraded")
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    return json.loads(content)


_NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "niner": "9"}


def _callsign_position(text: str, callsign: str) -> str:
    """'start' | 'end' | 'unknown' — where the callsign's name token sits in the transmission."""
    name = callsign.split(" ")[0].lower() if callsign else ""
    words = re.findall(r"[a-z]+", text.lower())
    if not name or name not in words:
        return "unknown"
    idx = words.index(name)
    if idx <= 1:
        return "start"
    if idx >= max(2, len(words) - 5):
        return "end"
    return "unknown"


def apply_readback_heuristic(text: str, parsed: Dict[str, Any]) -> None:
    """Deterministic fix for a known 4B weakness: ATC convention puts the callsign FIRST when the
    controller issues a clearance and LAST when the pilot/vehicle reads it back. A small model
    often labels readbacks as tower clearances; that would double-count clearances in the rules."""
    cls = parsed.get("clearances") or []
    if not cls:
        return
    pos = _callsign_position(text, cls[0].get("callsign", ""))
    if pos == "end" and all(c.get("action") != "readback" for c in cls):
        is_vehicle = bool(re.search(r"\b(truck|rescue|ops|vehicle|follow)\b", text, re.I))
        parsed["speaker"] = "vehicle" if is_vehicle else "pilot"
        for c in cls:
            c["readback_of"] = c["action"]
            c["action"] = "readback"
    elif pos == "start" and parsed.get("speaker") in ("pilot", "vehicle") and all(c.get("action") == "readback" for c in cls):
        parsed["speaker"] = "tower"


async def handle_segment(ev: Dict[str, Any]) -> None:
    d = ev["data"]
    if not d.get("final") and not ON_PROVISIONAL:
        return
    text = d.get("text", "").strip()
    if not text:
        return
    t0 = time.monotonic()
    try:
        parsed = await parse_text(text)
    except Exception as e:  # noqa: BLE001
        metrics["errors"] += 1
        await bus.publish(make_event("parser.error", "parser", {"utterance_id": d.get("utterance_id"), "error": str(e)[:200], "text": text}))
        return
    ms = round((time.monotonic() - t0) * 1000)
    metrics["latency_ms"].append(ms)
    metrics["parsed"] += 1
    apply_readback_heuristic(text, parsed)
    flags = list(parsed.get("flags", []))
    if not d.get("final"):
        flags.append("partial")
    for c in parsed.get("clearances", []):
        rec = {
            "t": ev.get("t"), "speaker": parsed.get("speaker", "unknown"), "callsign": c.get("callsign", ""),
            "action": c["action"], "runway": c.get("runway", ""), "taxiway": c.get("taxiway", ""),
            "confidence": float(c.get("confidence", 0)), "audio": d.get("audio"),
            "utterance_id": d.get("utterance_id"), "rev": d.get("rev"), "text": text, "parse_ms": ms,
            "readback_of": c.get("readback_of"),
        }
        low = rec["confidence"] < LOW_CONF or "low_confidence" in flags or "partial" in flags
        if low:
            metrics["flagged"] += 1
            await bus.publish(make_event("clearance.flagged", "parser", {**rec, "flags": sorted(set(flags + ["low_confidence"]))}))
        else:
            metrics["clearances"] += 1
            await bus.publish(make_event("clearance", "parser", {**rec, "flags": flags}))
    if not parsed.get("clearances"):
        await bus.publish(make_event("parser.non_clearance", "parser", {"utterance_id": d.get("utterance_id"), "text": text, "flags": flags, "parse_ms": ms}))


async def main() -> None:
    print(f"[parser] gateway={GATEWAY_URL} hub={bus.hub_url} provisional={ON_PROVISIONAL}", flush=True)
    async for ev in bus.subscribe(types=["transcript.segment"]):
        asyncio.create_task(handle_segment(ev))


if __name__ == "__main__":
    asyncio.run(main())
