#!/usr/bin/env python3
"""Compute pipeline latencies and counts from the hub event log (uses hub-assigned recv_ms).

  scripts/metrics.py [--since SEQ] [--hub http://127.0.0.1:7000]

Reports: final transcript -> clearance (parse), clearance -> alert (rules), final transcript -> alert,
transcript -> keyword candidate, alert -> write-up, counts of alerts by state, false alerts (alerts
during scenarios tagged normal), service-side ASR metrics.
"""
import argparse
import json
import statistics
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--since", type=int, default=0)
ap.add_argument("--hub", default="http://127.0.0.1:7000")
ap.add_argument("--asr", default="http://127.0.0.1:9100")
args = ap.parse_args()

with urllib.request.urlopen(f"{args.hub}/events?since={args.since}&limit=5000", timeout=10) as r:
    events = json.load(r)


def pct(xs):
    if not xs:
        return "n=0"
    xs = sorted(xs)
    p50 = xs[len(xs) // 2]
    p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
    return f"n={len(xs)} p50={p50}ms p95={p95}ms max={xs[-1]}ms"


by_utt_final = {}
clearances = []
alerts = []
candidates = []
updates = []
for e in events:
    d = e.get("data", {})
    t = e["type"]
    if t == "transcript.segment" and d.get("final"):
        by_utt_final[d["utterance_id"]] = e
    elif t in ("clearance", "clearance.flagged"):
        clearances.append(e)
    elif t == "alert":
        alerts.append(e)
    elif t == "candidate":
        candidates.append(e)
    elif t == "alert.update":
        updates.append(e)

parse_lat, rule_lat, total_lat, cand_lat, writeup_lat = [], [], [], [], []
for c in clearances:
    f = by_utt_final.get(c["data"].get("utterance_id"))
    if f:
        parse_lat.append(c["recv_ms"] - f["recv_ms"])
for a in alerts:
    if a["data"].get("rule") == "keyword_candidate":
        continue
    prior = [c for c in clearances if c["recv_ms"] <= a["recv_ms"]]
    if prior:
        c = prior[-1]
        rule_lat.append(a["recv_ms"] - c["recv_ms"])
        f = by_utt_final.get(c["data"].get("utterance_id"))
        if f:
            total_lat.append(a["recv_ms"] - f["recv_ms"])
    wu = [u for u in updates if u["data"].get("incident_id") == a["data"]["incident_id"] and u["data"].get("enriched_by") == "reason"]
    if wu:
        writeup_lat.append(wu[0]["recv_ms"] - a["recv_ms"])
for k in candidates:
    segs = [e for e in events if e["type"] == "transcript.segment" and e["data"].get("utterance_id") == k["data"].get("utterance_id") and e["data"].get("rev") == k["data"].get("rev")]
    if segs:
        cand_lat.append(k["recv_ms"] - segs[0]["recv_ms"])

states = {}
for a in alerts:
    key = f"{a['data'].get('state')}/{a['data'].get('severity')}/{a['data'].get('rule')}"
    states[key] = states.get(key, 0) + 1

print(f"events={len(events)} finals={len(by_utt_final)} clearances={len(clearances)} alerts={len(alerts)} candidates={len(candidates)}")
print("final transcript -> clearance (parse):      ", pct(parse_lat))
print("clearance -> alert (rules, template):       ", pct(rule_lat))
print("final transcript -> alert (total):          ", pct(total_lat))
print("transcript segment -> keyword candidate:    ", pct(cand_lat))
print("alert -> reason write-up:                   ", pct(writeup_lat))
print("alerts by state/severity/rule:")
for k, v in sorted(states.items()):
    print(f"   {v:3}  {k}")
try:
    with urllib.request.urlopen(f"{args.asr}/metrics", timeout=5) as r:
        m = json.load(r)
    print("ASR service:", {k: m[k] for k in ("utterances", "provisional", "final", "asr_ms", "audio_end_to_provisional_ms", "audio_end_to_final_ms")})
except Exception as e:  # noqa: BLE001
    print("ASR service metrics unavailable:", e)
