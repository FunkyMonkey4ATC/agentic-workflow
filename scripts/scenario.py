#!/usr/bin/env python3
"""Scenario injector (typed-transcript path). Drives the pipeline without audio so the incident
workflow can be demonstrated and timed deterministically. Everything injected here is SIMULATED.

  scripts/scenario.py lga        # Jazz 646 cleared to land rwy 4; Truck 1 cleared to cross at D -> R1 critical
  scripts/scenario.py normal     # routine traffic, expect zero alerts (false-alert baseline)
  scripts/scenario.py incursion  # vehicle track on runway 4 with no clearance -> R2
  scripts/scenario.py weather    # METAR below minima with landing active -> R3
  --speed 2.0  (faster)  --asr http://127.0.0.1:9100  --hub http://127.0.0.1:7000
"""
import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("scenario", choices=["lga", "normal", "incursion", "weather"])
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--asr", default="http://127.0.0.1:9100")
ap.add_argument("--hub", default="http://127.0.0.1:7000")
args = ap.parse_args()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def say(text, t, speaker=None):
    print(f"  t+{t:>5.1f}s  RADIO  {text}")
    return post(args.asr + "/inject", {"text": text, "t_start": t, "t_end": t + 3, "speaker": speaker, "audio": None})


def event(etype, data, t):
    print(f"  t+{t:>5.1f}s  {etype.upper():6} {json.dumps(data)[:90]}")
    return post(args.hub + "/events", {"type": etype, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "scenario", "data": {**data, "simulated": True}})


SCENARIOS = {
    # NTSB timeline (simplified): landing clearance, readback, crossing clearance ~20 s before the collision, stop calls.
    "lga": [
        (0.0, "radio", "Jazz six forty six runway four cleared to land wind zero four zero at one two"),
        (4.0, "radio", "cleared to land runway four Jazz six forty six"),
        (12.0, "weather", {"station": "KLGA", "visibility_sm": 10, "ceiling_ft": 4000, "wind": "04012KT", "raw": "KLGA 252351Z 04012KT 10SM FEW040", "as_of": "now", "stale": False}),
        (15.0, "track", {"callsign": "JAZZ 646", "lat": 40.766, "lon": -73.885, "alt_ft": 600, "gs_kt": 135, "on_runway": None, "clear_of_runway": False}),
        (20.0, "radio", "Truck one cross runway four at delta"),
        (24.0, "radio", "cross runway four at delta truck one"),
        (30.0, "track", {"callsign": "TRUCK 1", "lat": 40.775, "lon": -73.873, "alt_ft": 0, "gs_kt": 15, "on_runway": "4", "clear_of_runway": False}),
        (36.0, "radio", "Truck one stop stop stop"),
    ],
    "normal": [
        (0.0, "radio", "Delta four five two runway four cleared to land"),
        (4.0, "radio", "cleared to land runway four Delta four five two"),
        (10.0, "radio", "American eleven twenty runway one three line up and wait"),
        (14.0, "radio", "line up and wait runway one three American eleven twenty"),
        (22.0, "track", {"callsign": "DELTA 452", "lat": 40.777, "lon": -73.872, "alt_ft": 0, "gs_kt": 20, "on_runway": "4", "clear_of_runway": False}),
        (30.0, "track", {"callsign": "DELTA 452", "lat": 40.779, "lon": -73.870, "alt_ft": 0, "gs_kt": 12, "on_runway": None, "clear_of_runway": True}),
        (34.0, "radio", "American eleven twenty runway one three cleared for takeoff"),
        (38.0, "radio", "Truck one hold short runway four at delta"),
        (42.0, "radio", "hold short runway four at delta truck one"),
    ],
    "incursion": [
        (0.0, "track", {"callsign": "OPS 3", "lat": 40.776, "lon": -73.872, "alt_ft": 0, "gs_kt": 18, "on_runway": "4", "clear_of_runway": False}),
    ],
    "weather": [
        (0.0, "radio", "Jazz six forty six runway four cleared to land"),
        (5.0, "weather", {"station": "KLGA", "visibility_sm": 0.25, "ceiling_ft": 100, "wind": "04008KT", "raw": "KLGA 252351Z 04008KT 1/4SM FG VV001", "as_of": "now", "stale": False}),
    ],
}

steps = SCENARIOS[args.scenario]
print(f"scenario={args.scenario} speed={args.speed}x  (SIMULATED)")
t0 = time.monotonic()
for t, kind, payload in steps:
    delay = t0 + t / args.speed - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    if kind == "radio":
        say(payload, t)
    else:
        event(kind, payload, t)
print("done")
