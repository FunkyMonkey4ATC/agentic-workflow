"""Emergency phrase detector — the no-LLM earliest-warning path.

Runs on every ASR segment (provisional included) in parallel with the Qwen parser.
If the parser is slow or down, candidate alerts still fire. Rules + world state
confirm or downgrade them later. This is a demo robustness measure, not an
operationally reliable detector.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# (name, regex, severity_hint)
PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("stop", re.compile(r"\b(stop|stop stop|hold position|hold your position)\b", re.I), "critical"),
    ("go_around", re.compile(r"\bgo[- ]?around\b", re.I), "critical"),
    ("mayday", re.compile(r"\b(mayday|pan[- ]?pan|emergency|declaring an emergency)\b", re.I), "critical"),
    ("abort", re.compile(r"\b(abort|cancel (takeoff|take-off|departure)|reject(ed)? takeoff)\b", re.I), "critical"),
    ("runway_incursion", re.compile(r"\b(on the runway|without (a )?clearance|unauthori[sz]ed|incursion|traffic on (the )?runway)\b", re.I), "critical"),
    ("cross_runway", re.compile(r"\bcross(ing)? runway\s*(\d{1,2}[LRC]?)\b", re.I), "high"),
    ("cleared_to_land", re.compile(r"\bcleared to land\b", re.I), "info"),
    ("cleared_for_takeoff", re.compile(r"\bcleared for take-?off\b", re.I), "info"),
    ("line_up_and_wait", re.compile(r"\bline up and wait\b", re.I), "info"),
    ("hold_short", re.compile(r"\bhold short\b", re.I), "info"),
    ("correction", re.compile(r"\b(correction|disregard|negative)\b", re.I), "info"),
    ("vehicle", re.compile(r"\b(truck|rescue|ops|vehicle|follow[- ]?me)\s*\d*\b", re.I), "info"),
]

RANK = {"critical": 3, "high": 2, "info": 1}


def detect(text: str) -> Dict[str, object]:
    hits = []
    worst = None
    for name, rx, sev in PATTERNS:
        m = rx.search(text or "")
        if m:
            hits.append({"phrase": name, "match": m.group(0), "severity_hint": sev})
            if worst is None or RANK[sev] > RANK[worst]:
                worst = sev
    return {"phrases": hits, "severity_hint": worst, "is_candidate": worst in ("critical", "high")}
