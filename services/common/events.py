"""Event envelope shared by every service.

{"type": "...", "t": ISO-8601 UTC, "source": "asr|parser|keyword|decision|agent|connector|replay",
 "seq": int (assigned by hub), "data": {...}}

Event types used on the bus:
  transcript.segment   data: utterance_id, rev, final, t_start, t_end, text, audio, latency_ms
  candidate            data: utterance_id, rev, final, phrases[], text, severity_hint
  clearance            data: FMK-7 object (+ utterance_id, rev)
  clearance.flagged    data: FMK-7 object + flags (low confidence; never feeds a rule)
  track / weather      data: connector-normalized, with as_of + stale
  conflict             data: rule, evidence{calls[], positions[], weather}, incident_id (from checker)
  alert                data: incident_id, severity, state, title, body, audio, degraded, template
  alert.update         data: incident_id, body, severity, enriched_by
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_event(etype: str, source: str, data: Dict[str, Any], t: Optional[str] = None) -> Dict[str, Any]:
    return {"type": etype, "t": t or now_iso(), "source": source, "data": data}
