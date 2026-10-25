"""Durable local outbox — the authoritative alert record. No external delivery.

Each incident is a JSON file in data/outbox/<incident_id>.json holding ordered messages.
The dashboard reads it via the agent's /outbox endpoint; it survives restarts and the
offline test by construction (it never touches the network). Duplicate suppression via
dedupe_key keeps repeated rule hits from double-recording.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
OUTBOX_DIR = Path(os.environ.get("GT_DATA_DIR", ROOT / "data")) / "outbox"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)


class Outbox:
    def __init__(self) -> None:
        self.stats = {"recorded": 0, "updated": 0, "duplicates_suppressed": 0}
        self._lock = asyncio.Lock()

    def _path(self, iid: str) -> Path:
        return OUTBOX_DIR / f"{iid}.json"

    def load(self, iid: str) -> Dict[str, Any]:
        p = self._path(iid)
        if p.exists():
            return json.loads(p.read_text())
        return {"incident_id": iid, "created": time.time(), "messages": []}

    def save(self, rec: Dict[str, Any]) -> None:
        tmp = self._path(rec["incident_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(self._path(rec["incident_id"]))

    def all_records(self) -> List[Dict[str, Any]]:
        return [json.loads(p.read_text()) for p in sorted(OUTBOX_DIR.glob("inc_*.json"))]

    async def enqueue(self, iid: str, text: str, kind: str = "alert", dedupe_key: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            rec = self.load(iid)
            if dedupe_key and any(m.get("dedupe_key") == dedupe_key for m in rec["messages"]):
                self.stats["duplicates_suppressed"] += 1
                return rec
            rec["messages"].append({"id": f"{iid}/{len(rec['messages']) + 1}", "kind": kind, "text": text,
                                    "created": time.time(), "dedupe_key": dedupe_key})
            self.save(rec)
            self.stats["recorded"] += 1
            return rec

    async def replace_pending_body(self, iid: str, kind: str, text: str) -> bool:
        """Upgrade the latest message of this kind in place (e.g. template alert -> reason write-up)."""
        async with self._lock:
            rec = self.load(iid)
            for m in reversed(rec["messages"]):
                if m["kind"] == kind:
                    m["text"] = text
                    m["updated"] = time.time()
                    self.save(rec)
                    self.stats["updated"] += 1
                    return True
            return False

    def snapshot(self) -> Dict[str, Any]:
        recs = self.all_records()
        return {"incidents": len(recs), "messages": sum(len(r["messages"]) for r in recs), **self.stats}
