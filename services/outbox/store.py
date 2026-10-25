"""Durable outbox with at-least-once Slack delivery and duplicate suppression.

Each incident is a JSON file in data/outbox/<incident_id>.json holding ordered messages.
Message states: pending -> sending -> sent | failed(retry) ; "sending" that survives a crash is
reconciled on startup by searching recent Slack history for the message marker [gt:<msg_id>].
Exactly-once is NOT promised: a crash between Slack accepting and us recording `slack_ts` can
duplicate; the marker + reconcile step makes that rare and detectable.

Slack sinks (first configured wins):
  SLACK_BOT_TOKEN + SLACK_CHANNEL  -> chat.postMessage (threaded follow-ups, reconcile supported)
  SLACK_WEBHOOK_URL                -> incoming webhook (no reconcile possible, at-least-once only)
  none                             -> messages stay in the outbox (offline demo), status "no_sink"
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUTBOX_DIR = Path(os.environ.get("GT_DATA_DIR", ROOT / "data")) / "outbox"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
CHANNEL = os.environ.get("SLACK_CHANNEL")
WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")


class Outbox:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=8.0)
        self.online: Optional[bool] = None
        self.stats = {"queued": 0, "sent": 0, "failed_attempts": 0, "reconciled": 0, "duplicates_suppressed": 0}
        self._lock = asyncio.Lock()

    # -- storage ---------------------------------------------------------------
    def _path(self, iid: str) -> Path:
        return OUTBOX_DIR / f"{iid}.json"

    def load(self, iid: str) -> Dict[str, Any]:
        p = self._path(iid)
        if p.exists():
            return json.loads(p.read_text())
        return {"incident_id": iid, "created": time.time(), "thread_ts": None, "messages": []}

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
            msg = {"id": f"{iid}/{len(rec['messages']) + 1}", "kind": kind, "text": text, "status": "pending",
                   "attempts": 0, "created": time.time(), "slack_ts": None, "last_error": None, "dedupe_key": dedupe_key}
            rec["messages"].append(msg)
            self.save(rec)
            self.stats["queued"] += 1
            return rec

    async def replace_pending_body(self, iid: str, kind: str, text: str) -> bool:
        """If the latest message of this kind has not been sent yet, update it in place (no duplicate post)."""
        async with self._lock:
            rec = self.load(iid)
            for m in reversed(rec["messages"]):
                if m["kind"] == kind and m["status"] == "pending":
                    m["text"] = text
                    self.save(rec)
                    return True
            return False

    # -- delivery ----------------------------------------------------------------
    def sink(self) -> str:
        if BOT_TOKEN and CHANNEL:
            return "bot"
        if WEBHOOK:
            return "webhook"
        return "none"

    async def _post(self, rec: Dict[str, Any], m: Dict[str, Any]) -> Optional[str]:
        text = f"{m['text']}\n_[gt:{m['id']}]_"
        if self.sink() == "bot":
            payload = {"channel": CHANNEL, "text": text}
            if rec.get("thread_ts") and m["kind"] != "alert":
                payload["thread_ts"] = rec["thread_ts"]
            r = await self.http.post("https://slack.com/api/chat.postMessage", json=payload,
                                     headers={"Authorization": f"Bearer {BOT_TOKEN}"})
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(data.get("error", "slack error"))
            return data.get("ts")
        r = await self.http.post(WEBHOOK, json={"text": text})
        if r.status_code >= 300:
            raise RuntimeError(f"webhook {r.status_code}")
        return f"webhook-{int(time.time())}"

    async def flush_once(self) -> Dict[str, Any]:
        if self.sink() == "none":
            return {"sink": "none", "pending": sum(1 for r in self.all_records() for m in r["messages"] if m["status"] == "pending")}
        delivered = 0
        for rec in self.all_records():
            changed = False
            for m in rec["messages"]:
                if m["status"] not in ("pending", "failed"):
                    continue
                m["status"] = "sending"; m["attempts"] += 1; changed = True
                self.save(rec)
                try:
                    ts = await self._post(rec, m)
                    m["status"] = "sent"; m["slack_ts"] = ts; m["sent_at"] = time.time(); m["last_error"] = None
                    if m["kind"] == "alert" and not rec.get("thread_ts"):
                        rec["thread_ts"] = ts
                    self.online = True; self.stats["sent"] += 1; delivered += 1
                except Exception as e:  # noqa: BLE001
                    m["status"] = "failed"; m["last_error"] = str(e)[:120]
                    self.stats["failed_attempts"] += 1
                    self.online = False
                    self.save(rec)
                    return {"sink": self.sink(), "online": False, "delivered": delivered, "error": m["last_error"]}
            if changed:
                self.save(rec)
        return {"sink": self.sink(), "online": self.online, "delivered": delivered}

    async def reconcile(self) -> int:
        """Crash recovery: messages stuck in 'sending' -> check Slack history for the marker."""
        stuck = [(r, m) for r in self.all_records() for m in r["messages"] if m["status"] == "sending"]
        if not stuck:
            return 0
        found: Dict[str, str] = {}
        if self.sink() == "bot":
            try:
                r = await self.http.get("https://slack.com/api/conversations.history",
                                        params={"channel": CHANNEL, "limit": 200}, headers={"Authorization": f"Bearer {BOT_TOKEN}"})
                for msg in r.json().get("messages", []):
                    for _, m in stuck:
                        if f"[gt:{m['id']}]" in msg.get("text", ""):
                            found[m["id"]] = msg.get("ts")
            except Exception:  # noqa: BLE001
                pass
        n = 0
        for rec, m in stuck:
            if m["id"] in found:
                m["status"] = "sent"; m["slack_ts"] = found[m["id"]]; self.stats["reconciled"] += 1
            else:
                m["status"] = "pending"  # unknown -> resend (at-least-once)
            n += 1
            self.save(rec)
        return n

    def snapshot(self) -> Dict[str, Any]:
        recs = self.all_records()
        counts: Dict[str, int] = {}
        for r in recs:
            for m in r["messages"]:
                counts[m["status"]] = counts.get(m["status"], 0) + 1
        return {"sink": self.sink(), "online": self.online, "incidents": len(recs), "messages": counts, **self.stats}
