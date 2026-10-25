"""Event hub on :7000 — the shared bus.

POST /events            append one event (hub assigns seq), fan out to WebSocket subscribers
GET  /events?since=&types=&limit=   replay from the in-memory ring (durable copy in data/events.jsonl)
WS   /ws?types=a,b&since=N          live subscription with optional type filter and catch-up
GET  /health

No model calls here. The dashboard and every service subscribe to this.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GT_DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = DATA_DIR / "events.jsonl"

app = FastAPI(title="Ground Truth event hub")
ring: Deque[Dict[str, Any]] = deque(maxlen=20000)
subscribers: Set["Sub"] = set()
seq = 0
_lock = asyncio.Lock()


class Sub:
    def __init__(self, ws: WebSocket, types: Optional[Set[str]]):
        self.ws = ws
        self.types = types
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    def wants(self, ev: Dict[str, Any]) -> bool:
        return not self.types or ev["type"] in self.types


@app.on_event("startup")
async def _load() -> None:
    global seq
    if LOG_PATH.exists():
        with LOG_PATH.open() as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    ring.append(ev)
                    seq = max(seq, int(ev.get("seq", 0)))
                except json.JSONDecodeError:
                    continue


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "seq": seq, "subscribers": len(subscribers), "ring": len(ring)}


@app.post("/events")
async def post_event(ev: Dict[str, Any]) -> JSONResponse:
    global seq
    if "type" not in ev:
        return JSONResponse({"error": "type required"}, status_code=400)
    async with _lock:
        seq += 1
        ev["seq"] = seq
        ev["recv_ms"] = int(time.time() * 1000)  # ms-resolution receive time for latency measurement
        ring.append(ev)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
    for s in list(subscribers):
        if s.wants(ev):
            try:
                s.queue.put_nowait(ev)
            except asyncio.QueueFull:
                pass  # slow consumer: drop for them, never block producers
    return JSONResponse({"ok": True, "seq": seq})


@app.get("/events")
async def get_events(since: int = Query(0), types: Optional[str] = None, limit: int = Query(500, le=5000)) -> List[Dict[str, Any]]:
    wanted = set(types.split(",")) if types else None
    out = [e for e in ring if e.get("seq", 0) > since and (not wanted or e["type"] in wanted)]
    return out[-limit:]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, types: Optional[str] = None, since: Optional[int] = None) -> None:
    await ws.accept()
    sub = Sub(ws, set(types.split(",")) if types else None)
    subscribers.add(sub)
    try:
        if since is not None:
            for e in list(ring):
                if e.get("seq", 0) > since and sub.wants(e):
                    await ws.send_text(json.dumps(e))
        while True:
            ev = await sub.queue.get()
            await ws.send_text(json.dumps(ev))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        subscribers.discard(sub)
