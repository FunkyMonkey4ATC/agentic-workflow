"""Capability gateway on :4000.

OpenAI-compatible surface. Callers pass a *capability* as `model` (fast | reason | asr),
never a Hugging Face id or a port. Resolution + health live here; scheduling lives in
scheduler.py. If the primary backend is down, the fallback backend is used; if nothing
is available the caller gets HTTP 503 with {"degraded": true} and must render its template.

Headers:
  X-Priority:      P1 | P2 | P3   (default P2)
  X-Coalesce-Key:  arbitrary string; for `reason`, a newer request replaces a queued one
  X-Timeout-S:     per-request queue+run timeout
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from .scheduler import CapabilityScheduler, QueueFull, Superseded

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("CAPABILITIES_CONFIG", ROOT / "config" / "capabilities.yaml"))

app = FastAPI(title="Ground Truth capability gateway")
cfg: Dict[str, Any] = {}
schedulers: Dict[str, CapabilityScheduler] = {}
health: Dict[str, Dict[str, Any]] = {}
client: Optional[httpx.AsyncClient] = None
metrics = {"requests": 0, "fallbacks": 0, "degraded": 0, "by_capability": {}}


def load_config() -> None:
    global cfg, schedulers
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    schedulers = {}
    for cap, s in (cfg.get("scheduler") or {}).items():
        schedulers[cap] = CapabilityScheduler(
            cap, s.get("max_concurrent", 1), s.get("queue_limits", {}),
            s.get("reserved_for_p1", 0), s.get("coalesce", False), s.get("default_timeout_s", 30))


async def probe(name: str, b: Dict[str, Any]) -> Dict[str, Any]:
    url = b["url"].rstrip("/")
    path = "/health" if b["kind"] == "openai" else "/"
    t0 = time.monotonic()
    try:
        r = await client.get(url + path, timeout=2.0)
        ok = r.status_code < 500
        return {"ok": ok, "status": r.status_code, "ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": type(e).__name__}


async def health_loop() -> None:
    while True:
        for name, b in cfg["backends"].items():
            health[name] = await probe(name, b)
        await asyncio.sleep(3)


@app.on_event("startup")
async def _startup() -> None:
    global client
    load_config()
    client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=3.0))
    for name, b in cfg["backends"].items():
        health[name] = await probe(name, b)
    asyncio.create_task(health_loop())


def resolve(capability: str) -> tuple[str, Dict[str, Any], bool]:
    """Return (backend_name, backend_cfg, used_fallback). Raises 503 if nothing is up."""
    c = cfg["capabilities"].get(capability)
    if not c:
        raise HTTPException(404, f"unknown capability '{capability}'")
    for i, name in enumerate([c.get("backend"), c.get("fallback")]):
        if name and health.get(name, {}).get("ok"):
            return name, cfg["backends"][name], i == 1
    metrics["degraded"] += 1
    raise HTTPException(503, detail={"degraded": True, "capability": capability,
                                     "reason": "no healthy backend", "health": health})


@app.get("/health")
async def get_health() -> Dict[str, Any]:
    caps = {}
    for cap, c in cfg["capabilities"].items():
        primary_ok = health.get(c.get("backend"), {}).get("ok", False)
        fb = c.get("fallback")
        caps[cap] = {"backend": c.get("backend"), "primary_ok": primary_ok,
                     "fallback": fb, "fallback_ok": health.get(fb, {}).get("ok", False) if fb else None,
                     "available": primary_ok or (fb is not None and health.get(fb, {}).get("ok", False))}
    return {"backends": health, "capabilities": caps,
            "scheduler": {k: s.snapshot() for k, s in schedulers.items()}, "metrics": metrics}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": c, "object": "model", "owned_by": "ground-truth"} for c in cfg["capabilities"]]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request,
                           x_priority: str = Header("P2"),
                           x_coalesce_key: Optional[str] = Header(None),
                           x_timeout_s: Optional[float] = Header(None)) -> Any:
    body = await request.json()
    capability = body.get("model", "fast")
    name, backend, used_fallback = resolve(capability)
    if backend["kind"] != "openai":
        raise HTTPException(400, f"capability '{capability}' is not a chat backend")
    if used_fallback:
        metrics["fallbacks"] += 1
    metrics["requests"] += 1
    metrics["by_capability"][capability] = metrics["by_capability"].get(capability, 0) + 1
    body["model"] = backend.get("model", capability)
    if body.get("stream"):
        body["stream"] = False  # keep the gateway simple; agents don't need token streaming

    async def call() -> Any:
        r = await client.post(backend["url"].rstrip("/") + "/v1/chat/completions", json=body)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:500])
        out = r.json()
        out["gateway"] = {"capability": capability, "backend": name, "fallback": used_fallback}
        return out

    sched = schedulers.get(capability)
    t0 = time.monotonic()
    try:
        result = await (sched.run(x_priority, call, x_coalesce_key, x_timeout_s) if sched else call())
    except QueueFull as e:
        raise HTTPException(429, {"degraded": True, "reason": str(e)})
    except Superseded as e:
        raise HTTPException(409, {"superseded": True, "reason": str(e)})
    except asyncio.TimeoutError:
        raise HTTPException(504, {"degraded": True, "reason": f"{capability} queue timeout"})
    result["gateway"]["total_ms"] = round((time.monotonic() - t0) * 1000)
    return result


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...),
                         model: str = Form("asr"),
                         response_format: str = Form("verbose_json"),
                         x_priority: str = Header("P1")) -> Any:
    """ASR adapter: OpenAI-style upload in, whisper.cpp server out (segments with timestamps)."""
    name, backend, _ = resolve(model)
    if backend["kind"] != "whispercpp":
        raise HTTPException(400, f"capability '{model}' is not an ASR backend")
    data = await file.read()

    async def call() -> Any:
        files = {"file": (file.filename or "audio.wav", data, "audio/wav")}
        form = {"response_format": response_format, "temperature": "0.0", "temperature_inc": "0.2"}
        r = await client.post(backend["url"].rstrip("/") + "/inference", files=files, data=form, timeout=60.0)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:500])
        return r.json()

    sched = schedulers.get(model)
    try:
        return await (sched.run(x_priority, call) if sched else call())
    except QueueFull as e:
        raise HTTPException(429, {"degraded": True, "reason": str(e)})
    except asyncio.TimeoutError:
        raise HTTPException(504, {"degraded": True, "reason": "asr queue timeout"})


@app.post("/admin/reload")
async def reload_config() -> Dict[str, Any]:
    load_config()
    return {"ok": True, "capabilities": list(cfg["capabilities"])}


@app.exception_handler(HTTPException)
async def _http_exc(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=detail)
