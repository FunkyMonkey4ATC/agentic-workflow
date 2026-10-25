"""ASR service on :9100 — streaming-style transcription with provisional/final segments.

Chunked Whisper is not native streaming, so this service owns utterance boundaries:
  audio -> 16 kHz mono -> energy VAD -> utterance state machine
  while speaking: every PROVISIONAL_EVERY_S re-transcribe the utterance so far -> final=false, rev++
  on silence >= END_SILENCE_S (or MAX_UTTERANCE_S): transcribe once more -> final=true
Only final segments are meant to update authoritative clearance state (parser listens to
final; keyword detector listens to both and emits candidate alerts).

Endpoints
  POST /transcribe    multipart wav/mp3/flac -> segments (also published to the bus)
  POST /replay        {"path": "...", "speed": 1.0, "start_s": 0, "publish": true} -> starts a replay task
  POST /replay/stop
  GET  /health  GET /metrics

Transcription goes through the gateway capability `asr` (P1), so the backend can be swapped.
Set ASR_DIRECT_URL to hit whisper-server directly if the gateway is down.
"""
from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile

from services.common.bus import Bus
from services.common.events import make_event, now_iso
from services.keyword.detector import detect

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GT_DATA_DIR", ROOT / "data"))
CLIP_DIR = DATA_DIR / "clips"
CLIP_DIR.mkdir(parents=True, exist_ok=True)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:4000")
ASR_DIRECT_URL = os.environ.get("ASR_DIRECT_URL")  # e.g. http://127.0.0.1:9000 (whisper-server)
SR = 16000
FRAME_S = 0.1
PROVISIONAL_EVERY_S = float(os.environ.get("ASR_PROVISIONAL_EVERY_S", "1.5"))
END_SILENCE_S = float(os.environ.get("ASR_END_SILENCE_S", "0.7"))
MIN_SPEECH_S = 0.3
MAX_UTTERANCE_S = 20.0
PRE_ROLL_S = 0.3
ATC_PROMPT = os.environ.get("ASR_PROMPT", "LaGuardia tower, Jazz six forty six, Truck one, runway four, taxiway delta, cleared to land, cross runway, hold short, line up and wait.")

app = FastAPI(title="Ground Truth ASR service")
bus = Bus()
http = httpx.AsyncClient(timeout=60.0)
metrics: Dict[str, Any] = {"utterances": 0, "provisional": 0, "final": 0, "asr_ms": [], "final_latency_ms": [],
                           "provisional_latency_ms": [], "errors": 0}
replay_task: Optional[asyncio.Task] = None
clip_counter = 0


# --- audio helpers -----------------------------------------------------------
def load_audio(raw: bytes) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        n_out = int(round(len(mono) * SR / sr))
        mono = np.interp(np.linspace(0, len(mono) - 1, n_out), np.arange(len(mono)), mono).astype(np.float32)
    return mono


def to_wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def transcribe_wav(wav: bytes, priority: str = "P1") -> Dict[str, Any]:
    t0 = time.monotonic()
    if ASR_DIRECT_URL:
        files = {"file": ("u.wav", wav, "audio/wav")}
        form = {"response_format": "verbose_json", "temperature": "0.0", "temperature_inc": "0.2",
                "language": "en", "prompt": ATC_PROMPT}
        r = await http.post(ASR_DIRECT_URL.rstrip("/") + "/inference", files=files, data=form)
    else:
        files = {"file": ("u.wav", wav, "audio/wav")}
        r = await http.post(GATEWAY_URL + "/v1/audio/transcriptions", files=files,
                            data={"model": "asr", "response_format": "verbose_json"},
                            headers={"X-Priority": priority})
    r.raise_for_status()
    out = r.json()
    out["_asr_ms"] = round((time.monotonic() - t0) * 1000)
    metrics["asr_ms"].append(out["_asr_ms"])
    return out


def clean_text(res: Dict[str, Any]) -> str:
    text = (res.get("text") or "").strip()
    if not text and res.get("segments"):
        text = " ".join(s.get("text", "").strip() for s in res["segments"]).strip()
    return " ".join(text.split())


# --- utterance emission --------------------------------------------------------
async def emit_segment(utt_id: str, rev: int, final: bool, t_start: float, t_end: float, text: str,
                       audio_path: Optional[str], audio_end_wall: float, asr_ms: int) -> None:
    latency_ms = round((time.monotonic() - audio_end_wall) * 1000)
    (metrics["final_latency_ms"] if final else metrics["provisional_latency_ms"]).append(latency_ms)
    metrics["final" if final else "provisional"] += 1
    ev = make_event("transcript.segment", "asr", {
        "utterance_id": utt_id, "rev": rev, "final": final,
        "t_start": round(t_start, 2), "t_end": round(t_end, 2), "text": text,
        "audio": audio_path, "asr_ms": asr_ms, "latency_ms": latency_ms,
    })
    await bus.publish(ev)
    kw = detect(text)
    if kw["is_candidate"]:
        await bus.publish(make_event("candidate", "keyword", {
            "utterance_id": utt_id, "rev": rev, "final": final, "text": text,
            "phrases": kw["phrases"], "severity_hint": kw["severity_hint"],
            "latency_ms": round((time.monotonic() - audio_end_wall) * 1000),
        }))


class UtteranceMachine:
    """Feeds 100 ms frames; drives provisional/final transcriptions."""

    def __init__(self, publish: bool = True, clock_offset_s: float = 0.0):
        self.publish = publish
        self.clock_offset_s = clock_offset_s
        self.noise_floor = 0.004
        self.reset()

    def reset(self) -> None:
        self.buf: List[np.ndarray] = []
        self.pre: List[np.ndarray] = []
        self.speaking = False
        self.speech_s = 0.0
        self.silence_s = 0.0
        self.since_prov_s = 0.0
        self.utt_id = ""
        self.rev = 0
        self.t_start = 0.0
        self.pending: Optional[asyncio.Task] = None

    async def feed(self, frame: np.ndarray, t_now_s: float) -> None:
        rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-9)
        if not self.speaking:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
        is_speech = rms > max(0.012, self.noise_floor * 3.5)
        if not self.speaking:
            self.pre.append(frame)
            self.pre = self.pre[-int(PRE_ROLL_S / FRAME_S):]
            if is_speech:
                self.speaking = True
                self.utt_id = uuid.uuid4().hex[:10]
                self.rev = 0
                self.t_start = t_now_s - len(self.pre) * FRAME_S
                self.buf = list(self.pre)
                self.speech_s = FRAME_S
                self.silence_s = 0.0
                self.since_prov_s = 0.0
            return
        self.buf.append(frame)
        self.since_prov_s += FRAME_S
        if is_speech:
            self.speech_s += FRAME_S
            self.silence_s = 0.0
        else:
            self.silence_s += FRAME_S
        dur = len(self.buf) * FRAME_S
        if self.silence_s >= END_SILENCE_S or dur >= MAX_UTTERANCE_S:
            if self.speech_s >= MIN_SPEECH_S:
                await self._finalize(t_now_s)
            self.reset()
        elif self.since_prov_s >= PROVISIONAL_EVERY_S and self.speech_s >= MIN_SPEECH_S:
            self.since_prov_s = 0.0
            if self.pending is None or self.pending.done():
                self.pending = asyncio.create_task(self._provisional(t_now_s))

    async def _provisional(self, t_now_s: float) -> None:
        self.rev += 1
        rev = self.rev
        audio = np.concatenate(self.buf)
        audio_end_wall = time.monotonic()  # wall time of the latest audio frame we have
        try:
            res = await transcribe_wav(to_wav_bytes(audio), "P1")
        except Exception:
            metrics["errors"] += 1
            return
        text = clean_text(res)
        if text and self.publish:
            await emit_segment(self.utt_id, rev, False, self.t_start + self.clock_offset_s,
                               t_now_s + self.clock_offset_s, text, None, audio_end_wall, res["_asr_ms"])

    async def _finalize(self, t_now_s: float) -> None:
        global clip_counter
        if self.pending and not self.pending.done():
            self.pending.cancel()
        self.rev += 1
        audio = np.concatenate(self.buf)
        clip_counter += 1
        clip_path = CLIP_DIR / f"clip_{clip_counter:04d}.wav"
        wav = to_wav_bytes(audio)
        clip_path.write_bytes(wav)
        audio_end_wall = time.monotonic()
        try:
            res = await transcribe_wav(wav, "P1")
        except Exception:
            metrics["errors"] += 1
            return
        metrics["utterances"] += 1
        text = clean_text(res)
        if self.publish and text:
            await emit_segment(self.utt_id, self.rev, True, self.t_start + self.clock_offset_s,
                               t_now_s - self.silence_s + self.clock_offset_s, text, str(clip_path),
                               audio_end_wall, res["_asr_ms"])


async def stream_array(x: np.ndarray, speed: float, publish: bool, start_s: float = 0.0) -> Dict[str, Any]:
    """Feed an audio array through the machine, paced to real time (speed=1) or faster."""
    m = UtteranceMachine(publish=publish, clock_offset_s=start_s)
    n = int(SR * FRAME_S)
    t0 = time.monotonic()
    for i in range(0, len(x) - n + 1, n):
        t_now = start_s + (i + n) / SR
        await m.feed(x[i:i + n], t_now)
        if speed > 0:
            target = t0 + (i + n) / SR / speed
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)
    if m.speaking and m.speech_s >= MIN_SPEECH_S:
        await m._finalize(start_s + len(x) / SR)
    return {"seconds": len(x) / SR, "utterances": metrics["utterances"]}


# --- HTTP ----------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "replay_running": bool(replay_task and not replay_task.done()), "mode": "direct" if ASR_DIRECT_URL else "gateway"}


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    def pct(xs: List[int]) -> Dict[str, Any]:
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": len(xs), "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)), "max": int(a.max())}
    return {"utterances": metrics["utterances"], "provisional": metrics["provisional"], "final": metrics["final"],
            "errors": metrics["errors"], "asr_ms": pct(metrics["asr_ms"]),
            "audio_end_to_provisional_ms": pct(metrics["provisional_latency_ms"]),
            "audio_end_to_final_ms": pct(metrics["final_latency_ms"])}


@app.post("/transcribe")
async def transcribe_upload(file: UploadFile = File(...), publish: bool = True, stream: bool = True) -> Dict[str, Any]:
    """Whole clip at once (stream=false) or through the utterance machine as fast as possible (stream=true)."""
    raw = await file.read()
    x = load_audio(raw)
    if not stream:
        res = await transcribe_wav(to_wav_bytes(x), "P2")
        text = clean_text(res)
        if publish and text:
            global clip_counter
            clip_counter += 1
            p = CLIP_DIR / f"clip_{clip_counter:04d}.wav"
            p.write_bytes(to_wav_bytes(x))
            await emit_segment(uuid.uuid4().hex[:10], 1, True, 0.0, len(x) / SR, text, str(p), time.monotonic(), res["_asr_ms"])
        return {"text": text, "segments": res.get("segments", []), "asr_ms": res["_asr_ms"]}
    before = metrics["final"]
    out = await stream_array(x, speed=0, publish=publish)
    out["final_segments_emitted"] = metrics["final"] - before
    return out


@app.post("/replay")
async def replay(body: Dict[str, Any]) -> Dict[str, Any]:
    global replay_task
    if replay_task and not replay_task.done():
        return {"ok": False, "error": "replay already running"}
    path = Path(body["path"])
    x = load_audio(path.read_bytes())
    speed = float(body.get("speed", 1.0))
    start_s = float(body.get("start_s", 0.0))
    if start_s > 0:
        x = x[int(start_s * SR):]
    replay_task = asyncio.create_task(stream_array(x, speed, bool(body.get("publish", True)), start_s))
    await bus.publish(make_event("replay.started", "asr", {"path": str(path), "speed": speed, "seconds": len(x) / SR}))
    return {"ok": True, "seconds": len(x) / SR, "speed": speed}


@app.post("/inject")
async def inject(body: Dict[str, Any]) -> Dict[str, Any]:
    """Typed-transcript backend (FMK-6 fallback): publish a final segment without audio.
    body: {"text": "...", "t_start": 0.0, "t_end": 3.0, "speaker": "tower", "audio": null}"""
    text = " ".join(str(body.get("text", "")).split())
    if not text:
        return {"ok": False, "error": "text required"}
    utt = body.get("utterance_id") or uuid.uuid4().hex[:10]
    await emit_segment(utt, 1, True, float(body.get("t_start", 0.0)), float(body.get("t_end", 0.0)),
                       text, body.get("audio"), time.monotonic(), 0)
    return {"ok": True, "utterance_id": utt}


@app.post("/replay/stop")
async def replay_stop() -> Dict[str, Any]:
    if replay_task and not replay_task.done():
        replay_task.cancel()
        return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False}
