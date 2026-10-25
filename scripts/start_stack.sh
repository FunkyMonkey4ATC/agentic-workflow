#!/usr/bin/env bash
# One start script for the GB10 stack. Each component runs in its own tmux session so it can be
# inspected (`tmux attach -t <name>`) and restarted independently. Idempotent: skips sessions
# that already exist. Usage: scripts/start_stack.sh [models|services|all]   (default: all)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/home/dell/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOGS="$HF_HOME/logs"; mkdir -p "$LOGS" data/clips data/outbox
PY="$ROOT/.venv/bin/python"

WHISPER_BIN="${WHISPER_BIN:-/home/dell/whisper.cpp/build/bin/whisper-server}"
WHISPER_MODEL="${WHISPER_MODEL:-$HF_HOME/ggml/whisper-atc/ggml-model.bin}"
LLAMA_BIN="${LLAMA_BIN:-/home/dell/llama.cpp/build/bin/llama-server}"
FAST_GGUF="${FAST_GGUF:-$(ls "$HF_HOME"/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/*/Qwen3-4B-Instruct-2507-Q8_0.gguf 2>/dev/null | head -1 || true)}"
REASON_GGUF="${REASON_GGUF:-$(ls "$HF_HOME"/hub/models--*gpt-oss-20b*/snapshots/*/*.gguf 2>/dev/null | head -1 || true)}"

session() { # name, command
  if tmux has-session -t "$1" 2>/dev/null; then echo "  [skip] $1 already running"; return; fi
  tmux new -d -s "$1" "cd $ROOT; $2 2>&1 | tee -a $LOGS/$1.log"; echo "  [start] $1"
}

start_models() {
  echo "models:"
  session whisper-server "$WHISPER_BIN -m $WHISPER_MODEL --host 127.0.0.1 --port 9000 -t 4 -l en"
  if [[ -n "$FAST_GGUF" ]]; then
    # fast: Qwen3-4B-Instruct-2507 Q8_0. 4 parallel slots x 4k ctx. ~6 GB incl. KV.
    session llama-fast "$LLAMA_BIN -m $FAST_GGUF --alias qwen3-4b-instruct-2507 --host 127.0.0.1 --port 8002 -ngl 99 -c 16384 -np 4 -fa on --jinja --temp 0 -t 6"
  else echo "  [warn] fast GGUF not found yet; llama-fast not started"; fi
  if [[ -n "$REASON_GGUF" ]]; then
    session llama-reason "$LLAMA_BIN -m $REASON_GGUF --alias gpt-oss-20b --host 127.0.0.1 --port 8001 -ngl 99 -c 16384 -np 1 -fa on --jinja -t 6"
  else echo "  [info] reason GGUF absent; capability 'reason' falls back to 'fast' (by design until S1)"; fi
}

start_services() {
  echo "services:"
  session hub      "$PY -m uvicorn services.hub.app:app --host 127.0.0.1 --port 7000 --log-level warning"
  session gateway  "$PY -m uvicorn services.gateway.app:app --host 0.0.0.0 --port 4000 --log-level warning"
  session asr      "$PY -m uvicorn services.asr.service:app --host 127.0.0.1 --port 9100 --log-level warning"
  session parser   "$PY -m services.parser.service"
  [[ -f services/decision/service.py ]] && session decision "$PY -m services.decision.service" || true
  [[ -f services/agent/service.py ]]    && session agent    "$PY -m uvicorn services.agent.service:app --host 127.0.0.1 --port 9200 --log-level warning" || true
}

case "${1:-all}" in
  models) start_models ;;
  services) start_services ;;
  all) start_models; start_services ;;
  stop) for s in agent decision parser asr gateway hub llama-reason llama-fast whisper-server; do tmux kill-session -t "$s" 2>/dev/null && echo "  [stop] $s" || true; done ;;
  *) echo "usage: $0 [models|services|all|stop]"; exit 1 ;;
esac
echo "done. health: curl -s localhost:4000/health | python3 -m json.tool"
