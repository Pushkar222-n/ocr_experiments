#!/usr/bin/env bash
# The vLLM server that holds chandra's weights. Start it once; leave it up.
#
#   ./serve.sh            start (idempotent — a live server is left alone)
#   ./serve.sh --stop     stop it and give the VRAM back
#   ./serve.sh --status   is it up?
#   ./serve.sh --logs     follow the server log
#
# It is slow to start (~7 min cold: weights download + torch.compile + CUDA-graph capture;
# ~2-3 min once the weights are cached on disk) and hot forever after. That asymmetry is the
# whole reason this is a separate, resident process: run.sh attaches to it in milliseconds.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${VLLM_PORT:-8200}"
MODEL="${MODEL_CHECKPOINT:-datalab-to/chandra-ocr-2}"
SERVED_NAME="${VLLM_MODEL_NAME:-chandra}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18000}"
LOG="logs/vllm_server.log"
PIDFILE="work/vllm.pid"

mkdir -p logs work

is_up() { curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; }

case "${1:-start}" in
  --status)
    is_up && echo "vLLM is UP on :${PORT}" || { echo "vLLM is DOWN"; exit 1; }
    exit 0 ;;
  --logs)
    exec tail -f "$LOG" ;;
  --stop)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"
      echo "stopped — VRAM released"
    else
      pkill -f "vllm serve ${MODEL}" 2>/dev/null && echo "stopped" || echo "was not running"
    fi
    exit 0 ;;
esac

if is_up; then
  echo "vLLM already up on :${PORT} — nothing to do"
  exit 0
fi

# uv sync is skipped when the venv exists: with a cold uv cache a re-sync re-downloads ~16 GB.
if [[ ! -x .venv/bin/vllm || "${FORCE_SYNC:-0}" == "1" ]]; then
  echo "building venv (~16 GB, once per pod)..."
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}" uv sync
fi

echo "starting vLLM (this takes ~7 min cold, ~2-3 min with weights cached)..."
# Flag notes — every one of these is measured, see README:
#   --max-model-len       REQUIRED on a 46 GB card. vLLM sizes the KV cache to ~229k tokens
#                         while the model declares a 262,144-token context, and refuses to
#                         start when the context exceeds the cache.
#   --mm-processor-kwargs matches the client's own resize cap (3072x2048 = 6,291,456 px).
#   NOT set: --enable-prefix-caching (chandra puts the per-page-unique image BEFORE the
#            constant prompt, so no page can ever share a prefix with another), and
#            --max-num-seqs / --max-num-batched-tokens (the vendor's values are H100-scaled
#            and LOWER than vLLM's defaults; they would split one page's ~6.3k-token prefill
#            across two scheduler steps).
nohup .venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Cold start is dominated by weight download (~10 GB the first time) + torch.compile +
# CUDA-graph capture; on a slow volume or first pull this can run well past 15 min. Wait
# generously — override with SERVE_WAIT_SECS. We watch the pid and the log so a genuinely
# dead server fails fast rather than burning the whole budget.
WAIT_SECS="${SERVE_WAIT_SECS:-2400}"   # 40 min default
printf "waiting for the server (up to %d min; set SERVE_WAIT_SECS to change)" $((WAIT_SECS / 60))
deadline=$(( $(date +%s) + WAIT_SECS ))
while (( $(date +%s) < deadline )); do
  if is_up; then
    echo; echo "vLLM is UP on :${PORT}  (log: $LOG)"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo; echo "server process exited during startup — last lines of $LOG:"; tail -30 "$LOG"; exit 1
  fi
  printf "."; sleep 5
done
echo; echo "timed out after ${WAIT_SECS}s. The server may still be loading — check: ./serve.sh --logs"
exit 1
