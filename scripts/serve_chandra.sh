#!/usr/bin/env bash
# Start a chandra vLLM server that OUTLIVES any single run, and leave it up.
#
# Why: loading chandra costs ~6 minutes (weights + torch.compile + CUDA-graph capture), and
# `run.sh chandra` otherwise pays that on every invocation -- a smoke test followed by a
# real run pays it twice, and an A/B of two arms pays it four times. With a resident server,
# `run.sh chandra` detects the open port, attaches, and starts decoding immediately.
#
#   scripts/serve_chandra.sh            # start it (tuned flags, the default)
#   CHANDRA_TUNED=0 scripts/serve_chandra.sh   # start the control arm instead
#   ./run.sh chandra --out-dir <x>      # attaches to whatever is up
#   scripts/serve_chandra.sh --stop     # kill it and free the VRAM
#
# The server holds ~39 GB of the A40 for as long as it lives. Stop it before running any
# other model, or they will not both fit.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8200
VENV="models/chandra/.venv/bin"
LOG="work/chandra_vllm.log"

if [ "${1:-}" = "--stop" ]; then
    pkill -f '[v]llm serve datalab-to/chandra-ocr-2' && echo "stopped." || echo "nothing to stop."
    exit 0
fi

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "already serving on :$PORT -- nothing to do (--stop to kill it)."
    exit 0
fi

[ -x "$VENV/vllm" ] || { echo "no chandra venv; run: uv sync --project models/chandra"; exit 1; }
mkdir -p work

# Same flags run.sh would use, and for the same reasons -- see the chandra case in run.sh
# for why --max-model-len is a REQUIREMENT here and not a tuning nicety: without it vLLM
# sizes for the config's 262144-token context, which does not fit this card's KV cache
# (229,152 tokens measured), and refuses to start.
TUNED_ARGS=()
if [ "${CHANDRA_TUNED:-1}" = "1" ]; then
    TUNED_ARGS=(--max-model-len 18000
                --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}')
    echo "== starting chandra vLLM :$PORT [TUNED: ${TUNED_ARGS[*]}] =="
else
    echo "== starting chandra vLLM :$PORT [CONTROL: vLLM default serving flags] =="
fi

nohup "$VENV/vllm" serve datalab-to/chandra-ocr-2 \
    --served-model-name chandra --port "$PORT" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --limit-mm-per-prompt '{"image": 1}' \
    "${TUNED_ARGS[@]+"${TUNED_ARGS[@]}"}" \
    > "$LOG" 2>&1 &

echo "pid $! -- log: $LOG"
echo -n "waiting for it to come up (~6 min: weights, compile, cudagraph capture) "
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    pgrep -f '[v]llm serve datalab-to/chandra-ocr-2' >/dev/null || {
        echo; echo "vLLM died. Last lines of $LOG:"; tail -20 "$LOG"; exit 1; }
    echo -n "."
    sleep 10
done
echo
echo "vLLM up on :$PORT. It will stay up until you run: scripts/serve_chandra.sh --stop"
grep -E "GPU KV cache size|Maximum concurrency" "$LOG" | tail -2 || true
