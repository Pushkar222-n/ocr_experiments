#!/usr/bin/env bash
# Usage: ./run.sh <model> [args passed to the model's run.py]
# Models: got_ocr lightonocr unlimited_ocr paddleocr_vl glm_ocr mineru surya chandra
# Handles: uv env sync, vLLM server lifecycle for surya/chandra (pip vllm,
# no docker — works inside RunPod pods), then runs the model over sample_set.
# Re-running the same command resumes from checkpoints.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: ./run.sh <model> [args]}"
shift || true
PROJ="models/$MODEL"
[ -d "$PROJ" ] || { echo "unknown model '$MODEL' (see models/)"; exit 1; }

# keep HF downloads on the persistent volume if running from /workspace
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
mkdir -p work

echo "== syncing env for $MODEL =="
uv sync --project "$PROJ"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

wait_server() { # $1=port
    echo "waiting for vLLM on :$1 (log: work/${MODEL}_vllm.log) ..."
    until curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "vLLM died, see work/${MODEL}_vllm.log"; exit 1; }
        sleep 5
    done
    echo "vLLM up."
}

case "$MODEL" in
    surya)
        uv run --project "$PROJ" vllm serve datalab-to/surya-ocr-2 \
            --port 8100 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
            > "work/${MODEL}_vllm.log" 2>&1 &
        SERVER_PID=$!
        wait_server 8100
        export SURYA_INFERENCE_BACKEND=vllm
        export SURYA_INFERENCE_URL=http://127.0.0.1:8100/v1
        ;;
    chandra)
        uv run --project "$PROJ" vllm serve datalab-to/chandra-ocr-2 \
            --served-model-name chandra --port 8200 \
            --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
            --limit-mm-per-prompt '{"image": 1}' \
            > "work/${MODEL}_vllm.log" 2>&1 &
        SERVER_PID=$!
        wait_server 8200
        export VLLM_API_BASE=http://127.0.0.1:8200/v1
        export VLLM_MODEL_NAME=chandra
        export MODEL_CHECKPOINT=datalab-to/chandra-ocr-2
        ;;
esac

uv run --project "$PROJ" python "$PROJ/run.py" "$@"
