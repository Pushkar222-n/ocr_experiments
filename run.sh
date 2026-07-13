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
[ -d "$PROJ" ] && [ -f "$PROJ/run.py" ] || { echo "unknown model '$MODEL' (see models/)"; exit 1; }

# keep HF downloads on the persistent volume if running from /workspace
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
mkdir -p work

# Mirror everything (this script, uv, and the model's stdout+stderr) to a per-model
# run log while still printing to the terminal. Appended, not truncated, so a resumed
# run keeps the history of the attempts that got it there.
RUN_LOG="work/${MODEL}_run.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "===== $(date -Is) :: ./run.sh $MODEL $* ====="

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
    paddleocr_vl)
        # PaddleX caches weights under ~/.paddlex (paddlex/utils/cache.py:
        # CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", ~/.paddlex)). $HOME is the
        # container's 30 GB ephemeral overlay, not /workspace — weights would be lost on
        # pod restart and reclaim.sh could never find them. Keep them on the volume.
        export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/workspace/.cache/paddlex}"
        mkdir -p "$PADDLE_PDX_CACHE_HOME"
        echo "== paddlex cache: $PADDLE_PDX_CACHE_HOME =="
        ;;
    surya)
        # served from models/vllm_server so surya's own venv never carries vLLM's
        # ~14 GB of wheels (disk is the binding constraint on this pod)
        echo "== syncing vllm server env =="
        uv sync --project models/vllm_server
        uv run --project models/vllm_server vllm serve datalab-to/surya-ocr-2 \
            --port 8100 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
            > "work/${MODEL}_vllm.log" 2>&1 &
        SERVER_PID=$!
        wait_server 8100
        export SURYA_INFERENCE_BACKEND=vllm
        export SURYA_INFERENCE_URL=http://127.0.0.1:8100/v1
        ;;
    glm_ocr)
        # glmocr[selfhosted] ships no vllm: it is an HTTP *client* that POSTs to
        # {ocr_api_host}:{ocr_api_port}/v1/chat/completions. Only PP-DocLayout-V3 runs
        # in-process (on cpu, so the GPU is left to the decoder). Serve the decoder from
        # models/vllm_server, same as surya, so glm_ocr's venv stays vllm-free.
        echo "== syncing vllm server env =="
        uv sync --project models/vllm_server
        uv run --project models/vllm_server vllm serve zai-org/GLM-OCR \
            --served-model-name glm-ocr --port 8300 \
            --gpu-memory-utilization "${GPU_MEM_UTIL:-0.8}" \
            --limit-mm-per-prompt '{"image": 1}' \
            > "work/${MODEL}_vllm.log" 2>&1 &
        SERVER_PID=$!
        wait_server 8300
        export GLM_OCR_HOST=127.0.0.1
        export GLM_OCR_PORT=8300
        export GLM_OCR_MODEL=glm-ocr
        export LAYOUT_DEVICE="${LAYOUT_DEVICE:-cpu}"
        ;;
    mineru)
        # Default: no server, mineru runs the VLM in-process on the transformers engine.
        # MINERU_VLLM=1 instead serves the same weights from models/vllm_server and points
        # the adapter at them, which is the engine MinerU itself recommends -- the
        # transformers path measured 72.4 s/page over the 68-page set, 3-9x slower than
        # every other model here, purely because of the engine. Same weights either way, so
        # only the timings should move; the markdown should not.
        if [ "${MINERU_VLLM:-0}" = "1" ]; then
            echo "== syncing vllm server env =="
            uv sync --project models/vllm_server
            # Serve a patched copy, not the hub repo: vllm 0.19.1 reads tie_word_embeddings
            # from the top level of the config, MinerU2.5 only sets it in text_config, and
            # the mismatch kills weight loading on the missing (because tied) lm_head.
            # scripts/mineru_vllm_model.py explains it in full and prints the local path.
            MINERU_MODEL_DIR=$(uv run --project models/vllm_server python scripts/mineru_vllm_model.py)
            echo "== serving $MINERU_MODEL_DIR =="
            uv run --project models/vllm_server vllm serve "$MINERU_MODEL_DIR" \
                --served-model-name mineru --port 8400 \
                --gpu-memory-utilization "${GPU_MEM_UTIL:-0.7}" \
                --limit-mm-per-prompt '{"image": 1}' \
                > "work/${MODEL}_vllm.log" 2>&1 &
            SERVER_PID=$!
            wait_server 8400
            # base url, no /v1 -- mineru appends /v1/chat/completions itself. Copying the
            # ".../v1" form the surya/chandra cases use would produce /v1/v1/chat/completions.
            export MINERU_URL=http://127.0.0.1:8400
        fi
        ;;
    lightonocr)
        # Default: in-process transformers. LIGHTON_VLLM=1 serves the same weights instead.
        #
        # Served from models/chandra's venv, not models/vllm_server: chandra already carries
        # vllm 0.19.1 (identical version) and its venv is still resident, so this reuses
        # ~15 GB that is already on disk instead of re-syncing vllm_server from scratch.
        # If chandra's venv has been reclaimed, swap the --project back to models/vllm_server
        # (its uv.lock is committed) -- the serve command is otherwise unchanged.
        if [ "${LIGHTON_VLLM:-0}" = "1" ]; then
            VLLM_PROJ="${VLLM_PROJ:-models/chandra}"
            echo "== serving lightonocr via $VLLM_PROJ's vllm =="
            uv run --project "$VLLM_PROJ" vllm serve lightonai/LightOnOCR-2-1B \
                --served-model-name lightonocr --port 8500 \
                --gpu-memory-utilization "${GPU_MEM_UTIL:-0.8}" \
                --limit-mm-per-prompt '{"image": 1}' \
                > "work/${MODEL}_vllm.log" 2>&1 &
            SERVER_PID=$!
            wait_server 8500
            # WITH /v1 -- the adapter appends only /chat/completions (unlike mineru, whose
            # client appends the whole /v1/chat/completions path itself).
            export LIGHTON_URL=http://127.0.0.1:8500/v1
            export LIGHTON_MODEL=lightonocr
        fi
        ;;
    chandra)
        # --max-model-len and --mm-processor-kwargs come from datalab's own launcher
        # (chandra/scripts/vllm.py, which run.sh can't use directly -- no docker-in-docker
        # on this pod). --max-model-len 18000 replaces vLLM's auto-derived default, which
        # is the base model's raw max_position_embeddings (262144) -- wildly oversized for
        # a page here (measured max 4621 output tokens; ~6144 vision tokens at the client's
        # own 3072x2048 resize cap). --mm-processor-kwargs matches that same 3072x2048 cap,
        # so it is provably a no-op given the client already enforces it, but it's free and
        # matches the vendor config exactly. Deliberately NOT setting the vendor's
        # H100-scaled --max-num-batched-tokens/--max-num-seqs here: our client concurrency
        # (--batch-size, <=28) never approaches vLLM's own defaults (8192/1024) anyway, and
        # the vendor's scaled-down 4096 would undercut a single max-size page's ~6300-token
        # prefill (image+prompt), forcing chunked prefill where the higher default does it
        # in one step. Also NOT passing --enable-prefix-caching: it's already default-on in
        # this vLLM version, and chandra's own request builder puts the (per-page-unique)
        # image before the (constant) text prompt, so the shared prompt suffix never gets a
        # cache hit anyway on a single-pass-per-page batch job like this one.
        uv run --project "$PROJ" vllm serve datalab-to/chandra-ocr-2 \
            --served-model-name chandra --port 8200 \
            --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
            --limit-mm-per-prompt '{"image": 1}' \
            --max-model-len 18000 \
            --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
            > "work/${MODEL}_vllm.log" 2>&1 &
        SERVER_PID=$!
        wait_server 8200
        export VLLM_API_BASE=http://127.0.0.1:8200/v1
        export VLLM_MODEL_NAME=chandra
        export MODEL_CHECKPOINT=datalab-to/chandra-ocr-2
        ;;
esac

uv run --project "$PROJ" python "$PROJ/run.py" "$@"
