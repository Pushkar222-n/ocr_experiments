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
# PaddleX caches weights under ~/.paddlex (paddlex/utils/cache.py: CACHE_DIR =
# os.environ.get("PADDLE_PDX_CACHE_HOME", ~/.paddlex)). $HOME is the container's ephemeral
# overlay, not /workspace — weights would be lost on pod restart and reclaim.sh could never
# find them. Set globally, not just for paddleocr_vl: harness.orient_pdf's PP-LCNet
# orientation classifier pulls from the same cache, and any model may call it.
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-/workspace/.cache/paddlex}"
mkdir -p work "$PADDLE_PDX_CACHE_HOME"

# Mirror everything (this script, uv, and the model's stdout+stderr) to a per-model
# run log while still printing to the terminal. Appended, not truncated, so a resumed
# run keeps the history of the attempts that got it there.
RUN_LOG="work/${MODEL}_run.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "===== $(date -Is) :: ./run.sh $MODEL $* ====="

VENV="$PROJ/.venv/bin"

# Sync only when the venv isn't already built. `uv sync` re-fetches from the uv cache, and
# if that cache has been cleaned (scripts/reclaim.sh, or a bare `uv cache clean`) the sync
# re-downloads the whole dependency tree -- ~16 GB for a vLLM-carrying venv like chandra's.
# The venv is a complete, independent copy (this volume has no hardlinks, so uv *copies*
# out of the cache), so an existing venv needs neither the cache nor a re-sync. Force one
# with FORCE_SYNC=1 after editing a pyproject.
if [ "${FORCE_SYNC:-0}" = "1" ] || [ ! -x "$VENV/python" ]; then
    echo "== syncing env for $MODEL =="
    uv sync --project "$PROJ"
else
    echo "== env for $MODEL already built ($VENV), skipping sync (FORCE_SYNC=1 to override) =="
fi
[ -x "$VENV/python" ] || { echo "no venv python at $VENV/python"; exit 1; }

# `uv run` used to put the venv's bin/ on PATH for us. Calling .venv/bin/python directly
# does NOT, and an adapter that shells out to a console script installed in that venv --
# models/chandra/run.py does exactly this: subprocess.run(["chandra", ...]) -- then dies with
# FileNotFoundError: 'chandra'. Put the venv's bin/ on PATH ourselves.
export PATH="$PWD/$VENV:$PATH"

# Invoke the venv's binaries DIRECTLY rather than through `uv run`.
#
# `uv run` re-resolves the environment on every call, and to do that it probes the Python
# interpreter through a temp file **inside the uv cache**. If that cache is missing or is
# removed mid-flight, uv dies with:
#     error: Failed to query Python interpreter
#       Caused by: No such file or directory at "/workspace/.cache/uv/.tmpXXXXXX"
# which is exactly what killed a chandra run here: a `uv cache clean` (run to free disk for
# the weights) raced the in-flight `uv run`, and the adapter never started -- vLLM came up,
# served 0 requests, and the run sat there looking like a slow model load for 10 minutes.
#
# Calling .venv/bin/<x> touches the uv cache not at all, so the run is immune to whatever
# happens to that cache while it is in flight. (An earlier version of this comment blamed a
# `uv run` project *lock* held by the long-lived server. That was wrong -- there is no such
# deadlock; the error above is the real and only mechanism. Do not reintroduce that story.)
#
# Corollary, and it is a footgun worth stating plainly: **never `uv cache clean` while any
# uv process is live.** It fails silently and looks like a hang.

SERVER_PID=""
# Only kill a server THIS script started. A server we merely attached to (see server_up
# below) outlives us on purpose -- that is the whole point of keeping one resident.
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Is a vLLM already serving on this port? If so, reuse it instead of loading the weights
# again. A chandra server takes ~6 min to come up (weights + torch.compile + CUDA graph
# capture) and that cost is paid on EVERY run -- a smoke test then a real run pays it
# twice. Keeping one server resident across runs makes the second run start decoding
# immediately. scripts/serve_chandra.sh starts a detached one; this then finds it.
server_up() { curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }

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
        # Rotation-correction pre-pass, on the CPU, BEFORE vLLM reserves the card.
        # harness.classify_rotation already pins itself to device=cpu, but PaddleX resolves
        # a device by probing for a visible accelerator, so this hides the GPU outright:
        # two inference runtimes on one card, with vLLM at --gpu-memory-utilization 0.85,
        # is an OOM waiting to happen. orient_pdf caches to work/oriented/, so the
        # adapter's own orient_pdf() call later in the run is a pure cache read.
        if [[ " $* " != *" --no-orient "* ]]; then
            # mirror the run's own --pdfs selection, so a smoke test on one pdf does not
            # pay to classify all five (orient_prepass.py with no args = the whole set)
            ORIENT_PDFS=(); IN_PDFS=0
            for a in "$@"; do
                case "$a" in
                    --pdfs) IN_PDFS=1 ;;
                    -*)     IN_PDFS=0 ;;
                    *)      [ "$IN_PDFS" = 1 ] && ORIENT_PDFS+=("$a") ;;
                esac
            done
            echo "== rotation pre-pass (cpu-only, GPU hidden) =="
            CUDA_VISIBLE_DEVICES="" "$VENV/python" \
                scripts/orient_prepass.py "${ORIENT_PDFS[@]+"${ORIENT_PDFS[@]}"}"
        fi
        # CHANDRA_TUNED=1 (default) serves with the tuned flags below. CHANDRA_TUNED=0 is
        # the control arm: vLLM's own defaults, nothing but the model. Everything else --
        # weights, sampling, --batch-size, --include-headers-footers, and the rotation
        # pre-pass above -- is identical across both, so the serving flags are the only
        # variable. Orientation is ON in both arms; it is the default and is never gated
        # on this switch.
        #
        # --max-model-len and --mm-processor-kwargs come from datalab's own launcher
        # (chandra/scripts/vllm.py, which run.sh can't use directly -- no docker-in-docker
        # on this pod). --max-model-len 18000 replaces vLLM's auto-derived default, which
        # is the base model's raw max_position_embeddings (262144) -- wildly oversized for
        # a page here (measured max 4621 output tokens; ~6144 vision tokens at the client's
        # own 3072x2048 resize cap). Expect the control arm to strain or fail on exactly
        # this: sizing a KV cache for a 262144-token context on a 46 GB card is the flag's
        # whole reason to exist. --mm-processor-kwargs matches that same 3072x2048 cap, so
        # it is provably a no-op given the client already enforces it, but it's free and
        # matches the vendor config exactly. Deliberately NOT setting the vendor's
        # H100-scaled --max-num-batched-tokens/--max-num-seqs here: our client concurrency
        # (--batch-size, <=28) never approaches vLLM's own defaults (8192/1024) anyway, and
        # the vendor's scaled-down 4096 would undercut a single max-size page's ~6300-token
        # prefill (image+prompt), forcing chunked prefill where the higher default does it
        # in one step. Also NOT passing --enable-prefix-caching: chandra's own request
        # builder puts the (per-page-unique) image before the (constant) text prompt, so the
        # shared prompt suffix never gets a cache hit anyway on a single-pass-per-page batch
        # job like this one. (An earlier note here also claimed the flag was default-on in
        # this vLLM version and therefore inert -- that is FALSE: 0.19.1 logs
        # `enable_prefix_caching=False` in its engine config. The no-cache-hit argument is
        # the only one that holds, and it is sufficient.)
        TUNED_ARGS=()
        if [ "${CHANDRA_TUNED:-1}" = "1" ]; then
            TUNED_ARGS=(--max-model-len 18000
                        --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}')
            echo "== chandra: TUNED serving flags: ${TUNED_ARGS[*]} =="
        else
            echo "== chandra: CONTROL arm, vLLM default serving flags (CHANDRA_TUNED=0) =="
        fi
        # Reuse a resident server if one is already up on :8200 (scripts/serve_chandra.sh,
        # or a previous run left one). Loading chandra costs ~6 min of weights +
        # torch.compile + CUDA-graph capture, and that is paid on every run otherwise.
        # NOTE: an attached server keeps whatever flags it was STARTED with -- CHANDRA_TUNED
        # cannot retune a running server. Switching arms means restarting it, so the check
        # below refuses to silently mislabel a run.
        if server_up 8200; then
            echo "== attaching to the vLLM already serving on :8200 (not starting a new one) =="
            echo "   its serving flags are whatever it was started with; CHANDRA_TUNED=${CHANDRA_TUNED:-1}"
            echo "   is NOT applied to an existing server. Restart it to change arms:"
            echo "   pkill -f '[v]llm serve datalab' && scripts/serve_chandra.sh"
        else
            "$VENV/vllm" serve datalab-to/chandra-ocr-2 \
                --served-model-name chandra --port 8200 \
                --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
                --limit-mm-per-prompt '{"image": 1}' \
                "${TUNED_ARGS[@]+"${TUNED_ARGS[@]}"}" \
                > "work/${MODEL}_vllm.log" 2>&1 &
            SERVER_PID=$!
            wait_server 8200
        fi
        export VLLM_API_BASE=http://127.0.0.1:8200/v1
        export VLLM_MODEL_NAME=chandra
        export MODEL_CHECKPOINT=datalab-to/chandra-ocr-2
        ;;
esac

"$VENV/python" "$PROJ/run.py" "$@"
