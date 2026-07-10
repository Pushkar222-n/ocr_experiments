#!/usr/bin/env bash
# Usage: scripts/reclaim.sh <finished-model> [hf-repo-id ...]
# Reclaim disk after a model has finished its run. /workspace is MooseFS and has no
# hardlinks, so uv *copies* out of its cache: every `uv sync` writes the venv's bytes
# twice. Left alone, the cache alone reaches ~20 GB.
#
# Safe because the venvs are independent copies, not links into the cache — clearing
# the cache cannot break an env that already exists, and uv.lock makes any rebuild
# deterministic. Outputs under outputs/ are never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: scripts/reclaim.sh <model> [hf-repo-id ...]}"
shift || true
HF_HOME="${HF_HOME:-$PWD/hf_cache}"

before=$(du -sm /workspace/.cache "models/$MODEL" 2>/dev/null | awk '{s+=$1} END {print s}')

uv cache clean
rm -rf "models/$MODEL/.venv"
for repo in "$@"; do
    rm -rf "$HF_HOME/hub/models--${repo//\//--}"
done

after=$(du -sm /workspace/.cache "models/$MODEL" 2>/dev/null | awk '{s+=$1} END {print s}')
echo "reclaimed $(( before - after )) MiB (${before} -> ${after} MiB)"
