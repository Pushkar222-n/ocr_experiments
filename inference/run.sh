#!/usr/bin/env bash
# One command: OCR a folder (or zip) of documents.
#
#   ./run.sh /path/to/docs
#   ./run.sh /path/to/docs.zip
#   ./run.sh /path/to/docs --limit 2          # smoke test
#   ./run.sh /path/to/docs --name my_batch    # name the output run
#
# Brings the venv and the vLLM server up if they are not already, then runs. Safe to re-run:
# finished documents are skipped. The server is LEFT RUNNING afterwards, because loading it
# costs minutes and the next job should not pay that again — free the VRAM with
# `./serve.sh --stop` when you are done for the day.
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "usage: ./run.sh <input-folder|input.zip> [options]"
  echo "       ./run.sh --help   for all options"
  exit 1
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  exec .venv/bin/python -m ocr --help
fi

INPUT="$1"; shift

if [[ ! -x .venv/bin/python || "${FORCE_SYNC:-0}" == "1" ]]; then
  echo "building venv (~16 GB, once per pod)..."
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}" uv sync
fi

./serve.sh    # idempotent: attaches to a live server, starts one if needed

# The client never touches the GPU: chandra's vllm method is a pure HTTP client, and the
# orientation classifier runs on onnxruntime/CPU. Hiding the GPU from this process makes that
# structural instead of a rule someone has to remember — the card belongs to the server alone.
CUDA_VISIBLE_DEVICES="" exec .venv/bin/python -m ocr --input "$INPUT" "$@"
