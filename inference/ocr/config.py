"""Tuned defaults. Every value here was measured — see CHANDRA.md for the evidence.

Override any of them with the matching env var; the CLI flags win over both.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
LOGS = ROOT / "logs"

MODEL = os.environ.get("MODEL_CHECKPOINT", "datalab-to/chandra-ocr-2")
SERVED_NAME = os.environ.get("VLLM_MODEL_NAME", "chandra")
PORT = int(os.environ.get("VLLM_PORT", "8200"))
API_BASE = os.environ.get("VLLM_API_BASE", f"http://127.0.0.1:{PORT}/v1")

# Pages in flight per request batch. 16 is the measured optimum: at 48 the scheduler
# preempts (a page is ~6,100 vision tokens of prefill against a ~229k-token KV cache)
# and throughput drops 39%. Do not raise it without re-measuring.
BATCH_SIZE = int(os.environ.get("CHANDRA_BATCH_SIZE", "16"))

# chandra's own default. 256 saturates the model's pixel cap and is *worse*: -1.2% visible
# text and 35% slower over a 68-page set, and the flowchart class loses 2,048 chars.
# Anything above 256 is clamped by scale_to_fit to a byte-identical image.
IMAGE_DPI = int(os.environ.get("IMAGE_DPI", "192"))

# vLLM server flags. --max-model-len is a REQUIREMENT on a 46 GB card, not a nicety: vLLM
# sizes the KV cache to ~229k tokens while the model config declares a 262,144-token
# context, and it refuses to start when the context exceeds the cache. Lower this further
# on a 24 GB card (see README).
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "18000"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.85"))

MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "12384"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "6"))

# Headers/footers are excluded by chandra by default; we keep them (they carry document ids,
# batch numbers and page numbers, which matter for these documents).
INCLUDE_HEADERS_FOOTERS = os.environ.get("INCLUDE_HEADERS_FOOTERS", "1") == "1"
INCLUDE_IMAGES = os.environ.get("INCLUDE_IMAGES", "1") == "1"

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
