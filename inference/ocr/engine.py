"""The chandra client, and how a document's pages are batched onto the vLLM server.

We drive chandra's Python API rather than shelling out to its CLI, for two reasons:
  - its CLI's file discovery is one level deep (glob, not rglob), so nested input folders
    would be silently skipped;
  - the CLI re-pays Python import cost on every file.

`InferenceManager(method="vllm")` is an HTTP client — it never loads weights into this
process. The weights live in the server. That is why this process can stay entirely on CPU.
"""
import logging
import os
import time
from pathlib import Path

from .config import (
    API_BASE,
    BATCH_SIZE,
    INCLUDE_HEADERS_FOOTERS,
    INCLUDE_IMAGES,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL,
    SERVED_NAME,
)

log = logging.getLogger(__name__)


def _export_settings():
    """chandra reads these through pydantic BaseSettings at import time."""
    os.environ.setdefault("VLLM_API_BASE", API_BASE)
    os.environ.setdefault("VLLM_MODEL_NAME", SERVED_NAME)
    os.environ.setdefault("MODEL_CHECKPOINT", MODEL)


class Chandra:
    def __init__(self, batch_size: int = BATCH_SIZE):
        _export_settings()
        from chandra.model import InferenceManager

        self.batch_size = batch_size
        self.model = InferenceManager(method="vllm")
        log.info("chandra client ready (server=%s, batch=%d)", API_BASE, batch_size)

    def load_pages(self, path: Path) -> list:
        """Rasterize a pdf (or open an image) exactly the way chandra does — same DPI, same
        min-dimension upscaling — so what we send is what it was tuned on."""
        from chandra.input import load_file

        return load_file(str(path), {})

    def ocr_pages(self, images: list, on_batch=None) -> list:
        """Run every page of one document. Pages go up in batches of `batch_size`; chandra
        fans each batch out across threads, so the server sees `batch_size` concurrent
        requests and continuously batches them. 16 is the measured optimum — see config.py."""
        from chandra.model.schema import BatchInputItem

        results = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in chunk]
            t0 = time.perf_counter()
            out = self.model.generate(
                batch,
                include_images=INCLUDE_IMAGES,
                include_headers_footers=INCLUDE_HEADERS_FOOTERS,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                max_workers=min(len(batch), self.batch_size),
                max_retries=MAX_RETRIES,
            )
            results.extend(out)
            if on_batch:
                on_batch(len(chunk), time.perf_counter() - t0)
        return results


def server_is_up(api_base: str = API_BASE, timeout: float = 3.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False
