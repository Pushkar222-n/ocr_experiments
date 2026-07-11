"""GLM-OCR (zai-org) via glmocr[selfhosted]: vLLM decoder + PP-DocLayout-V3 layout.

Native outputs: markdown + layout json with bounding boxes. Layout model can
run on CPU (layout_device) leaving the GPU to the OCR decoder.
"""
import os
import tempfile
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli


def _read_first(td: str, suffix: str) -> str:
    for f in sorted(Path(td).rglob(f"*{suffix}")):
        return f.read_text()
    return ""


class GlmOcrAdapter(Adapter):
    def load(self):
        from glmocr import GlmOcr

        # mode="selfhosted" is not optional: MaaSApiConfig.enabled defaults to True, so a
        # bare GlmOcr() posts to Zhipu's *cloud* endpoint rather than anything local. In
        # selfhosted mode only PP-DocLayout-V3 runs in-process; the decoder is an HTTP call
        # to {host}:{port}/v1/chat/completions, which run.sh points at a local vllm serve.
        self.parser = GlmOcr(
            mode="selfhosted",
            ocr_api_host=os.environ.get("GLM_OCR_HOST", "127.0.0.1"),
            ocr_api_port=int(os.environ.get("GLM_OCR_PORT", "8300")),
            model=os.environ.get("GLM_OCR_MODEL", "glm-ocr"),
            layout_device=os.environ.get("LAYOUT_DEVICE", "cpu"),
        )

    def process_page(self, image_path: Path) -> PageResult:
        result = self.parser.parse(str(image_path))
        with tempfile.TemporaryDirectory() as td:
            result.save(output_dir=td)
            md = _read_first(td, ".md")
            js = _read_first(td, ".json")
        if not md:  # fall back to object attrs if save() layout differs
            md = getattr(result, "markdown", "") or str(result)
        return PageResult(markdown=md, native=js or None, native_ext="json")


if __name__ == "__main__":
    cli("glm_ocr", GlmOcrAdapter, default_batch=1)
