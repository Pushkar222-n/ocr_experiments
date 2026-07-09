"""PaddleOCR-VL-1.6 via the official paddleocr doc-parser pipeline.

Native outputs: markdown + layout json (with per-region scores). Batching is
handled inside the pipeline; we feed page images one by one for checkpointing.
"""
import tempfile
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli


def _read_first(td: str, suffix: str) -> str:
    for f in sorted(Path(td).rglob(f"*{suffix}")):
        return f.read_text()
    return ""


class PaddleOcrVl(Adapter):
    def load(self):
        from paddleocr import PaddleOCRVL

        self.pipeline = PaddleOCRVL(pipeline_version="v1.6")

    def process_page(self, image_path: Path) -> PageResult:
        res = self.pipeline.predict(str(image_path))[0]
        with tempfile.TemporaryDirectory() as td:
            res.save_to_markdown(save_path=td)
            md = _read_first(td, ".md")
        with tempfile.TemporaryDirectory() as td:
            res.save_to_json(save_path=td)
            js = _read_first(td, ".json")
        return PageResult(markdown=md, native=js, native_ext="json")


if __name__ == "__main__":
    cli("paddleocr_vl", PaddleOcrVl, default_batch=1)
