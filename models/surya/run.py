"""Surya 2 foundation OCR against an external pip-installed vLLM server.

run.sh starts `vllm serve datalab-to/surya-ocr-2` and sets
SURYA_INFERENCE_URL — no docker needed (RunPod pods can't do docker-in-docker).
Native output: block json (label, bbox, html, confidence — Surya is the one
model here that reports per-block confidence). Markdown assembled from block
HTML via markdownify.
"""
import json
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli


class SuryaAdapter(Adapter):
    def load(self):
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        self.rec = RecognitionPredictor(SuryaInferenceManager())

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        from markdownify import markdownify
        from PIL import Image

        preds = self.rec([Image.open(p).convert("RGB") for p in image_paths])
        results = []
        for pred in preds:
            blocks = getattr(pred, "blocks", []) or []
            html = "\n".join(b.html for b in blocks if getattr(b, "html", None))
            confs = [b.confidence for b in blocks
                     if getattr(b, "confidence", None) is not None]
            native = json.dumps(
                [{"label": getattr(b, "label", None),
                  "bbox": getattr(b, "bbox", None),
                  "confidence": getattr(b, "confidence", None),
                  "html": getattr(b, "html", None)} for b in blocks],
                ensure_ascii=False)
            extra = {"mean_confidence": round(sum(confs) / len(confs), 4)} if confs else {}
            results.append(PageResult(markdown=markdownify(html), native=native,
                                      native_ext="json", extra=extra))
        return results


if __name__ == "__main__":
    cli("surya", SuryaAdapter, default_batch=8)
