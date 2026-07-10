"""Baidu Unlimited-OCR via transformers (trust_remote_code).

Uses the model-card recipe: prompt "<image>document parsing.", base_size=1024,
image_size=640, crop_mode=True, no_repeat_ngram_size=35, ngram_window=128.
Wrong size/prompt combos are the usual cause of bad pages. The one deviation is
max_length -- see MAX_LENGTH below; the card's 32768 lets a degenerate page decode
for >8 minutes. Per-page mode for checkpointing; UNLIMITED_MULTI=1 switches to the
model's one-shot whole-pdf infer_multi (its headline feature, pdf-level checkpoint).
"""
import json
import os
import tempfile
import time
from pathlib import Path

from ocr_harness import (Adapter, PageResult, OUTPUTS, cli, combine, list_pdfs,
                         gpu_mem_mb, render_pdf)

MODEL_ID = os.environ.get("MODEL_ID", "baidu/Unlimited-OCR")

# Per-page decode ceiling (prompt + output; the prompt is ~907 image tokens).
# The card's 32768 is sized for whole-document infer_multi, not one page. Measured on
# Complex_table_layouts: healthy pages emit 823-1500 output tokens, but degenerate ones
# never emit EOS and run to whatever ceiling they are given -- page_0008 hit 7285/7285
# under an 8192 cap (154s) and was still going after 8 min under 32768. 4096 leaves ~2x
# headroom over the densest real page and bounds a runaway to ~67s. Greedy decoding
# (temperature=0), so this cannot change a page that already terminates on its own.
# It truncates degenerate output; it does not fix it -- char counts stay inflated.
MAX_LENGTH = int(os.environ.get("UNLIMITED_MAX_LENGTH", "4096"))


def _read_saved(out_dir: str) -> str:
    files = sorted(Path(out_dir).rglob("*"))
    texts = [f.read_text() for f in files
             if f.suffix in (".md", ".mmd", ".txt") and f.is_file()]
    return "\n\n".join(texts)


class UnlimitedOcr(Adapter):
    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()

    def process_page(self, image_path: Path) -> PageResult:
        with tempfile.TemporaryDirectory() as td:
            res = self.model.infer(
                self.tokenizer,
                prompt="<image>document parsing.",
                image_file=str(image_path),
                output_path=td,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
            text = res if isinstance(res, str) and res.strip() else _read_saved(td)
        return PageResult(markdown=text)

    def run_multi(self, pdfs):
        """One-shot long-horizon parsing: whole pdf in a single forward pass."""
        self.load()
        docs = []
        for pdf in list_pdfs(pdfs):
            out_md = OUTPUTS / "unlimited_ocr_multi" / f"{pdf.stem}.md"
            if out_md.exists():
                print(f"skip {pdf.stem} (done)")
                continue
            pages = render_pdf(pdf)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory() as td:
                res = self.model.infer_multi(
                    self.tokenizer,
                    prompt="<image>Multi page parsing.",
                    image_files=[str(p) for p in pages],
                    output_path=td,
                    image_size=1024,
                    max_length=32768,
                    no_repeat_ngram_size=35,
                    ngram_window=1024,
                    save_results=True,
                )
                text = res if isinstance(res, str) and res.strip() else _read_saved(td)
            dt = time.perf_counter() - t0
            out_md.write_text(text)
            docs.append({"pdf": pdf.name, "pages": len(pages),
                         "total_seconds": round(dt, 2),
                         "seconds_per_page": round(dt / len(pages), 3),
                         "total_chars": len(text), "gpu_mem_mb": gpu_mem_mb()})
            print(docs[-1])
        (OUTPUTS / "unlimited_ocr_multi" / "summary.json").write_text(
            json.dumps(docs, indent=2))


if __name__ == "__main__":
    if os.environ.get("UNLIMITED_MULTI") == "1":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--pdfs", nargs="*")
        UnlimitedOcr().run_multi(ap.parse_args().pdfs)
    else:
        cli("unlimited_ocr", UnlimitedOcr, default_batch=1)
