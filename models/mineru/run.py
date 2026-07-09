"""MinerU 2.5 with the vlm-vllm-engine backend (vLLM in-process via pip, no docker).

MinerU parses whole pdfs with its own internal page batching, so checkpointing
is per-pdf. Native outputs: markdown + content_list.json (+ middle.json layout).
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from ocr_harness import OUTPUTS, gpu_mem_mb, list_pdfs, pdf_page_count

MODEL = "mineru"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="*")
    ap.add_argument("--backend", default="vlm-vllm-engine",
                    help="vlm-vllm-engine | vlm-transformers | pipeline")
    args = ap.parse_args()

    out_root = OUTPUTS / MODEL
    out_root.mkdir(parents=True, exist_ok=True)
    docs = []
    for pdf in list_pdfs(args.pdfs):
        final_md = out_root / f"{pdf.stem}.md"
        if final_md.exists():  # per-pdf checkpoint
            print(f"skip {pdf.stem} (done)")
            continue
        work_out = out_root / pdf.stem
        work_out.mkdir(parents=True, exist_ok=True)
        n_pages = pdf_page_count(pdf)
        print(f"[{MODEL}] {pdf.stem}: {n_pages} pages")
        t0 = time.perf_counter()
        subprocess.run(
            ["mineru", "-p", str(pdf), "-o", str(work_out), "-b", args.backend],
            check=True,
        )
        dt = time.perf_counter() - t0
        mds = sorted(work_out.rglob("*.md"))
        if mds:
            shutil.copy(mds[0], final_md)
        hits = sorted(work_out.rglob("*content_list.json"))
        if hits:
            shutil.copy(hits[0], out_root / f"{pdf.stem}.content_list.json")
        doc = {"pdf": pdf.name, "pages": n_pages, "total_seconds": round(dt, 2),
               "seconds_per_page": round(dt / n_pages, 3),
               "total_chars": len(final_md.read_text()) if final_md.exists() else 0,
               "gpu_mem_mb": gpu_mem_mb()}
        (out_root / f"{pdf.stem}.metrics.json").write_text(json.dumps(doc, indent=2))
        docs.append(doc)
        print(doc)
    (out_root / "summary.json").write_text(json.dumps(docs, indent=2))


if __name__ == "__main__":
    main()
