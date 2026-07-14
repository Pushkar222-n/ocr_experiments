"""Chandra OCR 2 (datalab) via its CLI against an external pip vLLM server.

run.sh starts `vllm serve datalab-to/chandra-ocr-2 --served-model-name chandra`
and sets VLLM_API_BASE/VLLM_MODEL_NAME, replacing the docker-based
`chandra_vllm` launcher. Chandra processes whole pdfs with internal page
batching, so checkpointing is per-pdf, not per-page.
Native outputs: markdown + html + metadata json (token counts per page).
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from ocr_harness import gpu_mem_mb, list_pdfs, orient_pdf, output_root, pdf_page_count

MODEL = "chandra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="*")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true",
                    help="write to outputs/_smoke/<model>/; never resumed by a real run")
    ap.add_argument("--out-tag",
                    help="write to outputs/<model>/<tag>/ instead, so an A/B (e.g. the "
                         "orientation-corrected rerun) cannot overwrite or resume from "
                         "the baseline it is compared against")
    ap.add_argument("--no-orient", action="store_true",
                    help="skip the PP-LCNet rotation-correction pass (harness.orient_pdf)"
                         " and feed the pdf to chandra as-is")
    ap.add_argument("--out-dir",
                    help="write to outputs/<out-dir>/ instead of outputs/chandra/. Unlike "
                         "--out-tag (which nests under the model), this is a top-level "
                         "sibling, so scripts/compare.py picks it up as its own row and "
                         "the benchmarked outputs/chandra/ baseline is never touched")
    args = ap.parse_args()

    out_root = output_root(args.out_dir or MODEL, args.smoke, args.out_tag)
    out_root.mkdir(parents=True, exist_ok=True)
    docs = []
    for pdf in list_pdfs(args.pdfs):
        final_md = out_root / f"{pdf.stem}.md"
        metrics = out_root / f"{pdf.stem}.metrics.json"
        if final_md.exists():  # per-pdf checkpoint
            print(f"skip {pdf.stem} (done)")
            if metrics.exists():  # keep it in summary.json across resumes
                docs.append(json.loads(metrics.read_text()))
            continue
        work_out = out_root / pdf.stem
        work_out.mkdir(parents=True, exist_ok=True)
        n_pages = pdf_page_count(pdf)
        print(f"[{MODEL}] {pdf.stem}: {n_pages} pages")
        input_pdf = pdf
        if not args.no_orient:
            input_pdf, orient_report = orient_pdf(pdf)
            if orient_report["any_rotated"]:
                print(f"  rotation-corrected pages {orient_report['flagged_pages']} "
                      f"-> {input_pdf}")
        t0 = time.perf_counter()
        subprocess.run(
            ["chandra", str(input_pdf), str(work_out), "--method", "vllm",
             "--batch-size", str(args.batch_size), "--include-headers-footers"],
            check=True,
        )
        dt = time.perf_counter() - t0
        mds = sorted(work_out.rglob("*.md"))
        if mds:
            shutil.copy(mds[0], final_md)
        for suffix, dest_suffix in ((".html", ".html"), ("_metadata.json", ".metadata.json")):
            hits = sorted(work_out.rglob(f"*{suffix}"))
            if hits:
                shutil.copy(hits[0], out_root / f"{pdf.stem}{dest_suffix}")
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
