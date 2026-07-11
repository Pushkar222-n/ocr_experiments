"""MinerU 3.x, in-process (vlm-engine) or against a served model (vlm-http-client).

MinerU parses whole pdfs with its own internal page batching, so checkpointing
is per-pdf. Native outputs: markdown + content_list.json (+ middle.json layout).
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ocr_harness import gpu_mem_mb, list_pdfs, output_root, pdf_page_count

MODEL = "mineru"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", nargs="*")
    # MinerU 3.x renamed the backends: the old `vlm-transformers` is gone and the CLI now
    # takes `vlm-engine`, which strips to "engine" and calls get_vlm_engine('auto') ->
    # vllm, else lmdeploy, else transformers (mineru/utils/engine_utils.py). Neither vllm
    # nor lmdeploy is in this venv, so it resolves to transformers — the backend we want.
    # That fallback is implicit: installing vllm here would silently switch engines.
    # The chosen engine is logged ("Using <x> as the inference engine for VLM") — check it.
    ap.add_argument("--backend", default="vlm-engine",
                    help="vlm-engine (-> transformers here) | vlm-http-client | pipeline")
    # The transformers engine is the *unaccelerated* path: it static-batches 8 pages, pads
    # them, and runs one generate() per batch, so every batch costs as much as its longest
    # sequence (mineru_vl_utils/vlm_client/transformers_client.py:189-241). Measured 72.4
    # s/page over the 68-page set, vs the 2.12 fps the model card quotes for vllm. Point
    # --backend vlm-http-client at a vllm serve of the same weights to get the fast path.
    ap.add_argument("--url", default=os.environ.get("MINERU_URL"),
                    help="base url of an openai-compatible server, for vlm-http-client. "
                         "NOT the /v1 endpoint: mineru appends /v1/chat/completions itself "
                         "(mineru_vl_utils/vlm_client/http_client.py:120)")
    ap.add_argument("--out-tag",
                    help="write to outputs/<model>/<tag>/ instead of outputs/<model>/, so an "
                         "engine A/B run cannot overwrite the baseline it is compared against")
    ap.add_argument("--smoke", action="store_true",
                    help="write to outputs/_smoke/<model>/; never resumed by a real run")
    args = ap.parse_args()

    if args.backend.endswith("http-client") and not args.url:
        ap.error(f"--backend {args.backend} needs --url pointing at the served model")

    out_root = output_root(MODEL, args.smoke)
    if args.out_tag:
        out_root = out_root / args.out_tag
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
        t0 = time.perf_counter()
        cmd = ["mineru", "-p", str(pdf), "-o", str(work_out), "-b", args.backend]
        if args.url:
            cmd += ["-u", args.url]
        # mineru runs in a child process, so a single gpu_mem_mb() after it returns reads
        # 0 -- the child has exited and freed everything. Sample while it is alive and keep
        # the max, which is the same measurement the per-page adapters report.
        # Caveat under vlm-http-client: this reads the whole GPU, and the weights live in a
        # vllm server run.sh started outside this process. vllm preallocates its KV pool to
        # --gpu-memory-utilization, so the number reflects that reservation, not demand --
        # it is not comparable to the transformers figure. Only trust the timings there.
        proc = subprocess.Popen(cmd)
        peak_mb = 0
        while proc.poll() is None:
            sample = gpu_mem_mb()
            if sample:
                peak_mb = max(peak_mb, sample)
            time.sleep(2)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
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
               "max_gpu_mem_mb": peak_mb}
        (out_root / f"{pdf.stem}.metrics.json").write_text(json.dumps(doc, indent=2))
        docs.append(doc)
        print(doc)
    (out_root / "summary.json").write_text(json.dumps(docs, indent=2))


if __name__ == "__main__":
    main()
