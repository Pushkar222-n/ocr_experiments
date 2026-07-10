"""Shared OCR experiment harness.

Every model adapter implements load() + process_page() (or process_batch()).
The runner handles: pdf -> page png cache, per-page checkpointing (the
page_NNNN.meta.json metric file is written last and acts as the "done" marker),
timing, GPU memory sampling, and assembling per-page markdown into
outputs/<model>/<pdf_stem>.md.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SET = ROOT / "data" / "Evaluation set" / "sample_set"
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work" / "pages"


@dataclass
class PageResult:
    markdown: str
    native: str | None = None  # native output when it isn't markdown (json/html/latex)
    native_ext: str = "json"
    extra: dict = field(default_factory=dict)  # confidence, token counts, ...


class Adapter:
    def load(self):
        pass

    def process_page(self, image_path: Path) -> PageResult:
        raise NotImplementedError

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        return [self.process_page(p) for p in image_paths]


def render_pdf(pdf_path: Path, dpi: int = 200) -> list[Path]:
    import pypdfium2 as pdfium

    out_dir = WORK / f"{pdf_path.stem}_dpi{dpi}"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    paths = [out_dir / f"page_{i:04d}.png" for i in range(len(doc))]
    for i, p in enumerate(paths):
        if not p.exists():
            doc[i].render(scale=dpi / 72).to_pil().save(p)
    doc.close()
    return paths


def pdf_page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def gpu_mem_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return max(int(x) for x in out.stdout.split())
    except Exception:
        return None


def list_pdfs(names: list[str] | None = None) -> list[Path]:
    pdfs = sorted(SAMPLE_SET.glob("*.pdf"))
    if names:
        want = {Path(n).stem for n in names}
        pdfs = [p for p in pdfs if p.stem in want]
    return pdfs


def output_root(model_name: str, smoke: bool = False) -> Path:
    """Where a run writes. Smoke runs go to a throwaway tree so their pages can never
    satisfy a real run's checkpoint — a smoke test validates an *unproven* config, so
    its output is the last thing a real run should resume from."""
    return OUTPUTS / "_smoke" / model_name if smoke else OUTPUTS / model_name


def combine(out_root: Path, pdf: Path, page_dir: Path) -> dict:
    metas = [json.loads(p.read_text()) for p in sorted(page_dir.glob("page_*.meta.json"))]
    md = "\n\n".join(
        (page_dir / f"page_{m['page']:04d}.md").read_text() for m in metas
    )
    (out_root / f"{pdf.stem}.md").write_text(md)
    total_s = sum(m["seconds"] for m in metas)
    doc = {
        "pdf": pdf.name,
        "pages": len(metas),
        "total_seconds": round(total_s, 2),
        "seconds_per_page": round(total_s / max(len(metas), 1), 3),
        "total_chars": sum(m["chars"] for m in metas),
        "max_gpu_mem_mb": max((m.get("gpu_mem_mb") or 0) for m in metas) or None,
    }
    # average any numeric extras the adapter reported (confidence, token counts, ...)
    if metas:
        skip = {"page", "seconds", "chars", "gpu_mem_mb"}
        for k in metas[0]:
            if k in skip:
                continue
            vals = [m[k] for m in metas if isinstance(m.get(k), (int, float))]
            if vals:
                doc[f"mean_{k}"] = round(sum(vals) / len(vals), 4)
    (out_root / f"{pdf.stem}.metrics.json").write_text(json.dumps(doc, indent=2))
    return doc


def run(model_name: str, adapter: Adapter, batch_size: int = 1, dpi: int = 200,
        pdfs: list[str] | None = None, smoke: bool = False):
    docs = []
    loaded = False
    out_root = output_root(model_name, smoke)
    if smoke:
        print(f"[{model_name}] SMOKE RUN -> {out_root} (discardable; will not be "
              f"resumed by a real run)", flush=True)
    for pdf in list_pdfs(pdfs):
        pages = render_pdf(pdf, dpi)
        page_dir = out_root / pdf.stem / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            (i, p) for i, p in enumerate(pages)
            if not (page_dir / f"page_{i:04d}.meta.json").exists()
        ]
        print(f"[{model_name}] {pdf.stem}: {len(pages)} pages, {len(pending)} to do", flush=True)
        if pending and not loaded:
            adapter.load()
            loaded = True
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            t0 = time.perf_counter()
            results = adapter.process_batch([p for _, p in chunk])
            per_page = (time.perf_counter() - t0) / len(chunk)  # batch avg
            for (i, _), r in zip(chunk, results):
                (page_dir / f"page_{i:04d}.md").write_text(r.markdown)
                if r.native is not None:
                    (page_dir / f"page_{i:04d}.{r.native_ext}").write_text(r.native)
                meta = {"page": i, "seconds": round(per_page, 3),
                        "chars": len(r.markdown), "gpu_mem_mb": gpu_mem_mb(), **r.extra}
                # written last: acts as the per-page "done" marker. Kept distinct from
                # page_NNNN.json so it can't clobber a native_ext="json" adapter output.
                (page_dir / f"page_{i:04d}.meta.json").write_text(json.dumps(meta))
            print(f"  {start + len(chunk)}/{len(pending)} ({per_page:.2f}s/page)", flush=True)
        docs.append(combine(out_root, pdf, page_dir))
    (out_root / "summary.json").write_text(json.dumps(docs, indent=2))
    print(json.dumps(docs, indent=2))


def cli(model_name: str, make_adapter, default_batch: int = 1):
    ap = argparse.ArgumentParser(description=f"Run {model_name} over sample_set")
    ap.add_argument("--pdfs", nargs="*", help="pdf stems to run (default: all)")
    ap.add_argument("--batch-size", type=int, default=default_batch)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                    help="write to outputs/_smoke/<model>/; never resumed by a real run")
    args = ap.parse_args()
    run(model_name, make_adapter(), batch_size=args.batch_size, dpi=args.dpi,
        pdfs=args.pdfs, smoke=args.smoke)
