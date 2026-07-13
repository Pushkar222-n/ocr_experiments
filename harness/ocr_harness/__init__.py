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
ORIENTED_DIR = ROOT / "work" / "oriented"
ORIENTED_PREVIEW_DIR = ROOT / "work" / "oriented_preview"

_ORIENT_CLASSIFIER = None


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


def classify_rotation(bgr_image) -> tuple[int, float]:
    """4-class (0/90/180/270) document-orientation classifier: PaddleOCR's PP-LCNet
    model, onnxruntime backend so this needs neither paddlepaddle nor a GPU. `bgr_image`
    is a numpy array in BGR order (cv2/PaddleX convention).

    Returns (angle, score). `angle` is how far the page must be rotated
    **counter-clockwise** to be upright -- verified against PaddleX's own
    `doc_preprocessor` pipeline, which applies the predicted label straight into
    `rotate_image()` (cv2.getRotationMatrix2D, positive = counter-clockwise) with no
    sign flip. PIL's `Image.rotate(angle)` uses the same counter-clockwise convention,
    so `img.rotate(angle, expand=True)` reproduces PaddleX's own correction exactly.
    """
    global _ORIENT_CLASSIFIER
    if _ORIENT_CLASSIFIER is None:
        from paddleocr import DocImgOrientationClassification
        _ORIENT_CLASSIFIER = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori", engine="onnxruntime",
        )
    results = list(_ORIENT_CLASSIFIER.predict(bgr_image, batch_size=1))
    result = results[0]
    return int(result["label_names"][0]), float(result["scores"][0])


def orient_pdf(pdf_path: Path, dpi: int = 150, preview: bool = True) -> tuple[Path, dict]:
    """Detect and correct page rotation with `classify_rotation`, writing a copy of the
    pdf with each rotated page's `/Rotate` fixed up to `work/oriented/<stem>.pdf`. Any
    renderer that honors `/Rotate` (pypdfium2, and e.g. chandra's own pdf loader) then
    renders the page upright with no other change needed -- this is PDF-level, model
    agnostic, and any adapter can call it before handing off a pdf or rendering pages.

    Cached on (size, mtime) so a resumed run does not reclassify. Returns
    `(path_to_use, report)`: `path_to_use` is `pdf_path` itself, untouched, if no page
    needed correction. `report["flagged_pages"]` lists 0-indexed pages that were
    rotated, and if `preview`, `report["preview_dir"]` holds before/after pngs for each
    one so a human can check the correction before trusting it.
    """
    import numpy as np
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter

    ORIENTED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ORIENTED_DIR / f"{pdf_path.stem}.json"
    corrected_path = ORIENTED_DIR / f"{pdf_path.stem}.pdf"
    stat = pdf_path.stat()
    cache_key = {"size": stat.st_size, "mtime": stat.st_mtime}
    if report_path.exists():
        cached = json.loads(report_path.read_text())
        if cached.get("_cache_key") == cache_key:
            used = corrected_path if cached["any_rotated"] else pdf_path
            return used, cached

    preview_dir = ORIENTED_PREVIEW_DIR / pdf_path.stem
    doc = pdfium.PdfDocument(str(pdf_path))
    pages_report = []
    for i in range(len(doc)):
        pil_img = doc[i].render(scale=dpi / 72).to_pil().convert("RGB")
        bgr = np.array(pil_img)[:, :, ::-1]
        angle, score = classify_rotation(bgr)
        pages_report.append({"page": i, "detected_angle": angle, "score": round(score, 4)})
        if angle != 0 and preview:
            preview_dir.mkdir(parents=True, exist_ok=True)
            pil_img.save(preview_dir / f"page_{i:04d}_before.png")
            pil_img.rotate(angle, expand=True).save(preview_dir / f"page_{i:04d}_after.png")
    doc.close()

    flagged = [p["page"] for p in pages_report if p["detected_angle"] != 0]
    report = {
        "_cache_key": cache_key,
        "pdf": pdf_path.name,
        "num_pages": len(pages_report),
        "pages": pages_report,
        "flagged_pages": flagged,
        "any_rotated": bool(flagged),
        "preview_dir": str(preview_dir) if flagged and preview else None,
    }

    if flagged:
        reader = PdfReader(str(pdf_path))
        writer = PdfWriter()
        for i, page in enumerate(reader.pages):
            angle = pages_report[i]["detected_angle"]
            if angle:
                # pypdf's Page.rotate() is clockwise; -angle reproduces the
                # counter-clockwise correction classify_rotation calls for.
                page.rotate(-angle)
            writer.add_page(page)
        with open(corrected_path, "wb") as f:
            writer.write(f)

    report_path.write_text(json.dumps(report, indent=2))
    return (corrected_path if flagged else pdf_path), report


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


def output_root(model_name: str, smoke: bool = False, tag: str | None = None) -> Path:
    """Where a run writes. Smoke runs go to a throwaway tree so their pages can never
    satisfy a real run's checkpoint — a smoke test validates an *unproven* config, so
    its output is the last thing a real run should resume from.

    `tag` writes to outputs/<model>/<tag>/ instead, so an A/B of the same model on a
    different inference engine cannot overwrite (or resume from!) the baseline it is
    being compared against. compare.py prefers a tagged run over the untagged one and
    says so on stderr."""
    root = OUTPUTS / "_smoke" / model_name if smoke else OUTPUTS / model_name
    return root / tag if tag else root


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
        pdfs: list[str] | None = None, smoke: bool = False, out_tag: str | None = None):
    docs = []
    loaded = False
    out_root = output_root(model_name, smoke, out_tag)
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
    ap.add_argument("--out-tag",
                    help="write to outputs/<model>/<tag>/ instead, so an engine A/B cannot "
                         "overwrite or resume from the baseline it is compared against")
    args = ap.parse_args()
    run(model_name, make_adapter(), batch_size=args.batch_size, dpi=args.dpi,
        pdfs=args.pdfs, smoke=args.smoke, out_tag=args.out_tag)
