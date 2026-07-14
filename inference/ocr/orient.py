"""Page-rotation correction. Always on — it is a permanent default, never an A/B knob.

Sideways-scanned pages cost real text: on the 7 rotated pages of a 32-page test document it
recovers +4.8% more tokens, against -0.7% on the same document's unrotated pages (control).

Uses PaddleOCR's PP-LCNet_x1_0_doc_ori — a tiny 4-class (0/90/180/270) classifier — on the
**onnxruntime** backend, so it needs neither paddlepaddle nor a GPU.

THE CLASSIFIER MUST STAY OFF THE GPU. Here that is structural rather than a rule to remember:
this whole process is an HTTP client (chandra's vllm method never loads weights locally), and
run.sh exports CUDA_VISIBLE_DEVICES="" for it. The GPU belongs to the vLLM server alone.

Sign convention, verified from PaddleX's source rather than the model card (which documents
the labels but not the direction): its pipeline calls rotate_image(img, angle) with the raw
predicted label and applies it via cv2.getRotationMatrix2D, whose positive angle is
counter-clockwise. pypdf's Page.rotate() is clockwise. Hence -angle.
"""
import json
import logging
from pathlib import Path

from .config import WORK

log = logging.getLogger(__name__)
_classifier = None

# a page is only rotated when the model is reasonably sure; a coin-flip prediction that
# turns an upright page sideways would cost far more than it saves
MIN_CONFIDENCE = 0.70
RENDER_DPI = 150  # only for classification — the OCR render is separate and untouched


def _get_classifier():
    global _classifier
    if _classifier is None:
        from paddleocr import DocImgOrientationClassification

        _classifier = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori", engine="onnxruntime"
        )
    return _classifier


def _classify(bgr) -> tuple[int, float]:
    res = _get_classifier().predict(bgr)
    r = res[0] if isinstance(res, list) else res
    label = r.get("label_names", [None])[0] if isinstance(r, dict) else None
    score = float(r.get("scores", [0])[0]) if isinstance(r, dict) else 0.0
    try:
        return int(label), score
    except (TypeError, ValueError):
        return 0, 0.0


def orient_pdf(pdf: Path, cache_dir: Path | None = None) -> tuple[Path, dict]:
    """Return (pdf_to_ocr, report). If no page needs turning, the original path is returned
    unchanged — no copy, no rewrite. Cached on (size, mtime) so re-runs are free."""
    import numpy as np
    import pypdfium2 as pdfium

    cache_dir = cache_dir or (WORK / "oriented")
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = pdf.stat()
    key = f"{pdf.stem}_{stat.st_size}_{int(stat.st_mtime)}"
    report_path = cache_dir / f"{key}.json"
    fixed_path = cache_dir / f"{key}.pdf"

    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("any_rotated") and fixed_path.exists():
            return fixed_path, report
        if not report.get("any_rotated"):
            return pdf, report

    doc = pdfium.PdfDocument(str(pdf))
    angles: list[int] = []
    pages: list[dict] = []
    try:
        for i in range(len(doc)):
            img = doc[i].render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
            bgr = np.array(img)[:, :, ::-1]  # PIL is RGB; paddle expects BGR
            angle, conf = _classify(bgr)
            if angle % 360 == 0 or conf < MIN_CONFIDENCE:
                angle = 0
            angles.append(angle)
            pages.append({"page": i, "angle": angle, "confidence": round(conf, 4)})
    finally:
        doc.close()

    report = {
        "pdf": str(pdf),
        "pages": pages,
        "any_rotated": any(angles),
        "rotated_pages": [p["page"] for p in pages if p["angle"]],
    }
    report_path.write_text(json.dumps(report, indent=2))

    if not report["any_rotated"]:
        return pdf, report

    from pypdf import PdfReader, PdfWriter

    reader, writer = PdfReader(str(pdf)), PdfWriter()
    for i, page in enumerate(reader.pages):
        if angles[i]:
            page.rotate(-angles[i])  # pypdf rotates clockwise; the label is CCW
        writer.add_page(page)
    with fixed_path.open("wb") as fh:
        writer.write(fh)
    log.info("oriented %s: fixed pages %s", pdf.name, report["rotated_pages"])
    return fixed_path, report
