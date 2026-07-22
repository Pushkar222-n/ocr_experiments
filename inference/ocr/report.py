"""Writing a document's outputs, and the run's metrics.

Output layout mirrors the input tree exactly, then one folder per document:

    outputs/<run>/<same/sub/dirs>/<stem>/
        <stem>.md               markdown
        <stem>.html             chandra's native html
        <stem>.metadata.json    per-page token counts + page boxes (chandra's own)
        <stem>.metrics.json     our metrics for this document  <-- also the done-marker
        *.webp                  extracted images (siblings of the .md, as the links assume)

The per-document folder is not decoration: chandra's markdown links its extracted images by
bare filename, so they must sit beside the .md or every image link breaks.
"""
import csv
import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Page boundary marker, written before every page of the merged .md/.html.
#
# Why it exists: chandra returns one result per page, but the merged document used to join
# them with a bare "\n\n" — so the page boundaries were gone the moment the file was written,
# and no downstream step could route per page. A page number OCR'd from the page itself
# ("Page 1") is NOT a substitute: it is document content, appears only when the page happens
# to print one, and restarts at 1 for every attachment inside a section.
#
# It is an html comment, so it is invisible in rendered markdown and legal in the .html too,
# and it is 1-based to match both `metadata["pages"][]["page_num"]` and how a human counts.
PAGE_MARKER = "<!-- page: {n} -->"
_PAGE_MARKER_RE = re.compile(r"^<!-- page: (\d+) -->$", re.MULTILINE)


def _join_pages(parts) -> str:
    """Merge per-page text, each page preceded by its marker.

    Every page gets a marker, including the first, so a consumer can split on the marker
    alone without special-casing the head of the file.
    """
    return "\n\n".join(
        f"{PAGE_MARKER.format(n=n)}\n\n{text}" for n, text in enumerate(parts, 1)
    )


def split_pages(merged: str) -> list[str]:
    """Inverse of `_join_pages`: recover per-page text from a merged document.

    Returns [] for text written before markers existed, rather than guessing at boundaries —
    a wrong split is worse than an honest "unknown", because it would route the wrong page.
    """
    hits = list(_PAGE_MARKER_RE.finditer(merged))
    if not hits:
        return []
    bounds = [m.end() for m in hits] + [len(merged)]
    return [merged[bounds[i]:hits[i + 1].start() if i + 1 < len(hits) else len(merged)].strip()
            for i in range(len(hits))]


def visible_chars(markdown: str) -> int:
    """Characters of real text, markup stripped. Raw length is not comparable across
    documents — an html-table-heavy page inflates it — so we record both."""
    return len(_WS.sub(" ", _TAG.sub("", markdown)).strip())


def is_done(out_dir: Path, stem: str) -> bool:
    """Per-document checkpoint. metrics.json is written LAST, so its presence means the whole
    document finished — a run killed mid-document redoes only that document."""
    return (out_dir / f"{stem}.metrics.json").exists() and (out_dir / f"{stem}.md").exists()


def write_document(out_dir: Path, stem: str, results: list, *, save_images: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = _join_pages(r.markdown for r in results)
    html = _join_pages(r.html for r in results)
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{stem}.html").write_text(html, encoding="utf-8")

    if save_images:
        for r in results:
            for name, img in (r.images or {}).items():
                try:
                    img.save(out_dir / name)
                except Exception as e:  # a bad crop must not lose the page's text
                    log.warning("could not save image %s: %s", name, e)

    metadata = {
        "file_name": f"{stem}.pdf",
        "num_pages": len(results),
        "total_token_count": sum(r.token_count for r in results),
        "pages": [
            {
                # 1-based, matching the `<!-- page: N -->` marker in the .md/.html so the two
                # can be joined without an off-by-one. Nothing consumed this field when it was
                # changed from 0-based.
                "page_num": n,
                "page_box": r.page_box,
                "token_count": r.token_count,
                "num_chunks": len(r.chunks),
                "num_images": len(r.images or {}),
            }
            for n, r in enumerate(results, 1)
        ],
    }
    (out_dir / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"markdown": md, "metadata": metadata}


def write_metrics(out_dir: Path, stem: str, doc: dict) -> None:
    """Written last — it is the done-marker."""
    (out_dir / f"{stem}.metrics.json").write_text(json.dumps(doc, indent=2))


def write_summary(run_root: Path, docs: list[dict], started: datetime) -> dict:
    ok = [d for d in docs if not d.get("error")]
    failed = [d for d in docs if d.get("error")]
    pages = sum(d.get("pages", 0) for d in ok)
    seconds = sum(d.get("seconds", 0.0) for d in ok)
    summary = {
        "run": run_root.name,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "documents": len(docs),
        "succeeded": len(ok),
        "failed": len(failed),
        "pages": pages,
        "total_seconds": round(seconds, 2),
        "seconds_per_page": round(seconds / pages, 3) if pages else None,
        "total_chars": sum(d.get("chars", 0) for d in ok),
        "visible_chars": sum(d.get("visible_chars", 0) for d in ok),
        "rotated_pages": sum(len(d.get("rotated_pages", [])) for d in ok),
        "failures": [{"document": d["document"], "error": d["error"]} for d in failed],
        "docs": docs,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["document", "pages", "seconds", "seconds_per_page", "chars", "visible_chars",
            "tokens", "rotated_pages", "error"]
    with (run_root / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in docs:
            row = dict(d)
            row["rotated_pages"] = len(d.get("rotated_pages", []) or [])
            w.writerow(row)
    return summary
