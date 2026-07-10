#!/usr/bin/env python3
"""Aggregate all outputs/<model>/summary.json into one comparison table."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def text_ratio(md_path: Path) -> tuple[int, float] | None:
    """Characters of actual text, with markup stripped.

    `total_chars` is not comparable across these models -- they emit wildly different
    native formats and it counts every byte of markup. Measured on the 7-page printouts:
    paddleocr_vl is 17.2% text (inline CSS on every <td>), lightonocr 58.1%, dots_ocr
    59.5%, unlimited_ocr 43.2%, got_ocr 98.6%. So paddleocr_vl's 62,464 chars vs
    lightonocr's 16,082 is a 5.8x markup artifact, yet paddleocr_vl still extracts more
    real text (10,760 vs 9,349). Neither fact is visible from total_chars alone.

    This is a crude lower bound -- it does not strip LaTeX (got_ocr's mathpix) or
    dots_ocr's/unlimited_ocr's grounding tags, so those are still somewhat overcounted.
    Use it to catch order-of-magnitude markup effects, not to rank models by 3%.
    """
    if not md_path.exists():
        return None
    raw = md_path.read_text(errors="replace")
    vis = len(_WS.sub(" ", _TAG.sub("", raw)).strip())
    # ratio is against the .md file, NOT summary.json's total_chars: that field sums the
    # per-page char counts, while combine() joins pages with "\n\n", so the file is
    # longer and the ratio can exceed 100%.
    return vis, round(100 * vis / len(raw), 1) if raw else 0.0


def main():
    rows = []
    for summary in sorted(OUTPUTS.glob("*/summary.json")):
        model = summary.parent.name
        for doc in json.loads(summary.read_text()):
            row = {"model": model, **doc}
            stem = Path(doc.get("pdf", "")).stem
            got = text_ratio(summary.parent / f"{stem}.md")
            if got is not None:
                row["visible_chars"], row["pct_text"] = got
            rows.append(row)
    if not rows:
        print("no summary.json files found under outputs/ yet", file=sys.stderr)
        return
    cols = ["model", "pdf", "pages", "total_seconds", "seconds_per_page",
            "total_chars", "max_gpu_mem_mb"]
    extra_cols = sorted({k for r in rows for k in r} - set(cols))
    cols += extra_cols
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    (OUTPUTS / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUTPUTS/'comparison.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
