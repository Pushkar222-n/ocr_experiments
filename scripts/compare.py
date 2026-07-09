#!/usr/bin/env python3
"""Aggregate all outputs/<model>/summary.json into one comparison table."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main():
    rows = []
    for summary in sorted(OUTPUTS.glob("*/summary.json")):
        model = summary.parent.name
        for doc in json.loads(summary.read_text()):
            rows.append({"model": model, **doc})
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
