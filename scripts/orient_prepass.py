"""Rotation-correction pre-pass: classify + fix page rotation for the sample_set pdfs
*before* anything takes the GPU, and cache the result under work/oriented/.

Why this is its own step and not just left to the adapter. `harness.orient_pdf` is
idempotent and cached, so an adapter that calls it (models/chandra/run.py does) already
gets the right answer whenever it runs. But by the time chandra's run.py executes, run.sh
has already started a vLLM that reserved ~85% of the card. Running the PP-LCNet classifier
at *that* moment means standing up a second inference runtime alongside a nearly-full GPU.
`classify_rotation` pins itself to the cpu, which is the real fix -- this pre-pass is the
belt-and-braces one: run.sh invokes it with CUDA_VISIBLE_DEVICES="" before the server
starts, so the classifier could not touch the GPU even if the pin failed, and by the time
the adapter calls orient_pdf the work is already done and it is a pure cache read.

Prints a one-line-per-pdf summary. Exits nonzero only on a real error.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from ocr_harness import list_pdfs, orient_pdf  # noqa: E402


def main():
    names = sys.argv[1:] or None
    for pdf in list_pdfs(names):
        used, report = orient_pdf(pdf)
        flagged = report["flagged_pages"]
        angles = {p["page"]: p["detected_angle"] for p in report["pages"] if p["detected_angle"]}
        status = f"{len(flagged)}/{report['num_pages']} rotated {angles}" if flagged else "clean"
        print(f"[orient] {pdf.stem}: {status} -> {used}")


if __name__ == "__main__":
    main()
