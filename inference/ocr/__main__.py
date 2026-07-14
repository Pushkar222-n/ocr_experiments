"""Entrypoint: OCR a folder (or a zip) of documents with Chandra.

    python -m ocr --input /path/to/docs

Resumable: a finished document is never redone. Kill it any time; re-run the same command.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from . import config
from .discover import find_docs, resolve_input
from .engine import Chandra, server_is_up
from .report import is_done, visible_chars, write_document, write_metrics, write_summary


class TqdmHandler(logging.StreamHandler):
    """Log through tqdm.write so a progress bar is never torn in half by a log line."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(run_name: str, verbose: bool) -> Path:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.LOGS / f"{run_name}_{stamp}.log"

    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setFormatter(logging.Formatter(fmt))
    file_h.setLevel(logging.DEBUG)

    con_h = TqdmHandler()
    con_h.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    con_h.setLevel(logging.DEBUG if verbose else logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers[:] = [file_h, con_h]
    # chandra/httpx are chatty at INFO and would drown the progress bars
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="ocr", description="Chandra OCR over a folder or zip.")
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="folder (searched recursively) or .zip of documents")
    p.add_argument("--name", help="output run name (default: the input folder/zip name)")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                   help=f"pages in flight per batch (default {config.BATCH_SIZE}; "
                        "measured optimum — raising it is slower)")
    p.add_argument("--no-orient", action="store_true",
                   help="skip rotation correction (don't: it is worth +4.8%% on rotated pages)")
    p.add_argument("--images", action="store_true",
                   help="also OCR loose image files (png/jpg/...), not just pdfs")
    p.add_argument("--limit", type=int, help="process at most N documents (smoke tests)")
    p.add_argument("--redo", action="store_true", help="ignore checkpoints and redo everything")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    root, run_name = resolve_input(args.input)
    run_name = args.name or run_name
    run_root = config.OUTPUTS / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(run_name, args.verbose)
    log = logging.getLogger("ocr")

    docs = find_docs(root, include_images=args.images)
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        log.error("no documents found under %s", root)
        return 1

    pending = [d for d in docs if args.redo or not is_done(run_root / d.out_dir_rel, d.stem)]
    done_already = len(docs) - len(pending)

    log.info("input      : %s", root)
    log.info("output     : %s", run_root)
    log.info("log        : %s", log_path)
    log.info("documents  : %d found, %d already done, %d to do",
             len(docs), done_already, len(pending))

    if args.dry_run:
        for d in docs:
            state = "done" if is_done(run_root / d.out_dir_rel, d.stem) else "todo"
            print(f"  [{state}] {d.rel}")
        return 0
    if not pending:
        log.info("nothing to do — everything is already processed")
        return 0

    if not server_is_up():
        log.error("no vLLM server at %s — start it first:  ./serve.sh", config.API_BASE)
        return 2

    started = datetime.now(timezone.utc)
    engine = Chandra(batch_size=args.batch_size)
    orient = None
    if not args.no_orient:
        from .orient import orient_pdf

        orient = orient_pdf

    results_all: list[dict] = []
    total_pages = 0
    t_run = time.perf_counter()

    outer = tqdm(pending, desc="documents", unit="doc", position=0,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for doc in outer:
        outer.set_postfix_str(str(doc.rel)[:48])
        out_dir = run_root / doc.out_dir_rel
        entry: dict = {"document": str(doc.rel), "pages": 0, "seconds": 0.0}
        t0 = time.perf_counter()
        try:
            src = doc.path
            rotated: list[int] = []
            if orient and doc.path.suffix.lower() == ".pdf":
                src, rep = orient(doc.path)
                rotated = rep.get("rotated_pages", [])

            images = engine.load_pages(src)
            entry["pages"] = len(images)

            inner = tqdm(total=len(images), desc="  pages", unit="pg", position=1,
                         leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}")

            def on_batch(n, secs):
                inner.update(n)
                inner.set_postfix_str(f"{secs / max(n, 1):.1f}s/pg")

            results = engine.ocr_pages(images, on_batch=on_batch)
            inner.close()

            written = write_document(out_dir, doc.stem, results,
                                     save_images=config.INCLUDE_IMAGES)
            md = written["markdown"]
            entry.update(
                seconds=round(time.perf_counter() - t0, 2),
                chars=len(md),
                visible_chars=visible_chars(md),
                tokens=written["metadata"]["total_token_count"],
                rotated_pages=rotated,
                errors=sum(1 for r in results if getattr(r, "error", False)),
            )
            entry["seconds_per_page"] = round(entry["seconds"] / max(entry["pages"], 1), 3)
            write_metrics(out_dir, doc.stem, entry)  # written LAST = the done-marker
            total_pages += entry["pages"]
            log.info("%s: %d pages, %.1fs (%.2fs/pg), %s visible chars%s",
                     doc.rel, entry["pages"], entry["seconds"], entry["seconds_per_page"],
                     f"{entry['visible_chars']:,}",
                     f", rotated {rotated}" if rotated else "")
        except Exception as e:  # one bad document must never kill the run
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["seconds"] = round(time.perf_counter() - t0, 2)
            log.exception("FAILED %s", doc.rel)
        results_all.append(entry)
    outer.close()

    # carry already-finished documents into the summary so it describes the whole tree,
    # not just what this invocation happened to do
    for d in docs:
        if d not in pending:
            mp = run_root / d.out_dir_rel / f"{d.stem}.metrics.json"
            if mp.exists():
                import json

                results_all.append(json.loads(mp.read_text()))

    summary = write_summary(run_root, results_all, started)
    wall = time.perf_counter() - t_run

    log.info("=" * 66)
    log.info("done in %.1f min — %d/%d documents, %d pages",
             wall / 60, summary["succeeded"], summary["documents"], summary["pages"])
    if summary["seconds_per_page"]:
        log.info("throughput: %.2f s/page", summary["seconds_per_page"])
    log.info("visible chars: %s", f"{summary['visible_chars']:,}")
    if summary["failed"]:
        log.warning("%d FAILED — see summary.json", summary["failed"])
        for f in summary["failures"]:
            log.warning("  %s: %s", f["document"], f["error"])
    log.info("outputs: %s", run_root)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
