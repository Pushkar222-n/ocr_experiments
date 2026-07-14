"""Closed / paid document-parsing APIs, run over the same sample_set as the open models.

A "run" is a provider + tier combination, and each run gets its OWN output directory under
outputs/closed/<run>/ so tiers never overwrite each other:

  mistral                   Mistral OCR (mistral-ocr-latest)
  datalab                   LEGACY /api/v1/marker endpoint (deprecated; kept as an artifact)
  datalab_fast              /api/v1/convert  mode=fast
  datalab_balanced          /api/v1/convert  mode=balanced
  datalab_accurate          /api/v1/convert  mode=accurate
  llamaparse                parse_page_with_agent + gemini-2.5-flash   ("Agentic" / Balanced)
  llamaparse_agentic_plus   parse_page_with_agent + anthropic-sonnet-4.0 ("Agentic Plus")
  landing_ai                ADE dpt-2-latest

Datalab's /api/v1/convert is the CURRENT api (the /marker endpoint we first used is the old
one) and it *reports the real cost* in `cost_breakdown.final_cost_cents` plus a
`parse_quality_score`. Where an API reports actual cost or metered credits we use that;
only where it reports nothing do we fall back to the published list rate.

Per-pdf checkpoint: a finished <stem>.md is not re-fetched (these cost money).
`run.py prices` prints the rate table; `run.py reprice` recomputes stored costs offline.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SET = ROOT / "data" / "Evaluation set" / "sample_set"
OUT_BASE = ROOT / "outputs" / "closed"


def load_env(path: Path):
    """Minimal .env loader (avoids a python-dotenv dep). KEY=VALUE per line."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")

# ------------------------------------------------------------------------ pricing
# Rates read off each vendor's public pricing page on 2026-07-12.
#   $1.25 / 1000 credits (LlamaParse); $1 = 100 credits (Landing AI "Explore" plan).
# Landing AI does NOT publish a credits-per-page figure — the 3 cr/pg below is MEASURED
# from the credit_usage the API itself returned (204 credits / 68 pages, exactly 3.0).
# Datalab's /convert modes report their true cost per request, so no rate is guessed there.
CREDIT_USD = {
    "llamaparse": 0.00125,  # $1.25 / 1000 credits
    "landing_ai": 0.01,     # Explore: $1 = 100 credits (Team: $1 = 110 -> $0.00909)
}

# per-run rate info. cost_from_api=True means the provider returns the real charge and the
# list rate is never used. unconfirmed=True means we could not verify the rate anywhere.
PRICING = {
    "mistral":                 {"per_page": 0.004,
                                "label": "Mistral OCR (mistral-ocr-latest) — $4/1k"},
    # The legacy /marker response ALSO carries cost_breakdown — we just weren't reading it.
    # Its bill is byte-identical to mode=balanced (13/2/5/6/3 cents on the same 5 pdfs), so
    # the old "~$3/1k, unconfirmed" guess is dead: the real rate is $4.26/1k, reported.
    "datalab":                 {"cost_from_api": True, "provider": "datalab",
                                "label": "LEGACY /marker — bills identically to balanced"},
    "datalab_fast":            {"cost_from_api": True, "provider": "datalab",
                                "label": "/convert mode=fast — cost reported by API"},
    "datalab_balanced":        {"cost_from_api": True, "provider": "datalab",
                                "label": "/convert mode=balanced — cost reported by API"},
    "datalab_accurate":        {"cost_from_api": True, "provider": "datalab",
                                "label": "/convert mode=accurate — cost reported by API"},
    "llamaparse":              {"credits_per_page": 10, "provider": "llamaparse",
                                "label": "tier=agentic (Balanced) — 10 cr/pg = $12.50/1k"},
    "llamaparse_agentic_plus": {"credits_per_page": 45, "provider": "llamaparse",
                                "label": "tier=agentic_plus (premium) — 45 cr/pg = $56.25/1k"},
    "landing_ai":              {"credits_per_page": 3, "provider": "landing_ai",
                                "label": "ADE dpt-2-latest — 3 cr/pg (MEASURED) = $30/1k"},
}
# tiers we did not run, kept so an upgrade can be costed without running it
UNRUN_TIERS = {
    "mistral / Document AI":                  0.005,
    "llamaparse / Fast (1 cr/pg)":            1 * CREDIT_USD["llamaparse"],
    "llamaparse / Cost-effective (3 cr/pg)":  3 * CREDIT_USD["llamaparse"],
    "landing_ai / Team plan (3 cr/pg @ .00909)": 3 * 0.00909,
}


def list_rate(run: str) -> float | None:
    """USD per page from the published list rate (None when the API reports true cost)."""
    p = PRICING[run]
    if p.get("cost_from_api"):
        return None
    if "per_page" in p:
        return p["per_page"]
    return p["credits_per_page"] * CREDIT_USD[p["provider"]]


def compute_cost(run: str, usage: dict, our_pages: int) -> float:
    """Prefer what the provider actually charged, then metered credits, then list rate."""
    if usage.get("cost_usd_api") is not None:      # datalab /convert: real charge
        return round(usage["cost_usd_api"], 5)
    p = PRICING[run]
    credits = usage.get("credits")
    if credits and p.get("provider") in CREDIT_USD:  # metered credits (landing_ai)
        return round(credits * CREDIT_USD[p["provider"]], 5)
    pages = usage.get("billed_pages") or our_pages
    return round(pages * (list_rate(run) or 0), 5)


def cost_source(run: str, usage: dict) -> str:
    if usage.get("cost_usd_api") is not None:
        return "api_reported"
    if usage.get("credits") and PRICING[run].get("provider") in CREDIT_USD:
        return "metered_credits"
    return "list_rate_estimate"


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1])
    raise RuntimeError(f"pdfinfo: no page count for {pdf}")


# ---------------------------------------------------------------- provider clients
# Each returns (markdown, native_response, usage). usage may carry billed_pages, credits,
# cost_usd_api (the real charge, when the API reports one) and quality.

def run_mistral(pdf: Path) -> tuple[str, dict, dict]:
    key = os.environ["MISTRAL_API_KEY"]
    b64 = base64.b64encode(pdf.read_bytes()).decode()
    r = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url",
                         "document_url": f"data:application/pdf;base64,{b64}"},
            "include_image_base64": False,
        },
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    md = "\n\n".join(p.get("markdown", "") for p in data.get("pages", []))
    return md, data, {"billed_pages": (data.get("usage_info") or {}).get("pages_processed")}


def _datalab_poll(check_url: str, key: str, tries: int = 300) -> dict:
    for _ in range(tries):
        time.sleep(2)
        pr = requests.get(check_url, headers={"X-Api-Key": key}, timeout=60).json()
        if pr.get("status") == "complete":
            if not pr.get("success", True):
                raise RuntimeError(f"datalab failed: {pr.get('error')}")
            return pr
    raise TimeoutError("datalab: polling timed out")


def run_datalab_marker(pdf: Path) -> tuple[str, dict, dict]:
    """The OLD /api/v1/marker endpoint. Superseded by /api/v1/convert (see run_datalab_convert);
    kept only so the original run stays reproducible. It reports no cost."""
    key = os.environ["DATALAB_API_KEY"]
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://www.datalab.to/api/v1/marker",
            headers={"X-Api-Key": key},
            files={"file": (pdf.name, fh, "application/pdf")},
            data={"output_format": "markdown", "use_llm": "false",
                  "force_ocr": "false", "paginate": "false"},
            timeout=120,
        )
    r.raise_for_status()
    pr = _datalab_poll(r.json()["request_check_url"], key)
    return pr.get("markdown", ""), pr, {"billed_pages": pr.get("page_count")}


def run_datalab_convert(pdf: Path, mode: str = "balanced") -> tuple[str, dict, dict]:
    """Current Datalab API. mode = fast | balanced | accurate.

    The response carries the *actual* charge (cost_breakdown.final_cost_cents, rounded up to
    the nearest cent per request) and a parse_quality_score (0-5), so cost here is measured,
    not estimated."""
    key = os.environ["DATALAB_API_KEY"]
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://www.datalab.to/api/v1/convert",
            headers={"X-Api-Key": key},
            files={"file": (pdf.name, fh, "application/pdf")},
            data={"output_format": "markdown", "mode": mode},
            timeout=120,
        )
    r.raise_for_status()
    pr = _datalab_poll(r.json()["request_check_url"], key)
    cents = (pr.get("cost_breakdown") or {}).get("final_cost_cents")
    if cents is None and pr.get("total_cost") is not None:
        cents = pr["total_cost"]
    return pr.get("markdown", ""), pr, {
        "billed_pages": pr.get("page_count"),
        "cost_usd_api": (cents / 100) if cents is not None else None,
        "quality_score": pr.get("parse_quality_score"),
        "mode": mode,
    }


def run_llamaparse(pdf: Path, tier: str = "agentic", version: str = "latest") -> tuple[str, dict, dict]:
    """Tier-based parse: cost_effective (3 cr/pg) | agentic (10) | agentic_plus (45).

    Two traps here, both hit live:
      - Pinning a `model` is dead. `anthropic-sonnet-4.0` (still shown in the Agentic Plus
        docs) is RETIRED and 422s. The API itself says to migrate to tiers.
      - A `tier` REQUIRES a `version` or the job fails async with MISSING_VERSION_FOR_TIER —
        it accepts the upload, then errors during parsing. "latest" tracks the newest;
        pin a date (e.g. 2026-01-08) if you need a frozen config."""
    key = os.environ["LLAMAPARSE_API_KEY"]
    h = {"Authorization": f"Bearer {key}", "accept": "application/json"}
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://api.cloud.llamaindex.ai/api/v1/parsing/upload",
            headers=h,
            files={"file": (pdf.name, fh, "application/pdf")},
            data={"tier": tier, "version": version},
            timeout=120,
        )
    r.raise_for_status()
    job = r.json()["id"]
    base = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job}"
    for _ in range(600):  # sonnet agent mode is slow: allow ~20 min
        time.sleep(2)
        st = requests.get(base, headers=h, timeout=60).json()
        if st.get("status") == "SUCCESS":
            break
        if st.get("status") in ("ERROR", "CANCELED"):
            raise RuntimeError(f"llamaparse failed: {st}")
    else:
        raise TimeoutError("llamaparse: polling timed out")
    md = requests.get(f"{base}/result/markdown", headers=h, timeout=180).json().get("markdown", "")
    native = requests.get(f"{base}/result/json", headers=h, timeout=180).json()
    meta = native.get("job_metadata") or {}
    return md, native, {
        "credits": meta.get("job_credits_usage") or meta.get("credits_used"),
        "billed_pages": meta.get("job_pages"),
        "tier": tier,
    }


def run_landing_ai(pdf: Path) -> tuple[str, dict, dict]:
    key = os.environ["LANDING_API_KEY"]
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://api.va.landing.ai/v1/ade/parse",
            headers={"Authorization": f"Bearer {key}"},
            files={"document": (pdf.name, fh, "application/pdf")},
            data={"model": "dpt-2-latest"},
            timeout=900,
        )
    r.raise_for_status()
    data = r.json()
    meta = data.get("metadata") or {}
    return data.get("markdown", ""), data, {
        "credits": meta.get("credit_usage"),      # metered — this is the real bill unit
        "billed_pages": meta.get("page_count"),
    }


# name -> (client fn, kwargs). The name is also the output directory.
RUNS = {
    "mistral":                 (run_mistral, {}),
    "datalab":                 (run_datalab_marker, {}),                       # legacy endpoint
    "datalab_fast":            (run_datalab_convert, {"mode": "fast"}),
    "datalab_balanced":        (run_datalab_convert, {"mode": "balanced"}),
    "datalab_accurate":        (run_datalab_convert, {"mode": "accurate"}),
    "llamaparse":              (run_llamaparse, {"tier": "agentic"}),
    "llamaparse_agentic_plus": (run_llamaparse, {"tier": "agentic_plus"}),
    "landing_ai":              (run_landing_ai, {}),
}


def process_one(run: str, pdf: Path, out_root: Path) -> dict:
    final_md = out_root / f"{pdf.stem}.md"
    metrics_path = out_root / f"{pdf.stem}.metrics.json"
    if final_md.exists() and metrics_path.exists():
        print(f"  skip {pdf.stem} (done)", flush=True)
        return json.loads(metrics_path.read_text())
    fn, kw = RUNS[run]
    n_pages = pdf_page_count(pdf)
    t0 = time.perf_counter()
    md, native, usage = fn(pdf, **kw)
    dt = time.perf_counter() - t0
    final_md.write_text(md)
    (out_root / f"{pdf.stem}.native.json").write_text(json.dumps(native, indent=2)[:5_000_000])
    doc = {
        "pdf": pdf.name, "pages": n_pages,
        "total_seconds": round(dt, 2), "seconds_per_page": round(dt / max(n_pages, 1), 3),
        "total_chars": len(md),
        "billed_pages": usage.get("billed_pages"),
        "credits": usage.get("credits"),
        "cost_usd": compute_cost(run, usage, n_pages),
        "cost_source": cost_source(run, usage),
        "quality_score": usage.get("quality_score"),
        "run": run,
    }
    metrics_path.write_text(json.dumps(doc, indent=2))
    print(f"  {pdf.stem}: {n_pages}p {dt:.1f}s {len(md)}chars "
          f"${doc['cost_usd']} ({doc['cost_source']})", flush=True)
    return doc


def run_one(run: str, pdfs: list[Path], concurrency: int, smoke: bool):
    out_root = (OUT_BASE / "_smoke" / run) if smoke else (OUT_BASE / run)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[{run}] {len(pdfs)} pdfs, concurrency={concurrency} -> {out_root}", flush=True)
    docs = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(process_one, run, pdf, out_root): pdf for pdf in pdfs}
        for fut in as_completed(futs):
            try:
                docs.append(fut.result())
            except Exception as e:
                print(f"  ERROR {futs[fut].stem}: {type(e).__name__}: {e}", flush=True)
    docs.sort(key=lambda d: d["pdf"])
    (out_root / "summary.json").write_text(json.dumps(docs, indent=2))
    total = sum(d.get("cost_usd") or 0 for d in docs)
    print(f"[{run}] wrote summary.json ({len(docs)} pdfs, ${total:.3f})", flush=True)


def reprice():
    """Recompute cost_usd in stored metrics from usage the APIs already reported.
    Never touches the network — refreshing a cost column must not mean paying twice."""
    for run in RUNS:
        root = OUT_BASE / run
        if not root.exists():
            continue
        docs = []
        for mp in sorted(root.glob("*.metrics.json")):
            d = json.loads(mp.read_text())
            # recover the API-reported cost from the stored native response if present
            cost_api = None
            nat = root / f"{Path(d['pdf']).stem}.native.json"
            if nat.exists():
                try:
                    n = json.loads(nat.read_text())
                    cents = (n.get("cost_breakdown") or {}).get("final_cost_cents")
                    cost_api = (cents / 100) if cents is not None else None
                except Exception:
                    pass
            usage = {"billed_pages": d.get("billed_pages"), "credits": d.get("credits"),
                     "cost_usd_api": cost_api}
            old = d.get("cost_usd")
            d["cost_usd"] = compute_cost(run, usage, d["pages"])
            d["cost_source"] = cost_source(run, usage)
            d["run"] = run
            mp.write_text(json.dumps(d, indent=2))
            docs.append(d)
            if old != d["cost_usd"]:
                print(f"  {run}/{d['pdf']}: ${old} -> ${d['cost_usd']}")
        docs.sort(key=lambda d: d["pdf"])
        (root / "summary.json").write_text(json.dumps(docs, indent=2))
    print("repriced (no API calls made)")


def prices(pages: int = 68):
    """Print the rate table and dump it to outputs/closed/pricing.json for later analysis."""
    print(f"{'run':25} {'$/1k pages':>11} {f'{pages}p':>8}  source / label")
    print("-" * 100)
    table = {}
    for run, p in PRICING.items():
        rate = list_rate(run)
        # if we already have a real run, show the observed $/page from what was charged
        obs = None
        s = OUT_BASE / run / "summary.json"
        if s.exists():
            docs = json.loads(s.read_text())
            tot_c = sum(d.get("cost_usd") or 0 for d in docs)
            tot_p = sum(d["pages"] for d in docs)
            if tot_p:
                obs = tot_c / tot_p
        shown = obs if obs is not None else rate
        flag = " (UNCONFIRMED)" if p.get("unconfirmed") else ""
        src = "API-reported" if p.get("cost_from_api") else "list rate"
        if obs is not None:
            src += f", observed over {tot_p}p"
        elif p.get("cost_from_api"):
            src = "NOT RUN (cost only knowable by running it — the API reports it)"
        if shown is None:
            print(f"{run:25} {'—':>10} {'—':>8}  [{src}] {p['label']}")
            continue
        print(f"{run:25} {shown*1000:>10.2f} {shown*pages:>8.3f}  [{src}] {p['label']}{flag}")
        table[run] = {"label": p["label"], "list_rate_per_page": rate,
                      "observed_rate_per_page": obs, "cost_source": src,
                      "unconfirmed": p.get("unconfirmed", False),
                      "usd_per_1k_pages": round((shown or 0) * 1000, 4),
                      f"cost_{pages}_pages": round((shown or 0) * pages, 4)}
    print("\nTiers NOT run (cost if you upgrade):")
    for name, rate in UNRUN_TIERS.items():
        print(f"  {name:45} ${rate*1000:>6.2f}/1k   ${rate*pages:.3f} / {pages}p")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "pricing.json").write_text(json.dumps({
        "verified": "2026-07-12",
        "credit_usd": CREDIT_USD,
        "notes": {
            "datalab_convert": "cost is reported by the API (cost_breakdown.final_cost_cents), "
                               "rounded up to the nearest cent per request — not estimated. "
                               "/api/v1/convert (mode=fast|balanced|accurate) is the CURRENT api; "
                               "/api/v1/marker is the deprecated one.",
            "datalab_legacy_marker": "the old /api/v1/marker response ALSO carries cost_breakdown "
                                     "(we just weren't reading it), and bills IDENTICALLY to "
                                     "mode=balanced: 13/2/5/6/3 cents on the same 5 pdfs. So there "
                                     "is no unconfirmed Datalab rate — it is $4.26/1k, measured.",
            "landing_ai": "3 credits/page is MEASURED from the API's own credit_usage "
                          "(204 credits / 68 pages), not a documented per-page rate. "
                          "$1 = 100 credits on the Explore plan.",
            "llamaparse": "free tier (10k credits/mo) absorbed the runs, so the API reports "
                          "0 credits; cost shown is the published list rate.",
        },
        "runs": table,
        "unrun_tiers_usd_per_page": UNRUN_TIERS,
    }, indent=2))
    print(f"\nwrote {OUT_BASE/'pricing.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=[*RUNS, "all", "reprice", "prices"])
    ap.add_argument("--pdfs", nargs="*", help="pdf stems (default: all 5)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="write to outputs/closed/_smoke/")
    args = ap.parse_args()

    if args.run == "prices":
        return prices()
    if args.run == "reprice":
        return reprice()

    pdfs = sorted(SAMPLE_SET.glob("*.pdf"))
    if args.pdfs:
        want = {Path(n).stem for n in args.pdfs}
        pdfs = [p for p in pdfs if p.stem in want]
    if not pdfs:
        sys.exit("no pdfs matched")
    for run in (list(RUNS) if args.run == "all" else [args.run]):
        run_one(run, pdfs, args.concurrency, args.smoke)


if __name__ == "__main__":
    main()
