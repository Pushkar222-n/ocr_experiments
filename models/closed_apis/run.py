"""Closed / paid document-parsing APIs, run over the same sample_set as the open models.

Four providers, each on a *balanced* (mid) tier — not their premium option:
  mistral    -> Mistral OCR            (mistral-ocr-latest)
  datalab    -> Datalab Marker         (hosted, use_llm=false = the non-LLM base tier)
  llamaparse -> LlamaParse             (parse_page_with_agent = the "Balanced" preset)
  landing_ai -> Landing AI ADE         (dpt-2-latest)

All are hit over plain REST. Outputs are kept *separate* from the open models, under
outputs/closed/<provider>/, with the same file shapes so the frontend and compare
tooling can read them: <stem>.md, <stem>.native.json (full API response), <stem>.metrics.json,
and summary.json. Per-pdf checkpoint: a finished <stem>.md is not re-fetched (these cost money).

Usage + cost: each provider reports billed pages and/or credits; we store the raw usage
and a computed cost_usd from the rate table below. RATES ARE BUILD-TIME ESTIMATES
(2026-07) AND DRIFT — verify against current pricing before trusting the cost column.
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

# USD per unit. Best-effort, build-time (2026-07); VERIFY before quoting. cost_usd is
# computed from whichever unit the provider actually bills (pages or credits).
# Rates read off each vendor's public pricing page on 2026-07-12. `current` is the tier
# this runner actually uses (the balanced/mid option); the other tiers are kept so the
# cost of moving up can be predicted without re-running anything (`run.py prices`).
#
# Credit-billed providers: cost is credits x CREDIT_USD. We prefer the credits the API
# actually reported; if it reports 0 (LlamaParse's free tier absorbs the run), we fall
# back to the tier's list credits/page so the column isn't a misleading $0.00.
CREDIT_USD = {
    "llamaparse": 0.00125,  # $1.25 / 1000 credits
    "landing_ai": 0.01,     # Explore: $1 = 100 credits (Team: $1 = 110 credits -> 0.00909)
}

PRICING = {
    "mistral": {
        "current": "ocr",
        "tiers": {
            "ocr":         {"per_page": 0.004, "label": "Mistral OCR (mistral-ocr-latest) — $4/1k"},
            "document_ai": {"per_page": 0.005, "label": "Document AI (premium) — $5/1k"},
        },
    },
    "datalab": {
        "current": "marker_base",
        "tiers": {
            # NOT published anywhere fetchable — the pricing page is client-rendered and
            # there is no usage endpoint. $6/1k for High Accuracy IS confirmed (their blog),
            # so treat the base rate as the one soft number in this table.
            "marker_base":   {"per_page": 0.003, "unconfirmed": True,
                              "label": "Marker base, use_llm=false — ~$3/1k"},
            "high_accuracy": {"per_page": 0.006,
                              "label": "High Accuracy / use_llm=true / page_schema — $6/1k"},
        },
    },
    "llamaparse": {
        "current": "agentic",
        "tiers": {
            "fast":           {"credits_per_page": 1,  "label": "Fast / parse_page_without_llm — $1.25/1k"},
            "cost_effective": {"credits_per_page": 3,  "label": "Cost-effective / parse_page_with_llm — $3.75/1k"},
            "agentic":        {"credits_per_page": 10, "label": "Agentic (Balanced) / parse_page_with_agent — $12.50/1k"},
            "agentic_plus":   {"credits_per_page": 45, "label": "Agentic Plus (premium, sonnet) — $56.25/1k"},
        },
    },
    "landing_ai": {
        "current": "dpt2",
        "tiers": {
            # measured: 204 credits / 68 pages = exactly 3 credits/page
            "dpt2": {"credits_per_page": 3, "label": "ADE dpt-2-latest — 3 cr/pg = $30/1k (Explore)"},
        },
    },
}


def tier_rate(provider: str, tier: str | None = None) -> float:
    """USD per page for a provider's tier."""
    p = PRICING[provider]
    t = p["tiers"][tier or p["current"]]
    if "per_page" in t:
        return t["per_page"]
    return t["credits_per_page"] * CREDIT_USD[provider]


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1])
    raise RuntimeError(f"pdfinfo: no page count for {pdf}")


# ---------------------------------------------------------------- provider clients
# Each returns (markdown: str, native: dict, usage: dict). usage carries whatever the
# API billed: billed_pages and/or credits.

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
    billed = (data.get("usage_info") or {}).get("pages_processed")
    return md, data, {"billed_pages": billed}


def run_datalab(pdf: Path) -> tuple[str, dict, dict]:
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
    sub = r.json()
    check_url = sub.get("request_check_url")
    if not check_url:
        raise RuntimeError(f"datalab: no check_url in {sub}")
    for _ in range(300):  # poll up to ~10 min
        time.sleep(2)
        pr = requests.get(check_url, headers={"X-Api-Key": key}, timeout=60).json()
        if pr.get("status") == "complete":
            if not pr.get("success", True):
                raise RuntimeError(f"datalab failed: {pr.get('error')}")
            return pr.get("markdown", ""), pr, {"billed_pages": pr.get("page_count")}
    raise TimeoutError("datalab: polling timed out")


def run_llamaparse(pdf: Path) -> tuple[str, dict, dict]:
    key = os.environ["LLAMAPARSE_API_KEY"]
    h = {"Authorization": f"Bearer {key}", "accept": "application/json"}
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://api.cloud.llamaindex.ai/api/v1/parsing/upload",
            headers=h,
            files={"file": (pdf.name, fh, "application/pdf")},
            data={"parse_mode": "parse_page_with_agent"},  # the "Balanced" preset
            timeout=120,
        )
    r.raise_for_status()
    job = r.json()["id"]
    base = f"https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job}"
    for _ in range(300):
        time.sleep(2)
        st = requests.get(base, headers=h, timeout=60).json()
        if st.get("status") == "SUCCESS":
            break
        if st.get("status") in ("ERROR", "CANCELED"):
            raise RuntimeError(f"llamaparse failed: {st}")
    else:
        raise TimeoutError("llamaparse: polling timed out")
    md = requests.get(f"{base}/result/markdown", headers=h, timeout=120).json().get("markdown", "")
    native = requests.get(f"{base}/result/json", headers=h, timeout=120).json()
    meta = native.get("job_metadata") or {}
    return md, native, {"credits": meta.get("job_credits_usage") or meta.get("credits_used"),
                        "billed_pages": meta.get("job_pages") or meta.get("job_pages_pending")}


def run_landing_ai(pdf: Path) -> tuple[str, dict, dict]:
    key = os.environ["LANDING_API_KEY"]
    with pdf.open("rb") as fh:
        r = requests.post(
            "https://api.va.landing.ai/v1/ade/parse",
            headers={"Authorization": f"Bearer {key}"},
            files={"document": (pdf.name, fh, "application/pdf")},
            data={"model": "dpt-2-latest"},
            timeout=600,
        )
    r.raise_for_status()
    data = r.json()
    md = data.get("markdown", "")
    meta = data.get("metadata") or {}
    return md, data, {"credits": meta.get("credit_usage"),
                      "billed_pages": meta.get("page_count")}


PROVIDERS = {
    "mistral": run_mistral,
    "datalab": run_datalab,
    "llamaparse": run_llamaparse,
    "landing_ai": run_landing_ai,
}


def compute_cost(provider: str, usage: dict, our_pages: int, tier: str | None = None) -> float:
    credits = usage.get("credits")
    if credits and provider in CREDIT_USD:  # metered credits the API actually reported
        return round(credits * CREDIT_USD[provider], 5)
    pages = usage.get("billed_pages") or our_pages
    return round(pages * tier_rate(provider, tier), 5)


def process_one(provider: str, pdf: Path, out_root: Path) -> dict:
    final_md = out_root / f"{pdf.stem}.md"
    metrics_path = out_root / f"{pdf.stem}.metrics.json"
    if final_md.exists() and metrics_path.exists():
        print(f"  skip {pdf.stem} (done)", flush=True)
        return json.loads(metrics_path.read_text())
    n_pages = pdf_page_count(pdf)
    t0 = time.perf_counter()
    md, native, usage = PROVIDERS[provider](pdf)
    dt = time.perf_counter() - t0
    final_md.write_text(md)
    (out_root / f"{pdf.stem}.native.json").write_text(json.dumps(native, indent=2)[:5_000_000])
    doc = {
        "pdf": pdf.name, "pages": n_pages,
        "total_seconds": round(dt, 2), "seconds_per_page": round(dt / max(n_pages, 1), 3),
        "total_chars": len(md),
        "billed_pages": usage.get("billed_pages"),
        "credits": usage.get("credits"),
        "cost_usd": compute_cost(provider, usage, n_pages),
        "provider": provider,
    }
    metrics_path.write_text(json.dumps(doc, indent=2))
    print(f"  {pdf.stem}: {n_pages}p {dt:.1f}s {len(md)}chars cost=${doc['cost_usd']}", flush=True)
    return doc


def run_provider(provider: str, pdfs: list[Path], concurrency: int, smoke: bool):
    out_root = (OUT_BASE / "_smoke" / provider) if smoke else (OUT_BASE / provider)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[{provider}] {len(pdfs)} pdfs, concurrency={concurrency} -> {out_root}", flush=True)
    docs = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(process_one, provider, pdf, out_root): pdf for pdf in pdfs}
        for fut in as_completed(futs):
            pdf = futs[fut]
            try:
                docs.append(fut.result())
            except Exception as e:
                print(f"  ERROR {pdf.stem}: {type(e).__name__}: {e}", flush=True)
    docs.sort(key=lambda d: d["pdf"])
    (out_root / "summary.json").write_text(json.dumps(docs, indent=2))
    print(f"[{provider}] wrote {out_root/'summary.json'} ({len(docs)} pdfs)", flush=True)


def reprice():
    """Recompute cost_usd in every stored metrics.json/summary.json from the usage the APIs
    already reported. Rates change; re-running the APIs to update a cost column would mean
    paying for the same pages twice. This never touches the network."""
    for prov in PROVIDERS:
        root = OUT_BASE / prov
        if not root.exists():
            continue
        docs = []
        for mp in sorted(root.glob("*.metrics.json")):
            d = json.loads(mp.read_text())
            usage = {"billed_pages": d.get("billed_pages"), "credits": d.get("credits")}
            old = d.get("cost_usd")
            d["cost_usd"] = compute_cost(prov, usage, d["pages"])
            d["rate_per_page"] = tier_rate(prov)
            d["tier"] = PRICING[prov]["current"]
            mp.write_text(json.dumps(d, indent=2))
            docs.append(d)
            print(f"  {prov}/{d['pdf']}: ${old} -> ${d['cost_usd']}")
        docs.sort(key=lambda d: d["pdf"])
        (root / "summary.json").write_text(json.dumps(docs, indent=2))
    print("repriced (no API calls made)")


def prices(pages: int = 68):
    """Print every tier's rate and what the sample_set would cost on it."""
    print(f"{'provider':11} {'tier':16} {'$/1k pages':>11} {f'{pages}p':>8}  rate source")
    print("-" * 78)
    for prov, p in PRICING.items():
        for name, t in p["tiers"].items():
            rate = tier_rate(prov, name)
            cur = " <- RUN" if name == p["current"] else ""
            flag = " (UNCONFIRMED)" if t.get("unconfirmed") else ""
            print(f"{prov:11} {name:16} {rate*1000:>10.2f} {rate*pages:>8.3f}  {t['label']}{flag}{cur}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=[*PROVIDERS, "all", "reprice", "prices"])
    ap.add_argument("--pdfs", nargs="*", help="pdf stems (default: all 5)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="write to outputs/closed/_smoke/")
    args = ap.parse_args()

    if args.provider == "prices":
        return prices()
    if args.provider == "reprice":
        return reprice()

    pdfs = sorted(SAMPLE_SET.glob("*.pdf"))
    if args.pdfs:
        want = {Path(n).stem for n in args.pdfs}
        pdfs = [p for p in pdfs if p.stem in want]
    if not pdfs:
        sys.exit("no pdfs matched")
    providers = list(PROVIDERS) if args.provider == "all" else [args.provider]
    for prov in providers:
        run_provider(prov, pdfs, args.concurrency, args.smoke)


if __name__ == "__main__":
    main()
