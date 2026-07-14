# Chandra OCR — operator runbook

Everything needed to run chandra on a fresh pod, use it, and get the VRAM/disk back.
Chandra is the model this project settled on. `CLAUDE.md` has the benchmark reasoning;
this file is just *how to operate it*.

## What chandra actually is (understand this and the rest is obvious)

**Two processes, not one:**

1. **A vLLM server** — holds the weights (8.61 GiB) in VRAM, exposes an OpenAI-compatible
   endpoint on `http://127.0.0.1:8200/v1`. Slow to start (~7 min: weights + torch.compile +
   CUDA-graph capture), then hot forever.
2. **The `chandra` CLI** — an **HTTP client**. It rasterizes pdf pages, resizes them (max
   3072x2048), base64s them, POSTs to that endpoint, and assembles markdown + html.

The CLI is cheap and disposable. The server is expensive and should be kept alive. That is
the entire design of the setup below.

(Chandra's stock launcher is `docker run vllm` / `chandra_vllm`. RunPod pods cannot do
docker-in-docker, so we run `vllm serve` directly from chandra's own venv and point the CLI
at it with `VLLM_API_BASE`.)

## Cold start on a fresh pod

```bash
cd /workspace/ocr_experiments

# 1. Build chandra's venv (~16 GB, ~10 min). Needed once per pod.
UV_HTTP_TIMEOUT=180 uv sync --project models/chandra

# 2. Start the vLLM server. ~7 min the first time (it also downloads ~10 GB of weights
#    to $HF_HOME); ~2-3 min afterwards when the weights are cached on the volume.
scripts/serve_chandra.sh

# 3. Run. Attaches to the server above and starts decoding immediately.
./run.sh chandra --out-dir my_run

# 4. When done, free the 39 GB of VRAM.
scripts/serve_chandra.sh --stop
```

That's it. Steps 1-2 are per-pod; step 3 is per-job and is the only one you repeat.

## Daily use

```bash
scripts/serve_chandra.sh                       # start once, leave it up all day
./run.sh chandra --out-dir batch_2026_07_14    # job 1  — instant start
./run.sh chandra --out-dir batch_2026_07_15    # job 2  — instant start
./run.sh chandra --pdfs Handwritten --out-dir spot_check   # subset of the sample_set
scripts/serve_chandra.sh --stop                # release VRAM
```

- `--out-dir <name>` writes to a **top-level** `outputs/<name>/`, so runs never overwrite
  each other and `scripts/compare.py` picks each up as its own row.
- `--smoke` redirects to `outputs/_smoke/chandra/` — use it for any config you have not
  validated, so its (possibly garbage) pages can never satisfy a real run's checkpoint.
- **Re-running the same command resumes.** Checkpoint is per-pdf: a finished `<stem>.md`
  is not redone. Killing a run mid-pdf costs you that pdf and nothing else.
- `--batch-size N` = how many pages the client has in flight at once. See tuning below.

Outputs per pdf: `<stem>.md`, `<stem>.html`, `<stem>.metadata.json` (per-page token counts
and page boxes — genuinely useful, this is how the rotation gain was measured),
`<stem>.metrics.json`, plus a `summary.json` for the run.

## Running it on your own documents (not the sample_set)

`list_pdfs()` globs `data/Evaluation set/sample_set/*.pdf`, so today the adapter only sees
that directory. To point it at a dossier, either drop the pdfs in there, or change
`SAMPLE_SET` in `harness/ocr_harness/__init__.py`. There is no `--input-dir` flag yet; it is
a ~5-line change if you want one.

## Freeing things up

| what | how | frees |
|---|---|---|
| **VRAM** (39.5 GB) | `scripts/serve_chandra.sh --stop` | the whole card |
| venv + uv cache + weights | `./scripts/reclaim.sh chandra datalab-to/chandra-ocr-2` | ~26 GB of disk |
| just the weights | `rm -rf "$HF_HOME/hub/models--datalab-to--chandra-ocr-2"` | ~10 GB |

`reclaim.sh` never touches `outputs/`. The `uv.lock` makes the venv rebuild deterministic,
so reclaiming only costs a re-download.

**Never run `uv cache clean` (or `reclaim.sh`) while any `uv` process is live.** `uv run`
probes the interpreter through a temp file *inside* that cache; delete it mid-flight and uv
dies with `Failed to query Python interpreter`, the adapter never starts, and the run looks
exactly like a hang. This cost us ~20 minutes. `run.sh` now calls `.venv/bin/python` and
`.venv/bin/vllm` directly and is immune, but the rule still applies to anything else.

## Tuning knobs that matter

- **`--max-model-len 18000` is mandatory on a 46 GB A40, not an optimization.** vLLM sizes
  the KV cache to 229,152 tokens; the model's config declares a 262,144-token context.
  262,144 > 229,152, so **without this flag vLLM refuses to start.** It is set for you in
  `run.sh` and `serve_chandra.sh` (`CHANDRA_TUNED=1`, the default).
- **`--batch-size` — leave it at 16. Raising it makes things WORSE, measured.** vLLM
  reports headroom for ~45 concurrent requests, which *looks* like the default 16 is
  under-driving the card. It is not. Same pdf, same resident server, only the flag changed:

  | `--batch-size` | s/page | chars |
  |---|---|---|
  | **16 (default)** | **12.25** | 110,013 |
  | 48 | 17.04 (**+39% slower**) | 110,533 |

  A page costs ~6,100 vision tokens of prefill, so 32 concurrent pages is ~196k prefill
  tokens against a 229k-token KV cache: the scheduler starts preempting sequences and
  thrashes. "Maximum concurrency 45.74x" is computed against `--max-model-len` (18,000), not
  against what a *page* actually costs, so it badly overstates the usable batch. **Corollary
  worth knowing: the A40 is already saturated at 16**, so a faster card would translate to
  real throughput rather than being wasted on an under-fed GPU.
- **`--gpu-memory-utilization`** (`GPU_MEM_UTIL`, default 0.85). Only KV cache scales with
  this; the weights are a fixed 8.61 GiB.
- **`IMAGE_DPI` — leave it at chandra's default of 192. Tested; raising it is worse.**
  It is env-overridable (`IMAGE_DPI=256 ./run.sh chandra ...`), and 256 saturates the
  model's pixel cap (74% more vision tokens). Over the full 68-page set that is **−1.2% on
  visible text and 35% slower**, and `Flowchart` — chandra's best document class — loses
  2,048 chars and 6 mermaid edges. Dense tables gain a little; everything else loses. See
  CLAUDE.md for the table. Anything above DPI 256 is doubly pointless: `scale_to_fit`
  clamps it to the identical image, so it costs rasterization CPU for byte-identical input.

Flags deliberately NOT set, with reasons, are documented in the `chandra)` case of `run.sh`.
Short version: `--enable-prefix-caching` can never hit (chandra puts the per-page-unique
image *before* the constant prompt), and the vendor's `--max-num-batched-tokens 4096` /
`--max-num-seqs 32` are H100-scaled values *lower* than vLLM's own defaults and would chunk
a single page's ~6,300-token prefill across two scheduler steps.

## Page-rotation correction — on by default, always

Every run first passes each pdf through `harness.orient_pdf` (PaddleOCR's PP-LCNet 4-class
doc-orientation classifier). It writes a `/Rotate`-corrected copy to
`work/oriented/<stem>.pdf` and chandra reads that. Reports land in `work/oriented/<stem>.json`,
and before/after previews in `work/oriented_preview/<stem>/`.

**The classifier is pinned to the CPU** and `run.sh` runs it as a pre-pass with
`CUDA_VISIBLE_DEVICES=""` *before* vLLM takes the card (verified: 0 MiB of GPU across a full
68-page pass). Never let it onto the GPU — vLLM already holds 85% of it.

Opt out with `--no-orient`, but don't: on the 7 rotated pages of `Complex_table_layouts` it
buys **+4.8%** more text against a **−0.7%** control on the same document's unrotated pages.

## Known noise you should ignore

**vLLM logs a big `Failed to load plugin register_paddlex_genai_models` traceback at every
startup.** `paddlex` (pulled in by the orientation classifier) registers a vLLM entry-point
plugin that fails to import. vLLM logs it at ERROR, then **continues and serves normally.**
It is cosmetic. Do not chase it.

## Sanity checks after a run

```bash
cat outputs/<name>/summary.json     # must list 5 pdfs, pages 32/3/12/14/7 = 68
python scripts/compare.py           # folds the run into outputs/comparison.json
```

Verify by page count *and* by eyeballing the markdown — never by exit code alone.
