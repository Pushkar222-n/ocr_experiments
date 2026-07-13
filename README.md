# OCR experiments

Compare OCR models on `data/Evaluation set/sample_set/*.pdf` (merged per-category
PDFs: Complex_table_layouts 32p, Flowchart 3p, Formulas_with_tables 12p, Handwritten 14p,
printouts 7p — **68 pages**).

## Results — all 9 models, 68 pages each

| model | engine | s/page | raw chars | **visible chars** | %text |
|---|---|---|---|---|---|
| `lightonocr` | **vllm** | **2.8** | 221,578 | 115,566 | 52% |
| `surya` | vllm | 4.1 | 112,004 | 99,622 | 89% |
| `got_ocr` | transformers | 8.2 | 114,021 | 109,394 | 96%¹ |
| `glm_ocr` | vllm | 9.6 | 147,321 | 83,292 | 57% |
| `chandra` | vllm | 9.8 | 180,288 | 109,101 | 61% |
| `mineru` | **vllm** | 11.4 | 201,262 | **128,299** | 64% |
| `dots_ocr` | transformers | 21.2 | 154,836 | 91,283 | 59% |
| `unlimited_ocr` | transformers | 25.0 | 184,140 | 97,520 | 53% |
| `paddleocr_vl` | paddle | 25.5 | **422,808** | 88,841 | **21%** |

> ⚠️ **The s/page column compares engines as much as models.** Five models run under vLLM
> (continuous batching); four still run in-process, where a padded static batch costs as
> much as its longest page. Both models we A/B'd got **5-6x faster** just by moving to
> vLLM with *identical* weights and sampling — `lightonocr` 15.3 → **2.8**, `mineru`
> 72.4 → **11.4**. So `got_ocr`, `dots_ocr`, `unlimited_ocr` and `paddleocr_vl` are very
> likely understated by a similar factor, and their positions at the bottom of this table
> should not be read as "slow models". They are un-migrated ones. `compare.py` prints which
> engine's row it used for each model.

**Rank on `visible chars`, never on `total_chars`.** Each model emits a different native
format and `total_chars` counts every byte of markup. `paddleocr_vl` tops the raw column
with 422,808 chars and lands **8th of 9** on real text, because it puts inline CSS on
*every* `<td>`. ¹ `got_ocr`'s 96% is the mirror artifact — the stripper is a regex
`<[^>]+>` and does not remove its mathpix LaTeX, and it degenerates into token loops on
5 of 68 pages, inflating its count. ² VRAM for the vLLM-served models is a *reservation*
(`--gpu-memory-utilization`), not demand — do not compare it against the in-process models.

**Three findings that outrank the table:**

1. **Serve the model with vLLM. It is free speed, and sometimes free quality.** Two models
   were A/B'd with identical weights and identical sampling, engine the only variable:
   - **`lightonocr`: 15.3 → 2.8 s/page (5.4x), output unchanged** (218,449 → 221,578 chars,
     +1.4%; degeneration 3/68 → 4/68 pages, and most of those flags are false positives —
     repeated "Not Detected" cells in a real table). A pure, free 5.4x. **It becomes the
     fastest model in the benchmark.**
   - **`mineru`: 72.4 → 11.4 s/page (6.4x), and the output got *better*** — because its
     transformers client silently drops the `presence_penalty`/`frequency_penalty` that
     `mineru_vl_utils` sets for every content type (the library marks them
     `# not supported by hf`). 190,218 → 201,262 chars, and 2-of-3 → 3-of-3 flowchart
     graphs recovered. **On mineru the engine is not a performance choice, it is a
     correctness one.** See CLAUDE.md.

   The mechanism is the same in both cases: the transformers path static-batches and pads,
   so every batch costs as much as its longest page, and the GPU idles. vLLM continuously
   batches. **`mineru` extracts the most real text of any model here (128,299 visible) —
   but only when served.**
2. **`surya` is 2.4x faster than anything else**, and the only model reporting a confidence
   score — **which is not a coverage metric.** It is page-level (every block on a page
   carries an identical value), and it grades only what surya *chose* to read. On
   `Flowchart` it dropped the entire diagram and still reported **0.947**. Do not use it as
   a quality gate.
3. **Only `chandra` and `mineru` read diagrams at all** — both emit the flowchart's topology
   as mermaid. `chandra` is better at it (24 nodes / 55 edges / **10 feedback loops** vs
   21 / 40 / **0**): it captures the QC sampling steps that cycle *back* into the process
   line, which `mineru` flattens away. The other 6 layout-stage models crop the diagram to
   an image and never read inside it. That is a *choice* they make, not a limit they hit.

## Layout

```
harness/            shared lib: pdf->page-png rendering, per-page checkpointing,
                     timing/GPU-mem metrics, output assembly, page-rotation
                     correction (ocr_harness package)
models/<name>/       one uv project per model — isolated venv, own torch/
                     transformers/vllm/paddle pins, no cross-model version conflicts
outputs/<name>/      <pdf_stem>.md (+ native .json/.html), <pdf_stem>.metrics.json,
                     summary.json, and <stem>/pages/ per-page checkpoints.
                     Fully gitignored — derived from customer PDFs, repo is public
work/                rasterized page pngs, per-model run logs (<model>_run.log, appended
                     across resumed runs) + vllm server logs (regenerable, gitignored)
run.sh               entrypoint: uv sync + (if needed) start a pip-vllm server + run
scripts/reclaim.sh   free disk after a model finishes (uv cache + venv + hf weights)
scripts/compare.py   aggregate every outputs/*/summary.json into one table
```

Neither `outputs/` nor `data/` is in git. To move state between pods, carry the
resume bundle (~27 MB) and unzip it at the repo root — every checkpoint is restored
and the next `./run.sh <model>` skips what's already done:

```bash
zip -rq ocr_resume_bundle.zip outputs data   # before tearing a pod down
unzip -q ocr_resume_bundle.zip               # on the new pod, at repo root
```

Each model is its own `uv` project so torch/transformers/vllm/paddle versions never
collide — `uv sync --project models/<x>` only ever touches `models/<x>/.venv`.

## Page-rotation correction (`ocr_harness.orient_pdf`)

Any model adapter can call `harness.orient_pdf(pdf_path)` before handing a pdf to its
model (or before rendering pages) to detect and fix pages that were scanned sideways.
It uses PaddleOCR's `PP-LCNet_x1_0_doc_ori` — a tiny 4-class (0/90/180/270)
doc-orientation classifier — on the **onnxruntime** backend, so it needs neither
paddlepaddle nor a GPU. It's declared in `harness/pyproject.toml` (not per-model), so
every model's venv gets it transitively through the `ocr-harness` editable dependency.

`orient_pdf` renders each page, classifies it, and — only for pages that need it —
writes a corrected copy of the pdf under `work/oriented/<stem>.pdf` with that page's
`/Rotate` attribute fixed. Any renderer that honors `/Rotate` (pypdfium2, and chandra's
own pdf loader) then renders it upright automatically; nothing else about the pdf
changes. Cached on (size, mtime), and a per-pdf report lands at
`work/oriented/<stem>.json` (`flagged_pages`, per-page angle + confidence). If any page
was flagged, before/after preview pngs are saved to
`work/oriented_preview/<stem>/page_NNNN_{before,after}.png` so you can eyeball the fix
before trusting it.

Checked against the real `sample_set`: only `Complex_table_layouts.pdf` has rotated
pages — **7 of 32** (pages 3, 4, 25, 26 at 270°; pages 8, 13, 14 at 90°; confidence
0.83–0.93). Visually confirmed correct in both rotation directions.

`models/chandra/run.py` is the first (and so far only) caller — see CLAUDE.md for how
it's wired in and what's left to verify against real OCR output.

## Running a model

`./run.sh <model>` syncs that model's venv, starts a vLLM server if the model needs
one, runs it over all of `sample_set`, and tees everything to `work/<model>_run.log`
(appended, so a resumed run keeps the history of the attempts that got it there).

```bash
./run.sh got_ocr
./run.sh lightonocr                                          # in-process transformers
LIGHTON_VLLM=1 ./run.sh lightonocr --out-tag output_vllm     # same weights, served by vllm
./run.sh dots_ocr
./run.sh unlimited_ocr
UNLIMITED_MULTI=1 uv run --project models/unlimited_ocr python models/unlimited_ocr/run.py  # one-shot whole-pdf mode
./run.sh paddleocr_vl
./run.sh glm_ocr    # starts `vllm serve zai-org/GLM-OCR`; glmocr is only an http client
MINERU_VLLM=1 ./run.sh mineru --backend vlm-http-client --out-tag output_vllm  # see below
./run.sh surya      # starts `vllm serve datalab-to/surya-ocr-2` (pip vllm, no docker)
./run.sh chandra    # starts `vllm serve datalab-to/chandra-ocr-2` (pip vllm, no docker)
./run.sh chandra --out-tag oriented   # same, + rotation-correction (see above), --include-headers-footers,
                                      # and tuned vLLM flags; writes outputs/chandra/oriented/, baseline untouched

python scripts/compare.py
```

## Closed / paid API models

Four hosted document-parse APIs (Mistral OCR, Datalab Marker, LlamaParse, Landing AI ADE)
on their **balanced** mid-tier, over the same `sample_set`. Separate project, separate
storage (`outputs/closed/<provider>/`), plain `requests` (no vendor SDKs). Keys go in
`.env` (gitignored). See the "Closed / paid API models" section of `CLAUDE.md` for the
full results — headline: **Datalab Marker wins outright**: more visible text than any open
model here (149,941 vs MinerU's 128,299), the cheapest paid API ($3/1k pages, $0.20 for all
68), and near-fastest. `landing_ai` is the trap — $30/1k for the *least* text of the four.

```bash
# keys in .env: DATALAB_API_KEY, LANDING_API_KEY, MISTRAL_API_KEY, LLAMAPARSE_API_KEY
cd models/closed_apis
uv run python run.py all --concurrency 3          # all four providers, 5 pdfs each
uv run python run.py datalab --smoke --pdfs Flowchart   # cheap single-provider smoke
uv run python run.py prices                       # every tier's rate + cost for the 68p set
uv run python run.py reprice                      # recompute stored costs offline (no API calls)
cd ../.. && python scripts/compare.py             # folds closed rows into comparison.json
```

Restrict to specific PDFs: `./run.sh got_ocr --pdfs Handwritten Flowchart`.
Override batching: `./run.sh dots_ocr --batch-size 8`.

**Run mineru on vLLM, not on its default in-process engine.** Plain `./run.sh mineru`
resolves to the transformers engine, which does not just run slow — it runs a *different
decoding recipe* than mineru intends, because the transformers client silently drops the
`presence_penalty`/`frequency_penalty` that `mineru_vl_utils` sets for every content type
(the library marks them `# not supported by hf`). Over the 68-page set the served path is
**6.4x faster (72.4 → 11.4 s/page) and extracts *more* text** (190,218 → 201,262 chars);
on `Flowchart` the in-process path recovers 2 of 3 mermaid graphs where vLLM recovers 3.
`MINERU_VLLM=1` serves the model from `models/vllm_server` on :8400 and the `--out-tag`
keeps the two runs from overwriting each other. CLAUDE.md has the full analysis, plus the
`config.json` patch (`scripts/mineru_vllm_model.py`) vLLM needs to load these weights.

**Re-running the same command resumes.** Per-page models write
`outputs/<model>/<stem>/pages/page_NNNN.meta.json` *last*, as the page's done-marker;
per-pdf models (chandra, mineru, unlimited_ocr multi-mode) checkpoint on the final
`.md` existing. Killing a run mid-page costs you that page and nothing else.

**Always smoke-test with `--smoke`.** It redirects the run to `outputs/_smoke/<model>/`,
so those pages can never satisfy a real run's checkpoint. A smoke test exists to exercise
a config you have *not* validated yet — without the flag its (possibly garbage) pages
land in `outputs/<model>/` as done-markers, and the full run silently skips them.
`outputs/_smoke/` is throwaway; `scripts/compare.py` ignores it.

### The loop, one model at a time

A full `sample_set` pass is 68 pages and takes 20-40 min per model, and only one
model's venv fits on disk comfortably (see Pod constraints). So: run one, verify it,
reclaim, move on. Worked example, exactly how `lightonocr` and `dots_ocr` were done:

```bash
# 1. smoke-test on the smallest pdf first — catches a broken env in ~2 min
./run.sh lightonocr --smoke --pdfs printouts         # 7 pages
head -c 500 outputs/_smoke/lightonocr/printouts.md  # eyeball it before spending 30 min

# 2. full run, detached (it outlives any 10-min foreground timeout)
nohup ./run.sh lightonocr > /dev/null 2>&1 &
tail -f work/lightonocr_run.log               # follow along; ^C only stops the tail
pgrep -f 'models/lightonocr/.venv/bin/python' # -> pid, if you need to `kill` it

# 3. verify: page counts must be 32/3/12/14/7 (=68) and summary.json must list 5 pdfs
cat outputs/lightonocr/summary.json
for d in outputs/lightonocr/*/; do echo "$(basename $d) $(ls $d/pages/*.meta.json | wc -l)"; done

# 4. reclaim before the next model, or the pod fills up
./scripts/reclaim.sh lightonocr lightonai/LightOnOCR-2-1B
```

`scripts/reclaim.sh <model> [hf-repo-id ...]` runs `uv cache clean`, deletes
`models/<model>/.venv`, and deletes the named HF weight dirs. It never touches
`outputs/`. The `uv.lock` in each model dir makes the venv rebuild deterministic, so
reclaiming is cheap to undo — it only costs a re-download.

HF repo ids per model, for the reclaim call:

| model | hf repo id(s) |
|---|---|
| got_ocr | `stepfun-ai/GOT-OCR-2.0-hf` |
| lightonocr | `lightonai/LightOnOCR-2-1B` |
| dots_ocr | `rednote-hilab/dots.mocr` |
| unlimited_ocr | `baidu/Unlimited-OCR` |
| paddleocr_vl | (paddle caches under `~/.paddlex`, not HF) |
| glm_ocr | `zai-org/GLM-OCR` + PP-DocLayout-V3 |
| mineru | `opendatalab/MinerU2.5-Pro-2605-1.2B` (what mineru 3.x actually pulls; served from a config-patched copy in `work/mineru2.5_vllm/`) |
| surya | `datalab-to/surya-ocr-2` |
| chandra | `datalab-to/chandra-ocr-2` |

## Per-model notes

| model | backend | native format | batches on GPU | notes |
|---|---|---|---|---|
| GOT-OCR-2.0 | transformers | markdown (mathpix `format=True`) | yes (batch=4) | generic; `format=True` prompt |
| LightOnOCR-2-1B | transformers (`LightOnOcr*` classes), **or a pip-vllm server** (`LIGHTON_VLLM=1` -> `LIGHTON_URL`) | markdown | transformers: static batch=4 (pads); vllm: 8 concurrent requests | temp=0.2, top_p=0.9, images resized to 1540px longest side — **identical on both engines**, so the engine is the only variable. vLLM is ~3x faster at equivalent quality; see the A/B below |
| dots.mocr (rednote-hilab, formerly dots.ocr) | transformers, `trust_remote_code` | layout json (bbox+category+text; tables as HTML, formulas as LaTeX) | yes (batch=2) | uses card's `prompt_layout_all_en`; markdown built by joining element text in reading order; ships an SDPA shim for the vision tower's flash-attn import |
| Unlimited-OCR (Baidu) | transformers, `trust_remote_code` | markdown/text | no (1 img/call); `infer_multi` batches whole pdf in one forward pass | uses card's exact recipe (`base_size=1024, image_size=640, crop_mode=True`) — your earlier bad pages were likely a size/prompt mismatch |
| PaddleOCR-VL-1.6 | paddlepaddle-gpu 3.2.1 | markdown + layout json | pipeline-internal | best-in-class for tables/formulas per OmniDocBench |
| GLM-OCR | glmocr[selfhosted] (+ PP-DocLayout-V3) | markdown + json (bboxes) | pipeline-internal | layout model can run on CPU (`LAYOUT_DEVICE=cpu`) to save VRAM |
| MinerU 2.5 | `mineru[core]`, **`vlm-http-client` against a pip-vllm server** (`MINERU_URL`); the default in-process engine changes the output, see above | markdown + content_list.json | vLLM continuous batching | checkpoints per-pdf. **Extracts the most real text of any model here** (128,299 visible). Reads a flowchart's topology as mermaid — one of only two that do, and the weaker of the two (no feedback loops) |
| Surya 2 | external pip-vllm server (`SURYA_INFERENCE_URL`) | block json incl. **per-block confidence**, markdown via markdownify | vLLM continuous batching | fastest model here (4.1 s/page). Its confidence is **page-level and is not a coverage metric** — it scored 0.947 on a page where it dropped the whole diagram. Do not gate on it |
| Chandra OCR 2 | external pip-vllm server (its own docker launcher swapped for a plain `vllm serve`); carries its **own** vllm (~15 GB) — run it last | markdown + html + metadata json (token counts) | vLLM continuous batching (`--batch-size`) | checkpoints per-pdf. **Best diagram parser in the field** — 24 nodes / 55 edges / 10 feedback loops on `Flowchart` vs mineru's 21/40/0 — but fences the mermaid *bare*, so it will not auto-render. `run.py` now runs every pdf through `harness.orient_pdf` first (see above) and always passes `--include-headers-footers`; `run.sh` serves it with `--max-model-len 18000` + `--mm-processor-kwargs` (datalab's own recommended values — see CLAUDE.md for why the vendor's other two flags were deliberately left out). Re-run tagged `--out-tag oriented`, not yet verified end to end — see CLAUDE.md Status |

Surya's and Chandra's docs default to a `docker run vllm`/`chandra_vllm` launcher —
since RunPod pods can't do docker-in-docker, `run.sh` instead does
`vllm serve <model>` directly (pip-installed vllm) and points each tool at it via
env var (`SURYA_INFERENCE_URL`, `VLLM_API_BASE`).

## Pod constraints

Driver 570.x / CUDA 12.8, one A40 (46 GB VRAM). **torch must stay `<2.11`**: 2.11+
ships `+cu130` wheels that need driver >= 580. Every model pins `torch>=2.10,<2.11`
for this reason, and it is also why the in-process-vLLM adapters were rewritten onto
plain transformers — vLLM 0.20+ pins torch 2.11. vLLM 0.19.1 (torch 2.10 / cu128)
is the newest that runs here, and it is never imported in-process: `models/vllm_server`
serves it over HTTP to mineru, surya and glm_ocr (chandra carries its own). Note that
0.19.1 predates the transformers-v5 nested config layout, so a model whose `config.json`
nests fields under `text_config` can fail to load — see the `tie_word_embeddings` case in
CLAUDE.md, which is why mineru is served from a patched copy of its config.

Disk is the tighter constraint. `/workspace` is a MooseFS mount with **no hardlink
support**, so `uv` *copies* rather than links out of its cache — every `uv sync`
writes the venv's bytes twice (once in `/workspace/.cache/uv/archive-v0`, once in
`models/<x>/.venv`). A vLLM venv alone is ~14-15 GB. Run one model at a time and reclaim
after each:

> **The volume quota (~50 GB) is invisible to `df`.** `df` reports the whole MooseFS
> cluster and will cheerfully tell you there are hundreds of TB free while the volume is
> full. **Use `du -sh /workspace`.** When it fills, `uv sync` is **killed with no error**:
> the log simply stops mid-download and leaves a half-written venv. That happened once here
> and looked exactly like a hang. If a sync dies quietly, suspect the quota first, reclaim,
> and retry — it worked first time after. The uv cache regrows to ~10 GB with *every*
> model, so reclaiming between models is not housekeeping, it is what keeps you under the
> ceiling.

```bash
uv cache clean                  # safe: venvs are independent copies, not links
rm -rf models/<finished>/.venv  # uv.lock makes the rebuild deterministic
rm -rf "$HF_HOME/hub/models--<org>--<name>"
```

## Metrics captured

Per page: seconds, char count, GPU mem (nvidia-smi peak), any adapter-reported
extra (Surya: confidence; LightOnOCR: output token count). Per pdf
(`<stem>.metrics.json`): total/avg seconds per page, total chars, peak GPU mem,
and the mean of any numeric per-page extras. `scripts/compare.py` merges every
model's `summary.json` into one table + `outputs/comparison.json` for
downstream (human or LLM) review.


