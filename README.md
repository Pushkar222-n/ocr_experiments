# OCR experiments

Compare OCR models on `data/Evaluation set/sample_set/*.pdf` (merged per-category
PDFs: Complex_table_layouts, Flowchart, Formulas_with_tables, Handwritten, printouts).

## Layout

```
harness/            shared lib: pdf->page-png rendering, per-page checkpointing,
                     timing/GPU-mem metrics, output assembly (ocr_harness package)
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

## Running a model

`./run.sh <model>` syncs that model's venv, starts a vLLM server if the model needs
one, runs it over all of `sample_set`, and tees everything to `work/<model>_run.log`
(appended, so a resumed run keeps the history of the attempts that got it there).

```bash
./run.sh got_ocr
./run.sh lightonocr
./run.sh dots_ocr
./run.sh unlimited_ocr
UNLIMITED_MULTI=1 uv run --project models/unlimited_ocr python models/unlimited_ocr/run.py  # one-shot whole-pdf mode
./run.sh paddleocr_vl
./run.sh glm_ocr
./run.sh mineru
./run.sh surya      # starts `vllm serve datalab-to/surya-ocr-2` (pip vllm, no docker)
./run.sh chandra    # starts `vllm serve datalab-to/chandra-ocr-2` (pip vllm, no docker)

python scripts/compare.py
```

Restrict to specific PDFs: `./run.sh got_ocr --pdfs Handwritten Flowchart`.
Override batching: `./run.sh dots_ocr --batch-size 8`.

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
| mineru | `opendatalab/MinerU2.5-2509-1.2B` |
| surya | `datalab-to/surya-ocr-2` |
| chandra | `datalab-to/chandra-ocr-2` |

## Per-model notes

| model | backend | native format | batches on GPU | notes |
|---|---|---|---|---|
| GOT-OCR-2.0 | transformers | markdown (mathpix `format=True`) | yes (batch=4) | generic; `format=True` prompt |
| LightOnOCR-2-1B | transformers (`LightOnOcr*` classes) | markdown | yes (batch=4) | temp=0.2, top_p=0.9, images resized to 1540px longest side |
| dots.mocr (rednote-hilab, formerly dots.ocr) | transformers, `trust_remote_code` | layout json (bbox+category+text; tables as HTML, formulas as LaTeX) | yes (batch=2) | uses card's `prompt_layout_all_en`; markdown built by joining element text in reading order; ships an SDPA shim for the vision tower's flash-attn import |
| Unlimited-OCR (Baidu) | transformers, `trust_remote_code` | markdown/text | no (1 img/call); `infer_multi` batches whole pdf in one forward pass | uses card's exact recipe (`base_size=1024, image_size=640, crop_mode=True`) — your earlier bad pages were likely a size/prompt mismatch |
| PaddleOCR-VL-1.6 | paddlepaddle-gpu 3.2.1 | markdown + layout json | pipeline-internal | best-in-class for tables/formulas per OmniDocBench |
| GLM-OCR | glmocr[selfhosted] (+ PP-DocLayout-V3) | markdown + json (bboxes) | pipeline-internal | layout model can run on CPU (`LAYOUT_DEVICE=cpu`) to save VRAM |
| MinerU 2.5 | `mineru[core]`, vlm-transformers backend | markdown + content_list.json | pipeline-internal | checkpoints per-pdf |
| Surya 2 | external pip-vllm server (`SURYA_INFERENCE_URL`) | block json incl. **per-block confidence**, markdown via markdownify | vLLM continuous batching | only model here with a native quality score |
| Chandra OCR 2 | external pip-vllm server (its own docker launcher swapped for a plain `vllm serve`) | markdown + html + metadata json (token counts) | vLLM continuous batching (`--batch-size`) | checkpoints per-pdf |

Surya's and Chandra's docs default to a `docker run vllm`/`chandra_vllm` launcher —
since RunPod pods can't do docker-in-docker, `run.sh` instead does
`vllm serve <model>` directly (pip-installed vllm) and points each tool at it via
env var (`SURYA_INFERENCE_URL`, `VLLM_API_BASE`).

## Pod constraints

Driver 570.x / CUDA 12.8, one A40 (46 GB VRAM). **torch must stay `<2.11`**: 2.11+
ships `+cu130` wheels that need driver >= 580. Every model pins `torch>=2.10,<2.11`
for this reason, and it is also why the in-process-vLLM adapters were rewritten onto
plain transformers — vLLM 0.20+ pins torch 2.11. vLLM 0.19.1 (torch 2.10 / cu128)
is the newest that runs here, and it is used only by surya/chandra via a served endpoint.

Disk is the tighter constraint. `/workspace` is a MooseFS mount with **no hardlink
support**, so `uv` *copies* rather than links out of its cache — every `uv sync`
writes the venv's bytes twice (once in `/workspace/.cache/uv/archive-v0`, once in
`models/<x>/.venv`). A vLLM venv alone is ~14 GB. Run one model at a time and reclaim
after each:

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


