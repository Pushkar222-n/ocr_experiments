# Context for Claude Code

Read this first. It is the handoff doc for this repo: what we are doing, what the pod
can and cannot run, what is already done, and what is left. `README.md` has the
user-facing detail; this file has the state and the hard-won gotchas.

## Goal

Benchmark ~9 open OCR models on `data/Evaluation set/sample_set/*.pdf` — five merged
per-category PDFs (`Complex_table_layouts` 32p, `Flowchart` 3p, `Formulas_with_tables`
12p, `Handwritten` 14p, `printouts` 7p; **68 pages total**). One model at a time: set it
up, smoke-test it, run the full set, verify, reclaim disk, move to the next.
`scripts/compare.py` merges the per-model `summary.json` files at the end.

## Pod constraints (these drive almost every decision)

- **GPU**: one A40, 46 GB VRAM. **Driver 570.195.03 / CUDA 12.8.**
- **`torch` must stay `<2.11`.** torch 2.11+ ships `+cu130` wheels that need driver
  >= 580. Every `models/*/pyproject.toml` therefore pins `torch>=2.10,<2.11`.
  (Earlier comments in this repo claimed the pod was driver 550.x / CUDA 12.4 and
  24 GB — all three were wrong and have been corrected.)
- **This is why in-process vLLM was abandoned** for `dots_ocr`, `lightonocr` and
  `mineru`: vLLM 0.20+ pins torch 2.11. They were rewritten onto plain `transformers`.
  **vLLM 0.19.1** (torch 2.10 / cu128) is the newest that runs here, and it is used
  only by `surya` and `chandra`, via a served HTTP endpoint.
- **No docker-in-docker.** Surya's and Chandra's stock launchers (`docker run vllm`,
  `chandra_vllm`) are replaced in `run.sh` by a plain `vllm serve`, addressed over
  HTTP via `SURYA_INFERENCE_URL` / `VLLM_API_BASE`.
- **Disk is the binding constraint, not VRAM.** `/workspace` is a MooseFS mount with
  **no hardlink support**, so `uv` *copies* out of its cache instead of linking: every
  `uv sync` writes the venv's bytes **twice** (once in `/workspace/.cache/uv/archive-v0`,
  once in `models/<x>/.venv`). Verified — `libtorch_cuda.so` has `links=1` in both.
  A vLLM venv alone is ~14 GB. The cache had silently grown to 20 GB.
  - `uv cache prune` reclaims almost nothing here (36 MB). Use **`uv cache clean`** —
    safe *because* the venvs are independent copies; clearing the cache cannot break
    an env that already exists.
  - `HF_HOME=/workspace/.cache/huggingface`, `UV_CACHE_DIR=/workspace/.cache/uv`, both
    set in the image env. That is why a `.cache` dir sits outside the repo.
  - **Run one model, then `./scripts/reclaim.sh <model> <hf-repo-id>` before the next.**

## Harness gotchas

- **A foreground Bash tool call is killed at 10 minutes.** A full 68-page run takes
  20-40 min, so it *must* be launched detached:
  `nohup ./run.sh <model> > /dev/null 2>&1 &`. Nothing is lost when a run is killed —
  see checkpointing below.
- To get the worker PID, match the **venv python binary**, not `run.py`:
  `pgrep -f 'models/<model>/.venv/bin/python'`. Matching `run.py` also matches the
  `uv run` wrapper and short-lived children, and you will end up watching a dead PID.
- **Checkpointing**: per-page models write `outputs/<m>/<stem>/pages/page_NNNN.meta.json`
  *last*, as the done-marker (it is `.meta.json`, deliberately distinct from a
  `native_ext="json"` adapter's `page_NNNN.json`, so they cannot clobber each other).
  Per-pdf models (`chandra`, `mineru`, `unlimited_ocr` multi-mode) checkpoint on the
  final `<stem>.md` existing.
- **Smoke tests must pass `--smoke`** (routes to `outputs/_smoke/<model>/`). A smoke run
  exercises an unvalidated config, so its pages are the *last* thing a real run should
  resume from — yet without the flag they land in `outputs/<model>/` as done-markers and
  the full run skips them. This actually bit `got_ocr`: a 7-page smoke test on a broken
  generation config left 162-char pages that a later full run would have kept.
  `compare.py` ignores `_smoke/` (it globs `outputs/*/summary.json`, one level up).
- `run.sh` tees all output to `work/<model>_run.log`, appended across resumes.
- Neither `outputs/` nor `data/` is committed (public repo, customer PDFs). Carry state
  between pods with a resume bundle: `zip -rq ocr_resume_bundle.zip outputs data`,
  then `unzip -q` it at the repo root. Restores every checkpoint; runs resume in place.

## Status

Done, full 68-page set, verified:

| model | s/page (weighted) | peak VRAM | notes |
|---|---|---|---|
| `lightonocr` | ~15.2 | 15.5 GB | emits **HTML** tables, not markdown pipes |
| `dots_ocr` | ~20.4 | 19.0 GB | layout JSON w/ bbox+category; batch 8 ≈ batch 2 (decode-bound, padding eats the gain) |
| `got_ocr` | ~8.2 | 13.4 GB | native format is **mathpix LaTeX**, not markdown (`\title{}`, escaped `\&`) despite the `.md` extension. **Treat its char counts as inflated** — see below |

**Finding worth keeping**: on `Flowchart`, `dots_ocr` produced **626 chars** vs
`lightonocr`'s **4846** (7.7x). dots.mocr's `prompt_layout_all_en` appears to drop text
*inside* flowchart shapes. Check this before trusting dots_ocr on diagram-heavy docs.

**Finding worth keeping**: `got_ocr` degenerates into token loops on 5 of 68 pages —
two repeat a SMILES fragment (`[C@@H]1`) into a table, one loops a LaTeX column spec
(`|c`), one repeats ` 50000000`. Each runs to the 4096-token cap, so its `total_chars`
*overstates* real extraction. It also collapses on `printouts` (433 chars vs
lightonocr's 16082) and `Flowchart` (238 vs 4850), while staying competitive on
`Formulas_with_tables` (21.8k vs 25.0k). Not tunable: `crop_to_patches` fixes the input
side but the model degenerates past ~3 patches per forward (4/7 pages never emit EOS);
`format=False` degenerates on 2/7; fp32 rules out numerics; `sdpa` is unsupported for
this arch. Run it on the stock recipe and read its numbers with the loops in mind.
A quick scan for this failure mode across any model's pages:
`re.search(r'(.{1,12}?)\1{15,}', page_text)` — it also flags one `lightonocr` page.

**GOT-OCR needs an explicit `eos_token_id`.** Its `generation_config.eos_token_id` is
`None` and the tokenizer's eos is `<|endoftext|>` (151643), *not* the `<|im_end|>`
(151645) the model emits. `stop_strings` alone exposes no `eos_token_id`, so HF never
pads finished rows: a sequence that ends early keeps decoding garbage until the slowest
row in the batch finishes. Silent at batch=1, corrupts every page at the adapter's
`default_batch=4`. Fixed in `models/got_ocr/run.py`; watch for the same trap in any
other adapter that stops on a string rather than a token id.

Remaining, in this order (surya + chandra last, they need the vLLM server):

1. `unlimited_ocr` — transformers + `trust_remote_code`, batch 1. `baidu/Unlimited-OCR`
2. `paddleocr_vl` — **the only model with no `uv.lock`; never resolved.** Pins the
   `cu126` paddle index, which is fine under a 12.8 driver (CUDA 12.x minor compat).
3. `glm_ocr` — **suspect**: `pyproject.toml` claims `glmocr[selfhosted]` never depends
   on vLLM, but `run.py`'s docstring says "vLLM decoder". If `GlmOcr()` tries to start
   vLLM at runtime it will fail, since vLLM is not in that venv. Verify before running.
4. `mineru` — `vlm-transformers` backend.
5. `surya` — rebuild `models/vllm_server/.venv` (~14 GB, deleted to free space; the
   `uv.lock` is committed). `run.sh` serves on :8100.
6. `chandra` — carries its **own** vLLM (~14 GB) in its venv; `run.sh` serves on :8200.
   Run `surya` first, reclaim fully, then `chandra` — never both resident.
7. `python scripts/compare.py`

Also unverified: `mineru` and `chandra` both `shutil.copy(mds[0], ...)` where
`mds = sorted(work_out.rglob("*.md"))` — i.e. whichever `.md` sorts first. Confirm that
is the right file. Both also leave their intermediate `outputs/<model>/<stem>/` work
dirs behind. (Their resume-drops-skipped-pdfs-from-`summary.json` bug is fixed.)

## Working agreements

- Run one model end-to-end before starting the next; never leave two venvs resident.
- Smoke-test on `--smoke --pdfs printouts` (7 pages, ~2 min) before committing to a full run.
- Verify a run by page count (`32/3/12/14/7`) *and* by eyeballing the markdown, not by
  exit code alone.
- Never delete `outputs/`. Reclaim only venvs, the uv cache, and HF weights.
