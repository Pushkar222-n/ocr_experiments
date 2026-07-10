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
- **`pgrep -f` matches its own shell's command line.** A wait loop like
  `until ! pgrep -f 'run.sh mineru'; do sleep 10; done` never exits: the `bash -c`
  running it *contains* the string `run.sh mineru`, so pgrep finds itself and the loop
  spins forever while the real run finished long ago. Wait on the **PID** instead —
  `nohup ./run.sh <m> & PID=$!; while kill -0 $PID 2>/dev/null; do sleep 20; done` —
  or use the `[r]un.sh` bracket trick.
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
| `unlimited_ocr` | ~25.0 | 10.7 GB | 184,140 chars. Native format is grounding tags + HTML tables (`<\|det\|>text [x,y,x,y]<\|/det\|>`), not markdown. Needs `max_length=4096` or it hangs; 6/68 pages hit that cap, all in `Complex_table_layouts`. Capped pages are **under**-counted — see below |
| `paddleocr_vl` | ~25.5 | 17.6 GB | 422,808 raw chars but only ~88.8k visible — inline CSS on **every** `<td>`. Lowest real extraction on `Complex_table_layouts` (46,456) despite the highest raw count (266,794); highest on `printouts` (10,760). Has a layout stage → skips diagrams |

**Finding worth keeping**: on `Flowchart`, `dots_ocr` produced **626 chars** vs
`lightonocr`'s **4846** (7.7x). dots.mocr's `prompt_layout_all_en` appears to drop text
*inside* flowchart shapes. Check this before trusting dots_ocr on diagram-heavy docs.
`unlimited_ocr` collapses the same way (684 chars) but for a *different, verified*
reason: it classifies the diagram as a picture and emits `![](images/0.jpg)`, never
attempting the text inside. That is a layout call, not a decode failure.

**RESOLVED — the flowchart collapse is a layout decision, not an OCR failure.** Checked
`outputs/dots_ocr/Flowchart/pages/page_*.json` (no GPU needed). Across the 3 pages
dots.mocr emits **4 `Picture` elements covering 92.8% of all element area, every one with
no `text` key at all** (absent, not empty). Its whole 626 chars come from Page-header
(420) + Section-header (98) + Caption (56) + Page-footer (32). So dots_ocr *declines* to
OCR inside the diagram — the same behaviour as `unlimited_ocr`, which emits
`![](images/0.jpg)`. Neither model tried and failed.

`lightonocr` scores 4846 on the same pages because it has **no layout stage** and simply
reads the whole page. **This is a structural property of layout-then-OCR pipelines.**
`paddleocr_vl` was predicted to collapse the same way and did — it emits
`<img src="imgs/img_in_image_box_...">`, yields 161 visible chars (worst of all five),
and finishes `Flowchart` in 1.77 s/page *because* it skipped the work.

Scoreboard on the single most discriminating page in the set:

| model | raw | visible | layout stage? |
|---|---|---|---|
| `lightonocr` | 4846 | **4588** | no — reads everything |
| `unlimited_ocr` | 684 | 683 | yes — `![](images/0.jpg)` |
| `dots_ocr` | 626 | 618 | yes — `Picture`, no `text` key |
| `paddleocr_vl` | 609 | 161 | yes — `<img src=...>` |
| `got_ocr` | 234 | 236 | no, but degenerate |

**Still to check: `glm_ocr` and `mineru`** — both run a layout model first, so both are
predicted to skip diagrams. If either reads inside the shapes, that is the interesting
model for diagram-heavy documents and worth calling out. Costs nothing: just read its
`Flowchart` output. Corollary: on such documents `total_chars` and `visible_chars`
measure *whether the model has a layout stage*, not how well it reads — and no tuning
moves a model across that line.

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

Remaining, in this order (the vLLM-server models last — see the regrouping note):

1. `mineru` — `vlm-transformers` backend. **Never started.** Its venv was never built on
   the previous pod; nothing to resume, nothing to clean.
2. `surya` — rebuild `models/vllm_server/.venv` (~14 GB, deleted to free space; the
   `uv.lock` is committed). `run.sh` serves on :8100.
3. `glm_ocr` — **needs a served endpoint; see below.** Run it right after `surya` so the
   `models/vllm_server/.venv` gets built once, not twice.
4. `chandra` — carries its **own** vLLM (~14 GB) in its venv; `run.sh` serves on :8200.
   Run the others first, reclaim fully, then `chandra` — never two vLLMs resident.
5. `python scripts/compare.py`

## Resuming on a fresh pod (state as of 2026-07-10)

**5 of 9 models are done**: `lightonocr`, `dots_ocr`, `got_ocr`, `unlimited_ocr`,
`paddleocr_vl` — each a verified 68 pages (32/3/12/14/7).

The pod that produced them was torn down. Nothing is resident: no venvs, no `uv` cache,
no HF weights, no `work/`. Every remaining model therefore starts from a cold download —
budget ~4 min of `uv sync` plus a weight pull *before* `nvidia-smi` shows anything on the
GPU. **0 MiB during a new model's first ~10 minutes is normal, not a hang.** Confirm with
`ls /proc/<worker-pid>/fd | wc -l` and `ss -tnp | grep <pid>`: live HTTPS sockets and
growing CPU time mean it is downloading.

`outputs/` was downloaded off the pod, not committed (public repo, customer PDFs). To
resume, restore it at the repo root — `unzip -q ocr_resume_bundle.zip` or copy the
`outputs/` and `data/` trees back — and the finished models' `summary.json` files will be
picked up by `scripts/compare.py`. Without `outputs/`, `compare.py` prints nothing and the
five completed runs would have to be redone.

Nothing is uncommitted; `main` is pushed through the `paddleocr_vl` results.

`unlimited_ocr`'s `infer_multi` (whole-pdf one-shot, `UNLIMITED_MULTI=1`) writes
`outputs/unlimited_ocr_multi/` and is a throughput data point, not a benchmark row.
**Only `Flowchart` completed**: 3 pages, 23.7s (7.9 s/page), **1279 chars — vs 684 in
per-page mode.** The one-shot pass extracts ~2x more text from the same diagram, so the
flowchart collapse is partly an artifact of single-page context, not purely the
"it's a picture" layout call. Worth a look if diagram fidelity matters.

`printouts` (7 pages) was **killed after ~11 min** in one shot and its `summary.json` was
never written. `run_multi` passes `max_length=32768` — the same ceiling that let one page
decode for >8 min — and a one-shot has no per-page checkpoint to fall back on. Do not
point it at `Complex_table_layouts` (32 pages). If you want multi mode on real documents,
cap `max_length` there too.

**`glm_ocr` is a vLLM-server model, not a transformers model.** The old note here guessed
`GlmOcr()` might try to *start* vLLM in-process and fail on the missing import. Wrong on
both counts: `glmocr[selfhosted]` really has no vllm dep (checked on PyPI — only
torch/torchvision/transformers>=5.3/sentencepiece/accelerate/pypdfium2/opencv), and it
never loads the decoder at all. Reading `glmocr/api.py` + `config.py` + `ocr_client.py`:
"self-hosted" mode means it is an **HTTP client** that POSTs to
`http://{ocr_api_host}:{ocr_api_port}/v1/chat/completions`, defaulting to
`localhost:5002`. Only PP-DocLayout-V3 runs locally. So `run.sh` has no `glm_ocr` case
and `parse()` will fail with a connection error until one is added: serve
`zai-org/GLM-OCR` from `models/vllm_server` and point `GlmOcr(ocr_api_host=..., ocr_api_port=...)`
(or the `api_url` kwarg) at it. Note its `transformers>=5.3.0` floor is far above the
4.57.x the transformers-based adapters pin — that's fine, the venvs are independent.

`unlimited_ocr`'s `trust_remote_code` modeling file imports **matplotlib**, which the
pyproject did not list; `transformers.check_imports` hard-fails at `from_pretrained`
before any weight is touched. Added to `models/unlimited_ocr/pyproject.toml`.

**`unlimited_ocr` needs a capped `max_length`; the card's 32768 is a landmine.** Some
pages never emit EOS and decode until they hit whatever ceiling they are given. Measured
on `Complex_table_layouts` page_0008: still generating after 8 min under the card's
32768, and under an 8192 cap it emitted exactly 7285 tokens = `8192 - 907` (the prompt is
~907 image tokens). It consumes the ceiling, whatever it is. Healthy pages emit 823-1500
output tokens, so `max_length=4096` (now the default, override with
`UNLIMITED_MAX_LENGTH`) leaves ~2x headroom and bounds a runaway to ~70s. Decoding is
greedy (`temperature=0` -> `do_sample=False`), so the cap **cannot** alter a page that
terminates on its own — verified: page_0000 is byte-identical under 32768 and 8192. That
also means a killed run's existing checkpoints stay valid across a cap change, as long as
none of them hit the old cap. Rate on dense tables: **6 of 32 pages hit the cap.**

**For `unlimited_ocr`, wall-time is the degeneration signal, not `total_chars`.** Unlike
`got_ocr` (whose loops *inflate* char counts), this adapter runs `infer(save_results=True)`
and measures `chars` on the model's *post-processed* output files, not the raw decode.
The post-processor discards most of a degenerate span, so a capped page can report a
*small* char count: page_0028 burned the full 68s cap and emitted 770 chars; page_0025,
69.9s -> 3301 chars. Capped pages are therefore **under**-counted, not over-counted. Find
them with `seconds > 55` in the per-page meta, not with the repeat regex (which misses
page_0014 and page_0026 entirely).

**Do not raise `--batch-size` on `unlimited_ocr` (or any adapter that doesn't override
`process_batch`).** The GPU really is underutilized — sampled at 1 Hz during generation
it plateaus at 24-29% and sits at 0% between pages, 8.5 GB of 46 GB — but the flag
cannot fix it and actively corrupts the metrics. `Adapter.process_batch` *defaults to a
Python loop over `process_page`*, so `--batch-size 8` does the same 8 sequential
forwards, and then `harness.run()` divides the batch wall-time by 8 and reports an 8x
better `seconds_per_page` for identical work. Underneath, the model's `infer()` is
hardcoded to batch 1 anyway (`input_ids.unsqueeze(0)`, `images=[(crop, ori)]` at
`modeling_unlimitedocr.py:1027`); its `forward()` does loop over a batch dim, so a real
batched `infer()` is *possible* but means hand-rolling left-padding + attention mask +
padded `images_seq_mask` — precisely the code path where GOT-OCR's missing
`eos_token_id` silently corrupted every page. The model's own throughput answer is
`infer_multi()` (whole pdf, one forward), already wired up as `UNLIMITED_MULTI=1`.
Low GPU util at batch 1 is expected for decode-bound VLMs and is *not* a bug to chase.

Also unverified: `mineru` and `chandra` both `shutil.copy(mds[0], ...)` where
`mds = sorted(work_out.rglob("*.md"))` — i.e. whichever `.md` sorts first. Confirm that
is the right file. Both also leave their intermediate `outputs/<model>/<stem>/` work
dirs behind. (Their resume-drops-skipped-pdfs-from-`summary.json` bug is fixed.)

**`total_chars` is not a quality metric and is not comparable across models.** Each model
emits a different native format, and `total_chars` counts every byte of markup. On the
same 7-page `printouts`, stripping tags gives:

| model | raw | actual text | % text |
|---|---|---|---|
| `paddleocr_vl` | 62,464 | **10,760** | 17.2% |
| `lightonocr` | 16,082 | 9,349 | 58.1% |
| `dots_ocr` | 11,588 | 6,900 | 59.5% |
| `unlimited_ocr` | 14,164 | 6,112 | 43.2% |
| `got_ocr` | 433 | 427 | 98.6% |

`paddleocr_vl` looks 4x better than `lightonocr` on `total_chars` and is really only
~15% better on text — it puts `style='text-align: center; word-wrap: break-word;'` on
*every* `<td>`. `compare.py` now reports `visible_chars` and `pct_text` alongside.
Caveats: the strip is regex `<[^>]+>`, so it does **not** remove got_ocr's mathpix LaTeX
or the `<|det|>` grounding tags, which stay overcounted. It is a markup-inflation
detector, not a scoring function. `pct_text` is computed against the `.md` file, not
against `summary.json`'s `total_chars` — that field sums per-page counts while
`combine()` joins pages with `"\n\n"`, so the naive ratio exceeds 100% (got_ocr: 101.4%).

**The final table has two different memory columns, on purpose.** Per-page models go
through `harness.combine()` and report `max_gpu_mem_mb` — a genuine peak, the max over
every page's sample. The per-pdf adapters (`mineru`, `chandra`, `unlimited_ocr` multi
mode) report `gpu_mem_mb` from a *single* `nvidia-smi` sample taken after the pdf
finishes, by which point memory may already be freed. `compare.py` therefore shows both
columns, each half-populated. Do not merge them — they are not the same measurement, and
a unified "peak" column would silently overstate the per-pdf models' footprint.

## Working agreements

- Run one model end-to-end before starting the next; never leave two venvs resident.
- Smoke-test on `--smoke --pdfs printouts` (7 pages, ~2 min) before committing to a full run.
- Verify a run by page count (`32/3/12/14/7`) *and* by eyeballing the markdown, not by
  exit code alone.
- Never delete `outputs/`. Reclaim only venvs, the uv cache, and HF weights.
