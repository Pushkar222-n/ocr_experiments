# Context for Claude Code

Read this first. It is the handoff doc for this repo: what we are doing, what the pod
can and cannot run, what is already done, and what is left. `README.md` has the
user-facing detail; this file has the state and the hard-won gotchas.
**`CHANDRA.md` is the operator runbook for the winning model** — how to start it, run it,
tune it, and free the VRAM/disk. Read that one if you just want to *use* chandra.

## STANDING DECISIONS (do not relitigate these)

These are the user's settled calls. They live here, in the repo, because the pod (and
anything under `/root/`) is ephemeral and will be deleted.

1. **`chandra` is the model we are going with.** The benchmark is over; chandra won on the
   document classes that matter here. Everything below is about operating it, not choosing it.
2. **Page-rotation correction (`harness.orient_pdf`) is ALWAYS ON.** It is a permanent
   default, never an A/B variable. Hold it on in *both* arms of any future experiment.
   `--no-orient` exists as an escape hatch; do not use it for a benchmark row.
3. **The orientation classifier must stay on the CPU.** It is called from an adapter whose
   decoder already holds ~85% of the card; a second CUDA context there is an OOM waiting to
   happen. Three independent layers enforce this (see the orientation section below).
4. **Keep the vLLM server resident** (`scripts/serve_chandra.sh`). Loading chandra costs
   ~7 min; `run.sh` attaches to a live server on :8200 instead of reloading.

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
| `mineru` | **~11.4** | n/a* | **Run it on vLLM — the transformers engine changes the *output*, not just the speed (see below).** 201,262 chars, and **128,299 *visible* — the most real text any model here extracts** (17% clear of the next best). One of only **two** models that read a flowchart's topology (`chandra` is the other, and is better at it). Native output: markdown + `content_list.json`. Row comes from `outputs/mineru/output_vllm/`; the 72.4 s/page transformers run in `outputs/mineru/` is a kept artifact, **not** the benchmark row |
| `chandra` | ~9.8 | 38.0 GB† | 180,288 chars / 109,101 visible. Carries its **own** vLLM (~14 GB) in its venv — run it **last**, after `models/vllm_server/.venv` is reclaimed. **Best diagram parser in the field**: recovers 24 nodes / 55 edges / 10 feedback loops on `Flowchart`, beating `mineru` (21/40/**0**) — but fences the graph *bare* instead of as ` ```mermaid `, so it will not auto-render. Verified: `shutil.copy(mds[0])` is safe here — `rglob("*.md")` finds exactly one file, so there is no ambiguity about which markdown is taken |

| `glm_ocr` | ~9.6 | **38.3 GB** | **Heaviest VRAM in the set** (and that is *with* the layout model on cpu). 147,321 chars. Native format is **HTML** tables, not markdown pipes -> markup-inflated: `pct_text` runs 39-83% (worst on `printouts`: 12,023 raw -> 4,727 visible). Served on :8300 from `models/vllm_server`; arch resolves as `GlmOcrForConditionalGeneration`, loads clean. **Skips diagrams** — see below |
| `surya` | **~4.1** | 24.6 GB | **Fastest model in the set** (68 p in 4.6 min). 112,004 chars. Served from `models/vllm_server` on :8100; arch resolves as `Qwen3_5ForConditionalGeneration`, loads clean — no config patch needed. Only model reporting a **confidence** score — but read the trap below before using it |

**`surya`'s confidence is NOT a coverage metric, and it will mislead you.** It is the one
model here that self-reports quality, which makes it tempting as a quality gate. Do not
use it as one. Two things are true:

- **It is page-level, not per-block.** Every block on a page carries an *identical*
  confidence value (verified: `distinct_confidences=1` on every page checked). The
  adapter's `mean_confidence` is therefore averaging copies of a single number, not
  aggregating block scores. It cannot tell you *which* block was hard.
- **It scores what surya chose to read, never whether it read everything.** On `Flowchart`
  surya dropped the entire diagram — 571 chars vs mineru's 5069 — and still reported
  **0.947**, indistinguishable from the 0.95-0.98 it reports on documents it reads fully.
  A page where the whole diagram silently vanished looks healthy.

Mechanism, same layout-stage collapse as `dots_ocr`/`unlimited_ocr`/`paddleocr_vl`: surya
labels the diagram and **emits no html at all** for it. Across all 68 pages —
**`Picture` 45/45 empty, `Figure` 6/6 empty, `Diagram` 3/3 empty**, every one. It never
attempts them; it renders `![]()` and moves on. That is a layout decision, not a decode
failure, and no confidence number will ever flag it.

\* `mineru`'s VRAM is not measurable the same way as the others: under `vlm-http-client`
the weights live in a vLLM server that preallocates its KV pool to
`--gpu-memory-utilization` (0.7 -> ~32 GB), so the sampled 32.2 GB is a *reservation*, not
demand. The transformers run measured 13.7 GB of genuine demand, but that engine is not
the one we benchmark. Leave the cell blank rather than print a misleading number.

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
| `mineru` (vLLM) | **5069** | **4842** | yes — **and reads it anyway, as mermaid** |
| `lightonocr` | 4846 | 4588 | no — reads everything |
| `unlimited_ocr` | 684 | 683 | yes — `![](images/0.jpg)` |
| `dots_ocr` | 626 | 618 | yes — `Picture`, no `text` key |
| `paddleocr_vl` | 609 | 161 | yes — `<img src=...>` |
| `surya` | 571 | ~560 | yes — `Picture`/`Diagram`, **empty html**, conf still 0.947 |
| `got_ocr` | 234 | 236 | no, but degenerate |

**`mineru` is the counter-example that breaks the "layout stage ⇒ skips diagrams" rule.**
It has a layout stage and reads the diagram anyway. So the rule is not "layout pipelines
cannot read diagrams" — it is that most of them *choose* not to. That is a product
decision, not an architectural limit, and mineru proves the ceiling is higher.

**CHECKED — `glm_ocr` skips diagrams too, as predicted.** 288 raw / 286 visible chars over
the 3 `Flowchart` pages: it crops the diagram out (`![Image 0-0](imgs/cropped_page0_idx0.jpg)`)
and never reads inside it. Second-worst in the field. Corollary: on such documents
`total_chars` and `visible_chars` measure *whether the model has a layout stage*, not how
well it reads — and no tuning moves a model across that line.

(Cosmetic bug, no effect on any metric: every glm_ocr page emits the *same* image filename
`cropped_page0_idx0.jpg`, because the adapter saves each page into a fresh
`TemporaryDirectory` it then discards — it only keeps the `.md`/`.json`. So the image links
in `outputs/glm_ocr/*.md` are all dangling. We count text only, so this does not touch the
numbers, but the markdown is not self-contained.)

**The final scoreboard on `Flowchart`, all 9 models.** An earlier version of this file
claimed `mineru` was the *only* model that reads a flowchart's topology. **That was wrong**
— it was true of the first 8, and `chandra`, run last, breaks it. Two models recover the
process graph, and **`chandra` recovers more of it than `mineru` does**:

| model | raw | visible | reads the diagram? |
|---|---|---|---|
| `chandra` | **7751** | **7723** | **yes — mermaid, incl. the feedback loops** |
| `mineru` (vLLM) | 5069 | 4842 | **yes — mermaid, but forward-only** |
| `lightonocr` | 4846 | 4588 | text only — it has *no* layout stage, so it reads the pixels |
| `unlimited_ocr` | 684 | 683 | no |
| `dots_ocr` | 626 | 618 | no |
| `surya` | 571 | 554 | no (and still reports 0.947 confidence) |
| `glm_ocr` | 288 | 286 | no |
| `got_ocr` | 234 | 236 | no (degenerate) |
| `paddleocr_vl` | 609 | 161 | no |

Topology actually recovered, over the same 3 pages:

| | graphs | nodes | edges | **feedback loops** | fence |
|---|---|---|---|---|---|
| `chandra` | 3/3 | **24** | **55** | **10** | ` ``` ` (bare) |
| `mineru` | 3/3 | 21 | 40 | **0** | ` ```mermaid ` |

**`chandra` captures the QC feedback loops; `mineru` misses all of them.** These documents
are pharma manufacturing flowcharts, where sampling/approval steps cycle *back* into the
main process line (`I1 --> I`, `J1 --> J`). `mineru` flattens them into a forward-only
chain. In this document class those loops are the quality-control process — dropping them
loses the point of the diagram.

The one thing `mineru` does better is *packaging*: it fences the block as ` ```mermaid `
(inside a `<details>`, alongside the extracted image), so it renders downstream. `chandra`
emits its better graph in a **bare code fence**, so nothing will auto-render it — a
consumer has to sniff for `graph TD` itself. That is a packaging detail, not a modelling
one, and it is trivially fixable on our side.

**So: if diagram fidelity matters, the answer is `chandra` first, `mineru` second, and
nothing else is in the conversation.** Note both have a layout stage and read the diagram
anyway — so "layout pipeline ⇒ skips diagrams" is a *choice* 6 of the 8 layout models make,
not a limit they hit. `lightonocr` scores high here only because it has no layout stage at
all and reads the whole page as pixels; it returns the node text as loose prose, with none
of the graph structure.

**ANSWERED for `mineru`, and it breaks the pattern: it is the only model that reads the
flowchart's *topology*.** It has a layout stage and still does not skip the diagram — it
emits the process graph as **mermaid** (`graph TD`, `A[...] --> B[...]`) inside a
`<details>` block, alongside the extracted image. Nothing else here recovers the arrows.
The trap: on raw `total_chars` mineru (5069) "loses" to `lightonocr` (4916 with **no**
layout stage, which merely flattens the diagram into loose text). Char count actively
**mis-ranks this document** — do not let the final table imply mineru underperformed.

**mineru must be benchmarked on vLLM, not transformers — the engine changes the OUTPUT,
not just the speed.** This is the sharpest gotcha in the repo. `mineru_vl_utils` sets
per-block-type sampling params (`mineru_client.py:69-75`): `chart`, `image`, `table`,
`equation` and `[default]` all decode with **`presence_penalty=1.0` +
`frequency_penalty=0.05`**. The transformers client **never references either field** —
they appear nowhere in `transformers_client.py`, and the library admits it at
`base_client.py:32-33`: `# not supported by hf`. The http client sends both
(`http_client.py:273-276`). So the transformers path silently drops half of MinerU's own
decoding recipe — precisely the half that suppresses repetition and keeps the decoder
emitting distinct tokens, which is what a long mermaid graph of many distinct nodes needs.

Measured over the **full 68-page set**, same weights, engine the only variable
(`outputs/mineru/` vs `outputs/mineru/output_vllm/`):

| pdf | pages | s/pg transformers | s/pg vLLM | speedup | chars tf | chars vLLM |
|---|---|---|---|---|---|---|
| `Complex_table_layouts` | 32 | 92.6 | **5.43** | **17.1x** | 122,100 | 131,172 |
| `Handwritten` | 14 | 38.2 | **10.60** | 3.6x | 26,056 | 26,223 |
| `Formulas_with_tables` | 12 | 34.5 | **11.79** | 2.9x | 21,073 | 20,944 |
| `printouts` | 7 | 56.7 | **23.93** | 2.4x | 18,201 | 17,854 |
| `Flowchart` | 3 | 205.3 | **47.27** | 4.3x | 2,788 | **5,069** |
| **TOTAL** | **68** | **72.4** | **11.37** | **6.4x** | 190,218 | **201,262** |

Wall clock **82.1 min -> 12.9 min**, and extraction went *up* by 11,044 chars — the speed
costs nothing. On `Flowchart` the transformers path recovered **2 of 3** mermaid graphs;
vLLM recovers **3 of 3**.

The **per-document speedup varies 2.4x-17.1x, and that spread is an artifact, not signal**:
the mineru CLI pays ~100 s of *fixed* startup (imports, layout model, its internal FastAPI)
before decoding a single page, and that dominates a short document. Only
`Complex_table_layouts` (32 p) is long enough to amortize it, which is why it alone shows
the engine's true margin. Read the 17.1x, not the 2.4x, as the real cost of the
transformers path — and never quote a s/page from a 3-page document.

On page 3 the transformers path **gave up on the graph and emitted a prose summary**
("This flowchart illustrates the manufacturing process for a quality assurance
department...") where vLLM produced a full 21-node mermaid graph. That is a decode
degeneration caused by the missing penalties, not a layout call. **The transformers run
therefore under-measures mineru on both quality and speed, and is invalid as a benchmark
row.** Its outputs are kept at `outputs/mineru/` as a documented engine artifact; the row
that counts is `outputs/mineru/output_vllm/` (`--out-tag` keeps them from colliding).

Do not read `max_gpu_mem_mb` across the two: under `vlm-http-client` the weights live in a
vLLM server that preallocates its KV pool to `--gpu-memory-utilization` (0.7 -> ~32 GB),
so the number is a reservation, not demand. Only the timings are comparable.

Also do not read the 3-page `47.3 s/page` as mineru's true rate: the mineru CLI pays a
large *fixed* startup (~100 s of that 141.8 s went to imports, the layout model and its
internal FastAPI before a single page was decoded). It amortizes over a long document —
which is the whole reason the benchmark row comes from the full 68-page run.

**vLLM 0.19.1 cannot load MinerU2.5 straight from the hub — patch the config.** The
architecture *is* supported (it resolves `Qwen2VLForConditionalGeneration`), but weight
loading dies:

    ValueError: Following weights were not initialized from checkpoint:
                {'language_model.lm_head.weight'}

MinerU2.5 **ties** lm_head to the input embeddings — the checkpoint has 681 tensors,
`model.embed_tokens.weight` present, **no lm_head at all**. vLLM only expected one because
`qwen2_vl.py:1251` hands the *flat* top-level config to `Qwen2ForCausalLM`, which checks
`config.tie_word_embeddings` at `qwen2.py:555`. MinerU2.5 ships the **transformers-v5
nested** layout: the flag is `true` inside `text_config` and there is **no top-level key**,
so `PretrainedConfig` defaults it to `False` and vLLM builds a standalone head it has no
weights for. Transformers reads the nested value and ties correctly — which is exactly why
the in-process baseline worked and vLLM did not.

`scripts/mineru_vllm_model.py` fixes this: it symlinks the HF snapshot into
`work/mineru2.5_vllm/` and rewrites **only** `config.json`, hoisting the flag to the top
level (no weights copied; the volume has no hardlinks but symlinks work). It does not
change the model — it makes vLLM tie the head the way transformers already did. It
hard-fails rather than guessing if a future checkpoint drops the nested field. `run.sh`
serves that directory, never the hub id. **Do not patch the HF cache in place — a
re-download silently reverts it.**

**Expect this class of bug again on `surya` and `chandra`.** Same vLLM 0.19.1, same
loader. Any model whose config uses the transformers-v5 nested layout will hit the same
top-level-vs-nested mismatch, and any model with tied embeddings will hit it on `lm_head`.
If a `vllm serve` dies during weight loading, check the config shape before assuming the
architecture is unsupported.

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

1. `mineru` — **transformers baseline done (68 pages, 72.4 s/page) but superseded; the
   vLLM re-run into `outputs/mineru/output_vllm/` is the row that counts.** See the
   engine-changes-the-output finding above. `MINERU_VLLM=1 ./run.sh mineru --backend
   vlm-http-client --out-tag output_vllm`.
2. `surya` — **done** (68 pages, 4.06 s/page, the fastest model here). Loaded on vLLM with
   no config patch needed. See the confidence-is-not-coverage trap above.
3. `glm_ocr` — **needs a served endpoint; see below.** Run it right after `surya` so the
   `models/vllm_server/.venv` gets built once, not twice.
4. `chandra` — carries its **own** vLLM (~14 GB) in its venv; `run.sh` serves on :8200.
   Run the others first, reclaim fully, then `chandra` — never two vLLMs resident.
5. `python scripts/compare.py`

## `lightonocr` on vLLM: 5.4x faster, output unchanged

Same weights, same sampling (temp 0.2 / top_p 0.9 / 4096 tokens — the constants are shared
by both code paths in `models/lightonocr/run.py`, deliberately, so the engine is the only
variable). `LIGHTON_VLLM=1 ./run.sh lightonocr --out-tag output_vllm`.

| pdf | pages | transformers | vLLM | speedup |
|---|---|---|---|---|
| `Complex_table_layouts` | 32 | 20.72 | **3.92** | 5.3x |
| `Handwritten` | 14 | 11.97 | **2.01** | 6.0x |
| `Formulas_with_tables` | 12 | 7.91 | **1.55** | 5.1x |
| `printouts` | 7 | 13.27 | **2.18** | 6.1x |
| `Flowchart` | 3 | 6.34 | **1.70** | 3.7x |
| **TOTAL** | **68** | **15.26** | **2.83** | **5.4x** |

Chars 218,449 → 221,578 (**+1.4%** — unchanged). Wall **17.3 min → 3.2 min**. Degeneration
**3/68 → 4/68 pages**, i.e. the same rate — and most of those flags are *false positives*
(`'etected<br>Not D'` is a real table repeating "Not Detected" cells). **Loops are a
property of this model** (temp 0.2, no repetition penalty), not of the engine; the LaTeX
loop seen in the smoke run did not even recur in the full run, because sampling is
stochastic. At 2.83 s/page **`lightonocr` is now the fastest model in the benchmark.**

Unlike `mineru`, nothing about the *output* changed here — which is the expected result
when the sampling params really are identical, and is the control that makes the mineru
finding credible: there, the output changed because the engine silently dropped half the
recipe. Served via `models/chandra`'s venv (it already carries vllm 0.19.1), so this cost
**no new disk** — see the `VLLM_PROJ` note in `run.sh` if that venv is ever reclaimed.

**Worth doing next:** `got_ocr`, `dots_ocr` and `unlimited_ocr` are still on in-process
transformers and are very likely understated by the same 3-6x. The bottom of the speed
table is measuring *un-migrated engines*, not slow models.

## THE BENCHMARK IS COMPLETE — all 9 models, 68 pages each (state as of 2026-07-11)

| model | engine | s/page | raw chars | **visible** | %text |
|---|---|---|---|---|---|
| `lightonocr` | **vllm** | **2.8** | 221,578 | 115,566 | 52% |
| `surya` | vllm | 4.1 | 112,004 | 99,622 | 89% |
| `got_ocr` | transformers | 8.2 | 114,021 | 109,394 | 96%\* |
| `glm_ocr` | vllm | 9.6 | 147,321 | 83,292 | 57% |
| `chandra` | vllm | 9.8 | 180,288 | 109,101 | 61% |
| `mineru` | **vllm** | 11.4 | 201,262 | **128,299** | 64% |
| `dots_ocr` | transformers | 21.2 | 154,836 | 91,283 | 59% |
| `unlimited_ocr` | transformers | 25.0 | 184,140 | 97,520 | 53% |
| `paddleocr_vl` | paddle | 25.5 | **422,808** | 88,841 | **21%** |

**The s/page column compares engines as much as models.** Both models we moved to vLLM got
5-6x faster on identical weights and sampling. The four still running in-process are very
likely understated by the same factor — read the bottom of this table as "not yet migrated",
not "slow". `compare.py` prints which engine's row it used per model.

\* `got_ocr`'s 96% is an artifact — the tag-stripper is a regex `<[^>]+>` and does not
remove its mathpix LaTeX, and it degenerates into token loops on 5 of 68 pages. Do not read
it as the cleanest output. `paddleocr_vl` is the opposite artifact: the highest raw count in
the benchmark collapses to 8th of 9 on visible text, because it puts inline CSS on **every**
`<td>`. **Rank on `visible`, never on `total_chars`.**

Headlines: **`mineru` extracts the most real text** (128,299 visible, 17% clear of the next
best) *provided it runs on vLLM*. **`surya` is 2.4x faster than anything else** and the only
model reporting confidence — which is not a coverage metric, see the trap above.
**`chandra` is the best diagram parser.** `paddleocr_vl` is the trap.

Remaining work: none. `python scripts/compare.py` regenerates `outputs/comparison.json`
(it prefers `mineru`'s `output_vllm/` tagged row and says so on stderr).

**Worth doing if this is ever re-run:** `run.sh` unconditionally starts a vLLM server and
kills it on exit via the `trap`, so a smoke test followed by a real run pays the server
startup **twice** (~3 min each once weights are cached). It could `curl` the port first and
reuse a server that is already up. The smoke test is still worth its cost — it catches the
*adapter-side* failures a server start cannot (got_ocr's silent `eos_token_id` corruption,
glm_ocr silently posting to Zhipu's cloud) — but it need not pay for a second engine load.

**Disk bit us mid-benchmark and will again.** The RunPod volume quota is ~50 GB and is
*not* visible in `df` (which reports the whole MooseFS cluster, showing hundreds of TB
free — ignore it). Use `du -sh /workspace`. A surya `uv sync` was silently killed at 29 GB
used, leaving a half-written venv and a log that just stopped — **no error message, no
"No space" line**. If a sync dies quietly, suspect the quota first.
`scripts/reclaim.sh mineru opendatalab/MinerU2.5-Pro-2605-1.2B` freed 22.9 GB (11 GB venv +
10.6 GB uv cache + weights) and the retry then worked first try. Note `models/vllm_server/.venv`
alone is **15 GB** — keep it until `glm_ocr` is done (both serve through it), and reclaim it
*before* `chandra`, which carries its own ~14 GB vLLM. Two vLLM venvs resident will not fit.

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

## Closed / paid API models (state as of 2026-07-12)

A separate experiment: four hosted document-parse APIs on their **balanced** (mid) tier,
run over the *same* 68-page `sample_set`. Code: `models/closed_apis/run.py` (one uv
project, plain `requests`, no vendor SDKs). Keys live in `.env` (gitignored — it holds
live secrets and this repo is public). Outputs are stored **separately** under
`outputs/closed/<provider>/` (same file shapes as the open models: `<stem>.md`,
`<stem>.native.json` = full API response, `<stem>.metrics.json`, `summary.json`) so the
closed experiment never mixes into the open `outputs/<model>/` tree.

Run: `cd models/closed_apis && uv run python run.py all --concurrency 3` (or a single
provider: `... run.py datalab`). `--smoke` writes to `outputs/closed/_smoke/`. Per-pdf
checkpoint: a finished `<stem>.md` is not re-fetched (these cost money). Each provider
takes the whole PDF in one job and returns all pages, so concurrency is across the 5 PDFs.

Runs (each = provider + tier, each in its own `outputs/closed/<run>/`):
- `mistral` → `mistral-ocr-latest` (their only OCR model)
- `datalab` → **legacy** `/api/v1/marker`, `use_llm=false` (kept; deprecated endpoint)
- `datalab_balanced` / `datalab_accurate` → current `/api/v1/convert`, `mode=balanced|accurate`
- `llamaparse` → `tier=agentic` + `version=latest` (their "Balanced" preset)
- `llamaparse_agentic_plus` → `tier=agentic_plus` + `version=latest` (their premium preset)
- `landing_ai` → ADE `dpt-2-latest` (their only parse model)

Not every provider has a real tier ladder: **Mistral and Landing AI each expose one model**
(Mistral's other SKU, Document AI, is a different product; Landing AI's plans change the
credit *price*, not the model). Only Datalab and LlamaParse offer a genuine fast/mid/premium
choice — and in both cases, **the premium tier turned out not to be worth it** (see below).

Full 68-page results. Each run is a provider+tier and gets its **own** directory under
`outputs/closed/<run>/`, so tiers never overwrite each other.

| run | s/page | **visible** | $/68p | $/1k pages | $ per 10k visible | cost source |
|---|---|---|---|---|---|---|
| `datalab` (legacy /marker) | 0.96 | **149,941** | $0.290 | $4.26 | **$0.019** | API-reported |
| `datalab_balanced` | 1.10 | 147,416 | $0.290 | $4.26 | $0.020 | API-reported |
| `mistral` | **0.49** | 106,909 | $0.272 | $4.00 | $0.025 | list rate |
| `datalab_accurate` | 2.43 | 148,848 | $0.680 | $10.00 | $0.046 | API-reported |
| `llamaparse` (agentic) | 14.31 | 144,455 | $0.850 | $12.50 | $0.059 | list rate (free tier) |
| `landing_ai` | 2.04 | 90,404 | $2.040 | $30.00 | $0.226 | metered credits |
| `llamaparse_agentic_plus` | 10.48 | **167,108** | $3.825 | $56.25 | $0.229 | list rate (free tier) |

Total spent across all closed runs: **$8.25**.

**Datalab is the best value in the field, open or closed** — more visible text than the best
open model (`mineru`, 128,299) at $4.26/1k and ~1 s/page.

### Three findings that only appear once you run the tiers

**1. Datalab's "accurate" mode buys nothing here.** It costs **2.35x balanced** ($10/1k vs
$4.26/1k) and is 2.2x slower, for **148,848 visible chars vs balanced's 147,416 (+1%) — and
*less* than the legacy endpoint's 149,941**. Identical flowchart edges (39). On this document
set the premium tier is not justified. Do not assume the top tier is better; measure it.

**2. LlamaParse Agentic Plus extracts the most text of anything in this repo and is a WORSE
diagram parser than the tier below it.** At 45 cr/pg ($56.25/1k, **19x Datalab**) it produces
167,108 visible chars — the highest, open or closed. But on `Flowchart` it **flattens the
diagram into 72 table rows and emits ZERO graph syntax** (0 mermaid, 0 `-->`), where the
cheaper `agentic` tier emits **3 real ```mermaid blocks and 55 edges**. Paying 4.5x more
bought more text and *destroyed the topology*. If diagrams matter, the cheaper tier is
strictly better.

**3. The legacy `/api/v1/marker` endpoint bills identically to `mode=balanced`** — 13/2/5/6/3
cents on the same 5 pdfs, exactly — and extracts marginally *more* text. So migrating to
`/convert` is an API-lifecycle decision, not a cost or quality one.

### Pricing, verified 2026-07-12

Costs are **measured wherever the provider reports them**, not estimated:
- **Datalab `/api/v1/convert` returns the real charge** in `cost_breakdown.final_cost_cents`
  (rounded up to the nearest cent *per request*). So does the legacy `/marker` response — we
  simply were not reading it, which is why an earlier version of this file carried a bogus
  "~$3/1k, unconfirmed" guess. **There is no unconfirmed Datalab rate any more.**
- **Landing AI: 3.00 credits/page is MEASURED**, not documented — the API returns
  `metadata.credit_usage`, and it is exactly 3.00 cr/pg on all five documents (204 cr / 68 p).
  Landing AI never publishes a credits-per-page figure. Credit price ($1 = 100 credits on
  Explore) *is* documented. So the $2.04 is tracked-usage x documented-price.
- **LlamaParse's free tier (10k credits/month) absorbed both runs**, so its API honestly
  reports **0 credits**; its cost column is the published list rate, not a metered charge.
- Mistral bills per page at a published rate; nothing to measure.

| provider | tier | $/1k pages | 68p | |
|---|---|---|---|---|
| `mistral` | OCR (`mistral-ocr-latest`) | $4.00 | $0.272 | **<- run** |
| `mistral` | Document AI | $5.00 | $0.340 | not run |
| `datalab` | `/convert` mode=fast | *unknown* | — | not run (API reports it) |
| `datalab` | `/convert` mode=balanced | $4.26 | $0.290 | **<- run** |
| `datalab` | `/convert` mode=accurate | $10.00 | $0.680 | **<- run** |
| `llamaparse` | Fast (1 cr/pg) | $1.25 | $0.085 | not run |
| `llamaparse` | Cost-effective (3 cr/pg) | $3.75 | $0.255 | not run |
| `llamaparse` | **agentic** (10 cr/pg) | $12.50 | $0.850 | **<- run** |
| `llamaparse` | **agentic_plus** (45 cr/pg) | $56.25 | $3.825 | **<- run** |
| `landing_ai` | ADE `dpt-2-latest` (3 cr/pg) | $30.00 | $2.040 | **<- run** |

Credit rates: LlamaParse **$1.25 / 1000 credits**; Landing AI Explore **$1 = 100 credits**
($0.01/cr; the $250/mo Team plan only reaches $0.0091/cr = ~$27/1k).

`run.py prices` prints this table and dumps it to **`outputs/closed/pricing.json`** for later
analysis. `run.py reprice` recomputes stored costs offline — never re-call the APIs just to
refresh a cost column, that means paying for the same pages twice.

### Two API traps, both hit live

- **Datalab `/api/v1/marker` is the OLD endpoint.** The current one is
  **`POST /api/v1/convert`** with `mode=fast|balanced|accurate`, and it additionally returns
  `cost_breakdown` and `parse_quality_score` (the latter came back `None` for
  `output_format=markdown` on every request, so it is not usable as a quality gate here).
- **LlamaParse: pinning a `model` is dead, and a `tier` needs a `version`.**
  `model="anthropic-sonnet-4.0"` — still what the Agentic Plus docs show — is **RETIRED** and
  422s. The API itself says to migrate to tiers. But `tier=agentic_plus` alone fails
  *asynchronously* with `MISSING_VERSION_FOR_TIER`: the upload is accepted, then the job
  errors. You must send **`tier` + `version`** (`version=latest`, or pin a date like
  `2026-01-08` to freeze the config).

`scripts/compare.py` folds the closed rows into `outputs/comparison.json` (tagged
`closed:true`, one level deeper under `outputs/closed/*/summary.json`), so the frontend
metrics view reads one file. The frontend groups them under a **paid API** section in the
compare picker (dashed chips, `$` flag), shows per-document cost/credits badges on each
pane, and adds a **Cost** bar card + `cost_usd`/`billed_pages`/`credits` columns in metrics.
Closed outputs are gitignored (under `outputs/`) like the open ones — carry them in the
resume bundle, not the repo.

## IN PROGRESS — chandra rotation-correction + tuned vLLM rerun (state as of 2026-07-13)

User request: chandra (already benchmarked, see the table above) was seen struggling on
rotated pages, especially in `Complex_table_layouts`. Three asks: (1) add PaddleOCR's
tiny PP-LCNet 4-class doc-orientation classifier as a rotation-correction pass and re-run
chandra over the whole `sample_set` with it; (2) turn on chandra's
`--include-headers-footers` flag (exists, confirmed against the actual CLI source,
defaults to *excluded*); (3) evaluate a list of vLLM serving flags an LLM suggested
(`--enable-prefix-caching`, `--mm-processor-kwargs`, `--max-model-len 18000`,
`--max-num-batched-tokens`, `--max-num-seqs`) and adopt only the ones that actually
matter on this A40. User also asked that the rotation logic live in the shared harness,
not bolted onto chandra alone, so any other model adapter can call it too, and that this
run be a tagged A/B (`--out-tag`), not an overwrite of the already-benchmarked baseline.

**All of this turned out to be traceable to datalab's own code, not guesswork** — worth
reading if you're re-deriving any of it. `chandra/scripts/vllm.py` (repo default branch
is `master`, not `main` — `git ls-remote`/the GitHub API `default_branch` field will
save you a 404) is chandra's **own** docker-based vLLM launcher, the one `run.sh` already
replaces with a plain `vllm serve` because this pod can't do docker-in-docker. It hardcodes
exactly the flags the user was asking about:

```
--max-model-len 18000
--mm-processor-kwargs {"min_pixels": 3136, "max_pixels": 6291456}
--enable-prefix-caching
--max-num-batched-tokens <GPU-scaled>   --max-num-seqs <GPU-scaled>   # H100=8192/64 baseline
```

The GPU-scaling formula in that file (ratio = your VRAM / 80GB, batched-tokens rounds
down to a power of 2, seqs rounds down to a multiple of 8) has no entry for `a40`, but
A40 (46 GB) and `l40s` (48 GB) land on the identical rounded result either way: **4096 /
32**. Its GPU list is `{h100, a100-80, a100, a100-40, l40s, a10, l4, 4090, 3090, t4}` —
add `a40` there first if this launcher is ever used directly instead of being replicated
into `run.sh`.

**Verified which of those four actually matter here, against this repo's own completed
chandra run data (`outputs/chandra/*.metadata.json`) and the model's real HF config**,
not just by trusting the vendor's H100-tuned defaults:

- **`--max-model-len 18000` — adopted, high confidence.** Without it vLLM auto-derives
  from `datalab-to/chandra-ocr-2`'s `config.json`: `max_position_embeddings: 262144`.
  Real data from the completed baseline run: max output `token_count` over all 68 pages
  is **4621** (`Complex_table_layouts` page 6); max plausible vision-token load is
  ~6144 (client caps images at 3072x2048px — see next point — and the processor's
  `patch_size=16`/`merge_size=2` gives (3072/16)x(2048/16)/4 = 6144). So real usage never
  goes near 10k tokens, let alone 262144. Added in `run.sh`'s chandra case.
- **`--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'` — added, but
  confirmed a no-op for us.** Fetched `chandra/model/util.py`'s `scale_to_fit()`: the
  chandra CLI itself already resizes every image to at most `(3072, 2048)` =
  **6,291,456px** before base64-encoding it for the server — the exact same number.
  The server-side clamp this flag sets can never bind, since the client-sent image is
  never larger. Kept it anyway (matches the vendor config exactly, free to add, and
  guards against a future chandra client change that stops pre-resizing) but do not
  expect it to move any number.
- **`--enable-prefix-caching` — deliberately NOT added.** Two independent reasons: (a)
  it's already default-on in vLLM 0.19.1's V1 engine, so the flag changes nothing; (b)
  fetched `chandra/model/vllm.py`'s `generate_vllm()` — the message content list puts
  the **image block first, the constant OCR prompt text second**. Since every page's
  image differs, the very first block in the sequence already fails to hash-match
  across requests, and vLLM's prefix-cache block hashing chains on prior blocks, so the
  identical trailing prompt text never gets a cache hit either. Datalab's own rationale
  ("prompt is constant across thousands of pages") only pays off for a usage pattern
  that revisits the *same* image with multiple prompts, or multi-turn chat about one
  image — not this repo's one-pass-per-page batch job.
- **`--max-num-batched-tokens 4096` / `--max-num-seqs 32` (vendor's A40-scaled values)
  — deliberately NOT added.** Our own client concurrency (chandra's `--batch-size`,
  used here at 16, CLI's own vllm-method default is 28) never gets remotely close to
  either vLLM default (8192 / 1024), so `max-num-seqs` is moot regardless of its value.
  Worse, chunked-prefill math: a single max-size page needs ~6300 prefill tokens
  (6144 vision + prompt), which the vendor's scaled-down 4096 would split across two
  scheduler steps where the current higher default does it in one — a plausible
  regression, not a win, for a workload that is many-modest-requests rather than
  many-long-chat-turns (which is what that H100 scaling was tuned for). Left at vLLM's
  own defaults.

**What changed in code:**

- `harness/ocr_harness/__init__.py`: added `classify_rotation(bgr_image)` and
  `orient_pdf(pdf_path, dpi=150, preview=True)`. Lazy-imports `paddleocr`'s
  `DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori", engine="onnxruntime")`
  — onnxruntime backend needs neither paddlepaddle nor a GPU (verified: `paddlex[ocr-core]`
  on PyPI has no paddlepaddle dependency at all; the framework is a separate, optional
  runtime the engine picks up if present). `orient_pdf` renders each page with pypdfium2,
  classifies it, and for any page with a nonzero angle writes a corrected copy of the pdf
  to `work/oriented/<stem>.pdf` using `pypdf`'s `page.rotate(-angle)`. **Sign convention
  verified from PaddleX's own source**, not assumed: `paddlex/inference/pipelines/
  doc_preprocessor/pipeline.py` calls `rotate_image(img, angle)` with the raw predicted
  label, no sign flip; `rotate_image` (in
  `paddlex/inference/pipelines/components/common/warp_image.py`) applies it via
  `cv2.getRotationMatrix2D(center, angle, scale)`, and OpenCV's convention is
  positive-angle-is-counter-clockwise. PIL's `Image.rotate(angle)` uses the same CCW
  convention, so the preview code matches PaddleX's own correction exactly; pypdf's
  `Page.rotate()` is clockwise, hence the `-angle` there. **Do not re-derive this from the
  HF model card** — it documents the four class labels but not the rotation direction;
  the pipeline source is the only place that actually disambiguates it.
- `harness/pyproject.toml`: added `numpy`, `pypdf>=4`, `paddleocr>=3.7`, `onnxruntime`.
  Added here (not per-model) so every model's venv gets `orient_pdf` for free through the
  `ocr-harness` editable dependency, per the user's ask that this be broadly reusable.
  Confirmed cheap: `uv sync --project harness` alone (no chandra, no torch, no vllm) pulls
  in ~4 GB (mostly paddlex + opencv-contrib-python + modelscope + onnxruntime) — small next
  to any single model's vLLM venv, and it was verified standalone before ever touching
  chandra's venv.
- `models/chandra/run.py`: calls `harness.orient_pdf(pdf)` before invoking the `chandra`
  CLI (opt out with `--no-orient`); always passes `--include-headers-footers`; added
  `--out-tag` (threaded into `output_root()`, same convention as mineru/lightonocr's
  vLLM A/Bs) so this run cannot overwrite or resume from the existing benchmarked
  `outputs/chandra/` baseline.
- `run.sh`: chandra's `vllm serve` call gained `--max-model-len 18000` and
  `--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'` (see above for
  which flags were and weren't added, and why).

**Verified against real data so far (no GPU used for any of this — the classifier runs
on CPU/onnxruntime):**

Ran `harness.orient_pdf` over all 5 `sample_set` pdfs via `uv run --project harness`
(standalone, ~4 GB venv, no torch/vllm). Only `Complex_table_layouts.pdf` has rotated
pages: **7 of 32** — pages 3, 4, 25, 26 at 270°, pages 8, 13, 14 at 90°, confidence
0.83–0.93. `Flowchart`, `Formulas_with_tables`, `Handwritten`, `printouts` all came back
clean (`any_rotated: false`, cached, `orient_pdf` is a pure passthrough for them). Visually
confirmed correct in **both** rotation directions by reading the before/after preview pngs
myself (`work/oriented_preview/Complex_table_layouts/page_0003_{before,after}.png` and
`page_0008_{before,after}.png`) — page 3 (270°) and page 8 (90°) both render upright after
correction, with no distortion or wrong-way rotation. The corrected pdf that chandra will
actually be pointed at is `work/oriented/Complex_table_layouts.pdf`; full report at
`work/oriented/Complex_table_layouts.json`.

## DONE (2026-07-14) — chandra oriented + tuned, full 68-page run

Run: `scripts/serve_chandra.sh` then `./run.sh chandra --out-dir chandra_oriented_optimized`.
Output: **`outputs/chandra_oriented_optimized/`** (a *top-level* sibling, not a tag under
`outputs/chandra/`, so `compare.py` picks it up as its own row and the benchmarked baseline
is untouched). 68 pages, 11.0 min, **9.72 s/page** (baseline 9.8 — unchanged).

| pdf | pages | base raw | new raw | base visible | new visible | Δ visible |
|---|---|---|---|---|---|---|
| `Complex_table_layouts` | 32 | 106,694 | 110,013 | 58,113 | 59,936 | +1,823 |
| `Flowchart` | 3 | 7,751 | 8,310 | 7,751 | 8,310 | +559 |
| `Formulas_with_tables` | 12 | 23,128 | 23,576 | 18,935 | 19,440 | +505 |
| `Handwritten` | 14 | 28,983 | 30,129 | 19,394 | 20,561 | +1,167 |
| `printouts` | 7 | 13,732 | 14,265 | 10,197 | 10,065 | −132 |
| **TOTAL** | **68** | 180,288 | **186,293** | 114,390 | **118,312** | **+3,922 (+3.4%)** |

**Attribute the gain carefully — the headline +3.4% is NOT all orientation.** Four of the
five pdfs have *zero* rotated pages, so their deltas can only come from
`--include-headers-footers` and decode nondeterminism. Only `Complex_table_layouts`
exercises rotation. Isolating it with chandra's own per-page `token_count`
(`*.metadata.json`), against the 25 unrotated pages of the *same* document as a control:

| | pages | base tok | new tok | Δ |
|---|---|---|---|---|
| **rotated** (3,4,25,26 @270°; 8,13,14 @90°) | 7 | 15,540 | 16,280 | **+4.8%** |
| unrotated (same pdf, control) | 25 | 49,882 | 49,548 | −0.7% |

**+4.8% on the corrected pages against −0.7% on the control is a real, attributable
signal** (best: page 14 +13.0%, page 8 +10.5%). It is modest, not transformative — chandra
was already partially coping with sideways pages. The `page_box` field proves the fix
reached the model: `[1588, 2246]` -> `[2246, 1588]`, i.e. the dims swapped, so chandra
genuinely rendered them upright.

`Flowchart` also improved: **36 -> 46 mermaid edges**, still 3/3 graphs.

**`--max-model-len 18000` is a REQUIREMENT on this A40, not a tuning nicety.** vLLM reports
`GPU KV cache size: 229,152 tokens`, against the model's declared 262,144-token context.
229,152 < 262,144, so without the flag vLLM **refuses to start** ("max seq len larger than
the maximum number of tokens that can be stored in KV cache"). A genuinely flag-free
control arm therefore *cannot exist* on this card — do not go looking for one. (This also
means the `CHANDRA_TUNED=0` arm in `run.sh` will not boot as-is; it is kept only to make
that failure reproducible.)

Correction to the flag analysis above: `--enable-prefix-caching` is **NOT** default-on in
vLLM 0.19.1 — the engine config logs `enable_prefix_caching=False`. The decision to omit it
still stands, but *only* on the image-first-block argument (chandra puts the per-page-unique
image before the constant prompt, so no page ever shares a prefix with another).

### Two operational traps hit while doing this — both cost ~20 min each

**1. Never `uv cache clean` while a `uv` process is live.** `uv run` probes the interpreter
through a temp file *inside* the cache. Deleting the cache mid-flight kills it with:

    error: Failed to query Python interpreter
      Caused by: No such file or directory at ".../.cache/uv/.tmpXXXXXX"

vLLM came up, served **0** requests, and the adapter never started — indistinguishable from
a slow model load. `run.sh` now calls `.venv/bin/python` / `.venv/bin/vllm` **directly**, so
it never touches the uv cache and cannot be broken this way. It also **skips `uv sync` when
the venv already exists** (`FORCE_SYNC=1` to override) — with the cache gone, a sync
re-downloads ~16 GB. Consequence of bypassing `uv run`: the venv's `bin/` is no longer on
`PATH`, so `run.sh` now exports it — without that, chandra's adapter dies with
`FileNotFoundError: 'chandra'` when it shells out to the CLI.

**2. paddleocr in chandra's venv makes vLLM log a scary traceback at every boot.** `paddlex`
registers a vLLM entry-point plugin, `register_paddlex_genai_models`, which fails to import.
vLLM logs a full `Failed to load plugin` traceback at ERROR — **then continues and serves
normally.** It is cosmetic. Do not chase it.

### Throughput: `--batch-size 16` is right, and bigger is WORSE (measured)

vLLM prints `Maximum concurrency for 18,000 tokens per request: 45.74x`, which *looks* like
the default `--batch-size 16` under-drives the card. **It does not.** Same pdf, same
resident server, only the flag changed:

| `--batch-size` | s/page | chars |
|---|---|---|
| **16 (default)** | **12.25** | 110,013 |
| 48 | 17.04 (**+39% slower**) | 110,533 |

That "45.74x" is computed against `--max-model-len` (18,000), **not against what a page
actually costs**. A page is ~6,100 vision tokens of prefill, so 32 concurrent pages is
~196k prefill tokens against a 229k-token KV cache — the scheduler starts preempting and
thrashes. Do not raise it.

**Corollary that matters for hardware shopping: the A40 is already saturated at batch 16.**
So a faster card would convert to real throughput rather than being wasted on an under-fed
GPU — see the VRAM/bandwidth note below.

### What chandra actually costs (for sizing a different GPU)

Measured from the vLLM log and the run's own `*.metadata.json`:

- **Weights: 8.61 GiB.** Everything else in the 39.5 GB you see on the card is KV-cache
  *reservation* (`--gpu-memory-utilization 0.85`), not demand. `Available KV cache memory:
  28.02 GiB -> 229,152 tokens`.
- **A page costs ~6,100 vision tokens + 1,725 generated (median; p95 3,329; max 4,608).**
  So ~8k tokens in flight per concurrent page, not 18,000.
- **Therefore chandra fits on a 24 GB card** (4090/3090): 8.61 GiB of weights leaves
  ~12-13 GiB of KV, ~100k tokens, ~9 pages in flight. Tight but workable at a lower
  `--batch-size`. A 32 GB 5090 is comfortable. **VRAM is not the blocker.**
- Decode is **memory-bandwidth bound**, so a faster card scales roughly with bandwidth
  (A40 = 696 GB/s). L40S 864 (~1.2x), 3090 936 (~1.3x), 4090 1008 (~1.4x), 5090 1792
  (~2.5x). These are *extrapolations, not measurements.* Ada/Blackwell also add FP8, which
  Ampere/A40 lacks — potentially another real factor, but quantizing an OCR model is
  untested here and risks accuracy.

**Read: a 4090/L40S buys ~20-40% and is probably not worth a big premium; the 5090 is the
only genuine step change.**

### Input resolution: TESTED — raising DPI makes chandra WORSE. Leave it at 192.

**Do not re-run this experiment.** `IMAGE_DPI=256` (which saturates the model's pixel cap —
74% more vision tokens than the default) was run over the full 68-page set into
`outputs/chandra_dpi256/`. It is **−1.2% on visible text and 35% slower**:

| pdf | dpi192 visible | dpi256 visible | Δ | 192 s/pg | 256 s/pg |
|---|---|---|---|---|---|
| `Complex_table_layouts` | 59,936 | 60,416 | +480 | 12.25 | 19.07 |
| `Flowchart` | 8,310 | 6,262 | **−2,048** | 16.21 | 13.62 |
| `Formulas_with_tables` | 19,440 | 19,754 | +314 | 5.48 | 6.07 |
| `Handwritten` | 20,561 | 20,395 | −166 | 5.15 | 6.07 |
| `printouts` | 10,065 | 10,111 | +46 | 11.78 | 12.18 |
| **TOTAL** | **118,312** | **116,938** | **−1,374 (−1.2%)** | **9.72** | **13.15** |

Dense tables gain slightly from the extra resolution (+480), but **`Flowchart` collapses** —
2,048 fewer chars, and mermaid edges drop 61 → 55. That is the document class chandra is
*best* at, and the one we most care about. The vision encoder appears to have been trained
in a specific resolution regime and degrades when pushed to the edge of its pixel budget:
more pixels is not more signal. **datalab's `IMAGE_DPI = 192` is a tuned default, not a
lazy one.** The mechanics below are kept only so nobody re-derives them.

#### (mechanics, for reference)

`chandra/settings.py: IMAGE_DPI = 192`, and `chandra/model/util.py: scale_to_fit()` caps
every image at `(3072, 2048)` = 6,291,456 px. Our pages render to 1587x2246 = **3.58 Mpx,
only 57% of that cap** — so the default is *not* saturating the model's own input budget.

`IMAGE_DPI` is a pydantic `BaseSettings` field, so it is **env-overridable with zero code
change**: `IMAGE_DPI=256 ./run.sh chandra --out-dir chandra_dpi256`.

**DPI 256 saturates the cap; anything above it is wasted work.** `scale_to_fit` clamps
them all to the identical size:

| DPI | rendered | what the model sees | vision tokens |
|---|---|---|---|
| **192 (default)** | 1587x2246 | 1596x2240 = 3.58 Mpx | ~3,491 |
| **256** | 2116x2994 | 2100x2968 = 6.23 Mpx | ~6,086 |
| 300 | 2480x3509 | **2100x2968 — identical** | ~6,086 |
| 400 | 3306x4678 | **2100x2968 — identical** | ~6,086 |

So never run above 256: it costs rasterization CPU and delivers byte-identical input.
(Going *past* 6.29 Mpx would mean patching `scale_to_fit`'s `max_size` — a model-input
change, not a config one, and the server's `--mm-processor-kwargs max_pixels` would have to
rise with it.)

### Keep the vLLM server resident (`scripts/serve_chandra.sh`)

Loading chandra costs ~7 min (weights + torch.compile + CUDA-graph capture) and `run.sh`
used to pay that on *every* invocation. `run.sh` now probes :8200 and **attaches** to a
live server instead of starting one; its `trap` only kills a server it started itself.

```bash
scripts/serve_chandra.sh            # start once, stays up, holds ~39 GB
./run.sh chandra --out-dir <name>   # attaches, decodes immediately
scripts/serve_chandra.sh --stop     # free the VRAM before running another model
```

An attached server keeps the flags it was **started** with — `CHANDRA_TUNED` cannot retune a
running server, and `run.sh` says so loudly rather than silently mislabelling a run.

**Orientation is now a permanent default, per the user, and is never an A/B variable.**
Hold it ON in both arms of any future experiment.

**Superseded — the original plan, kept for the reasoning only:**

1. `models/chandra/.venv` **is synced and confirmed working** (`uv sync --project
   models/chandra` — needed a retry with `UV_HTTP_TIMEOUT=180` the first time, since the
   default 30s timeout hit a transient failure downloading the ~large
   `nvidia-cutlass-dsl-libs-cu12` wheel; that is a network flake, unrelated to anything in
   this change). Confirmed `import chandra, paddleocr, pypdf` all succeed together in this
   one venv with no dependency conflict. Current disk: `models/chandra/.venv` 16 GB,
   `harness/.venv` 2.5 GB (standalone test venv, not needed once chandra's venv exists,
   but harmless to leave), **24 GB total** — comfortably under the ~50 GB quota.
2. **No vLLM server has been started and no chandra inference has actually been run
   yet in this session.** A `--smoke --pdfs Flowchart` smoke test was queued
   (`nohup ./run.sh chandra --smoke --pdfs Flowchart &`) but the session was stopped by
   the user before it got past the `uv sync` step (the retry above happened *outside*
   `run.sh`, directly against `models/chandra/.venv`, precisely so this re-sync would not
   have to be repeated). **Next action is to actually run that smoke test now that the
   venv is confirmed ready:**
   ```bash
   nohup ./run.sh chandra --smoke --pdfs Flowchart > /dev/null 2>&1 &
   # wait on the real PID, not the launcher shell — see Harness gotchas above
   tail -f work/chandra_vllm.log   # confirm the server actually accepts the new
                                    # --max-model-len/--mm-processor-kwargs flags and
                                    # doesn't crash on them
   cat outputs/_smoke/chandra/Flowchart.md   # eyeball before trusting it
   ```
3. Then specifically smoke/verify the rotation path — `Complex_table_layouts` is the
   only pdf that exercises it: `./run.sh chandra --smoke --pdfs Complex_table_layouts`
   (32 pages, ~5-6 min at the old ~9.8 s/page rate) and diff pages 3/4/8/13/14/25/26
   against the *baseline* `outputs/chandra/Complex_table_layouts.md` — this is the actual
   question the user asked ("see how it performs"), so don't skip straight to the full
   run without checking these specific pages improved.
4. Full tagged run: `nohup ./run.sh chandra --out-tag oriented > /dev/null 2>&1 &` — do
   **not** omit `--out-tag`, it is what keeps `outputs/chandra/` (the existing
   benchmarked baseline) intact. Verify page counts 32/3/12/14/7 (68 total) in
   `outputs/chandra/oriented/summary.json`.
5. Update the results tables (this file's Status section + README's results table) with
   a before/after comparison once the tagged run completes — that comparison is the
   actual deliverable the user is waiting on, not just "the flags got added."
6. Reclaim when done or moving to another model: `./scripts/reclaim.sh chandra
   datalab-to/chandra-ocr-2` (also fine to `rm -rf harness/.venv` first — nothing else
   needs it once chandra's own venv has `ocr-harness` installed editable).
