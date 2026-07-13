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

Balanced tier chosen per provider (NOT their premium option):
- `mistral` → `mistral-ocr-latest`
- `datalab` → hosted Marker, `use_llm=false` (the non-LLM base tier)
- `llamaparse` → `parse_mode=parse_page_with_agent` (the "Balanced" preset, gemini-2.5-flash)
- `landing_ai` → ADE `dpt-2-latest`

Full 68-page results (weighted s/page = whole-PDF wall clock, not GPU decode):

| provider | s/page | raw chars | **visible** | $/1k pages | cost / 68p | $ per 10k visible |
|---|---|---|---|---|---|---|
| `datalab` | 0.96 | 266,456 | **149,941** | **$3.00** | **$0.204** | **$0.014** |
| `mistral` | **0.49** | 123,134 | 106,909 | $4.00 | $0.272 | $0.025 |
| `llamaparse` | 14.31 | 291,707 | 144,455 | $12.50 | $0.850 | $0.059 |
| `landing_ai` | 2.04 | 259,384 | 90,404 | $30.00 | $2.040 | $0.226 |

Actual spend for the run: **$3.37**.

**Datalab Marker wins outright and it is not close.** It extracts the most visible text of
anything in this repo, open or closed (149,941 vs the best open model `mineru` at 128,299),
is the **cheapest** paid API, and is near-fastest. On cost-per-text it is **16x better than
`landing_ai`**. It also recovers flowchart edges.

**`landing_ai` is the trap of the closed set**: metered at exactly 3 credits/page x $0.01 =
**$30/1k pages — 10x Datalab — for 40% LESS text** than Datalab. Nothing recommends it here.
**`mistral` is the speed/cost floor** (0.49 s/page, $4/1k) and is genuinely good on
text/tables/formulas, but it is **not a diagram parser** (flowchart -> 724 chars, zero arrows).
**`llamaparse` (Agentic) recovers the most flowchart edges (58)** and near-Datalab text, but
costs 4x Datalab and is **15-30x slower** (14 s/page; 568 s on the 32-page doc).

### Pricing, verified 2026-07-12 (`uv run python run.py prices`)

Rates were read off each vendor's public pricing page. **An earlier version of this file
understated every one of them** (mistral was quoted at $1/1k when it is $4/1k; llamaparse's
Agentic tier is 10 credits/page, not 3). Costs are recomputed from the usage the APIs already
reported via `run.py reprice`, which never re-calls the network — re-running to refresh a cost
column would mean paying for the same pages twice.

| provider | tier | $/1k pages | 68p | |
|---|---|---|---|---|
| `mistral` | OCR (`mistral-ocr-latest`) | $4.00 | $0.272 | **<- run** |
| `mistral` | Document AI | $5.00 | $0.340 | premium |
| `datalab` | Marker base (`use_llm=false`) | ~$3.00 | $0.204 | **<- run**, *unconfirmed* |
| `datalab` | High Accuracy (`use_llm=true`) | $6.00 | $0.408 | premium |
| `llamaparse` | Fast (`parse_page_without_llm`) | $1.25 | $0.085 | 1 cr/pg |
| `llamaparse` | Cost-effective (`parse_page_with_llm`) | $3.75 | $0.255 | 3 cr/pg |
| `llamaparse` | Agentic (`parse_page_with_agent`) | $12.50 | $0.850 | **<- run**, 10 cr/pg |
| `llamaparse` | Agentic Plus (sonnet) | $56.25 | $3.825 | premium, 45 cr/pg |
| `landing_ai` | ADE `dpt-2-latest` | $30.00 | $2.040 | **<- run**, 3 cr/pg |

Credit rates: LlamaParse **$1.25 / 1000 credits**; Landing AI Explore **$1 = 100 credits**
($0.01/cr; the $250/mo Team plan only gets you to $0.0091/cr = ~$27/1k).

**The one soft number is Datalab's base rate.** Their pricing page is client-rendered, there
is no usage endpoint on the API, and the base `use_llm=false` rate is not published anywhere
fetchable. $6/1k for High Accuracy **is** confirmed (their blog). The $3/1k base is flagged
`unconfirmed: True` in `PRICING` — verify it before quoting, because Datalab winning on cost
depends on it (it still wins on text at any plausible rate).

**LlamaParse's free tier (10k credits/month) absorbed this run**, so its API honestly reported
**0 credits**. The cost shown is the list rate, not a metered charge. Landing AI *did* meter:
204 credits for 68 pages, exactly 3/page.

`scripts/compare.py` folds the closed rows into `outputs/comparison.json` (tagged
`closed:true`, one level deeper under `outputs/closed/*/summary.json`), so the frontend
metrics view reads one file. The frontend groups them under a **paid API** section in the
compare picker (dashed chips, `$` flag), shows per-document cost/credits badges on each
pane, and adds a **Cost** bar card + `cost_usd`/`billed_pages`/`credits` columns in metrics.
Closed outputs are gitignored (under `outputs/`) like the open ones — carry them in the
resume bundle, not the repo.
