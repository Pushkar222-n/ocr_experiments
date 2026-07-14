# Chandra OCR — production inference

OCR a folder (or a zip) of PDFs with [Chandra](https://github.com/datalab-to/chandra) on a
resident vLLM server. Point it at an input folder, get a mirrored output folder back.

Self-contained: this directory is the whole thing. Copy it to a pod, run two commands.

```bash
./serve.sh                 # start the model server once (~7 min cold). Leave it up.
./run.sh /path/to/docs     # OCR everything under it. Re-runnable, resumable.
./serve.sh --stop          # give the ~39 GB of VRAM back when you're done
```

## Quickstart on a fresh pod

```bash
git clone <this repo> && cd inference
rsync -avz ~/local/docs/  root@<pod>:/workspace/inference/input/   # get your pdfs there

./run.sh input/                 # builds the venv, starts the server, runs
                                # (or: ./run.sh input.zip)

# results are in outputs/input/ — pull them back
rsync -avz root@<pod>:/workspace/inference/outputs/  ~/local/results/
```

`run.sh` builds the venv and starts the server if they are not already up, so the first
command is genuinely the only one you need. It leaves the server **running** afterwards —
loading it costs minutes, and the next job should not pay that twice.

## Input

Anything you point `--input` at:

- **a folder** — searched **recursively**, so nested sub-folders are fine
- **a `.zip`** — extracted once into `work/extracted/` and reused (zip-slip is rejected)
- a single `.pdf`

Chandra takes a whole PDF at a time and batches its pages internally, so **checkpointing is
per-document, not per-page**. A killed run redoes at most the one document it was in.

## Output — mirrors the input tree

For input `mydocs/` containing `top.pdf`, `BatchA/a.pdf`, `BatchA/sub/deep.pdf`:

```
outputs/mydocs/
├── summary.json                 whole-run metrics + any failures
├── summary.csv                  the same, one row per document
├── top/
│   ├── top.md                   markdown
│   ├── top.html                 chandra's native html
│   ├── top.metadata.json        per-page token counts + page boxes
│   ├── top.metrics.json         our metrics  ← written last, this is the done-marker
│   └── *.webp                   extracted images
├── BatchA/a/a.md ...
└── BatchA/sub/deep/deep.md ...
```

The per-document folder is not decoration: chandra's markdown links its images by bare
filename, so they have to sit next to the `.md` or every image link breaks.

Run name defaults to the input folder's name (`outputs/<input_folder_name>/`); override with
`--name`.

## Options

```
./run.sh <input> [options]

  --name NAME         output run name          (default: input folder/zip name)
  --batch-size N      pages in flight          (default 16 — see "Do not tune these")
  --limit N           process only N documents (smoke test)
  --redo              ignore checkpoints, reprocess everything
  --dry-run           list what would be processed, then exit
  --images            also OCR loose png/jpg files, not just pdfs
  --no-orient         skip rotation correction (don't — see below)
  -v                  debug logging
```

Server control:

```
./serve.sh            start (idempotent — a live server is left alone)
./serve.sh --status   up or down?
./serve.sh --logs     follow the vLLM log
./serve.sh --stop     release the VRAM
```

## Logs

- `logs/<run>_<timestamp>.log` — one file per run, DEBUG level, every document and failure
- `logs/vllm_server.log` — the server's own log
- Console gets a documents progress bar with a nested pages bar and live s/page. Log lines
  are printed through `tqdm.write`, so they never tear a bar in half.

A document that fails is logged with a traceback, recorded in `summary.json` under
`failures`, and **the run continues**. Exit code is non-zero if anything failed.

## Do not tune these (they are measured, not guessed)

| setting | value | why |
|---|---|---|
| `--batch-size` | **16** | At 48 it is **39% slower**. A page is ~6,100 vision tokens of prefill; 32 concurrent pages is ~196k against a ~229k-token KV cache, so the scheduler preempts and thrashes. vLLM's "max concurrency 45x" is computed against `--max-model-len`, not against what a page actually costs, and badly overstates the usable batch. |
| `IMAGE_DPI` | **192** | Chandra's default, and tuned. 256 saturates the model's pixel cap (74% more vision tokens) and is **worse**: −1.2% visible text and 35% slower, and diagram-heavy pages lose the most. Above 256 the client clamps to a byte-identical image, so it costs CPU for nothing. |
| `--max-model-len` | **18000** | **Required**, not an optimization. vLLM sizes the KV cache to ~229k tokens while the model config declares a 262,144-token context, and **refuses to start** when the context exceeds the cache. |
| rotation | **on** | Worth **+4.8%** more text on rotated pages, against −0.7% on unrotated pages of the same document (control). |

Deliberately **not** set: `--enable-prefix-caching` (chandra puts the per-page-unique image
*before* the constant prompt, so no two pages ever share a prefix — it can never hit), and
`--max-num-seqs` / `--max-num-batched-tokens` (the vendor's values are H100-scaled and
*lower* than vLLM's defaults; they would split one page's ~6.3k-token prefill across two
scheduler steps).

Everything above is overridable by env var if you really mean it (`CHANDRA_BATCH_SIZE`,
`IMAGE_DPI`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`), and `ocr/config.py` documents each one.

## How it is put together

Two processes, and the split is the whole design:

1. **The vLLM server** (`serve.sh`) holds the weights (8.61 GiB) and owns the GPU. Slow to
   start, hot forever.
2. **The client** (`run.sh` → `python -m ocr`) rasterizes pages, corrects rotation, and POSTs
   to that server. It is **pure CPU** — chandra's `vllm` method never loads weights locally,
   and the orientation classifier runs on onnxruntime. `run.sh` exports
   `CUDA_VISIBLE_DEVICES=""` for it, which makes "the classifier must not touch the GPU"
   structural rather than a rule someone has to remember.

```
ocr/config.py     every tuned constant, with the reason
ocr/discover.py   recursive + zip input discovery (chandra's own CLI globs one level deep)
ocr/orient.py     CPU rotation correction (PP-LCNet doc-orientation, onnxruntime)
ocr/engine.py     the chandra client and page batching
ocr/report.py     output writing, metrics, checkpointing
ocr/__main__.py   orchestration, progress, logging
```

## Hardware

Sized on a 46 GB A40 at ~9.7 s/page (~370 pages/hour).

- **Weights are only 8.61 GiB** — the ~39 GB you see on the card is KV-cache *reservation*
  (`--gpu-memory-utilization 0.85`), not demand.
- **24 GB cards work but are a downgrade**, not a saving: 8.61 GiB of weights leaves ~13 GB of
  KV instead of ~31 GB, which forces the batch well below 16 — and the batch is what the
  throughput is made of. Lower `MAX_MODEL_LEN` (~8000–10000) or the server will not start.
- **48 GB is the sweet spot.** The A40 is already saturated at batch 16, so a faster card of
  the same VRAM class (L40S) converts to real throughput rather than being wasted.

## Known noise

vLLM logs a large `Failed to load plugin register_paddlex_genai_models` traceback at every
startup. `paddlex` (pulled in by the orientation classifier) registers a vLLM entry-point
plugin that fails to import; vLLM logs it at ERROR, then **serves normally**. It is cosmetic.
Do not chase it.

## Sanity check after a run

```bash
cat outputs/<run>/summary.json     # documents, pages, s/page, visible chars, failures
```

Verify by page count *and* by eyeballing a markdown file — never by exit code alone.
