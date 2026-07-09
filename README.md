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
                     summary.json — the only things git-tracked, rest is gitignored
work/                rasterized page pngs + vllm server logs (regenerable, gitignored)
run.sh               entrypoint: uv sync + (if needed) start a pip-vllm server + run
scripts/compare.py   aggregate every outputs/*/summary.json into one table
```

Each model is its own `uv` project so torch/transformers/vllm/paddle versions never
collide — `uv sync --project models/<x>` only ever touches `models/<x>/.venv`.

## Running a model

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

Re-running the same command resumes: per-page models checkpoint each page
(`outputs/<model>/<stem>/pages/page_NNNN.{md,json}`), per-pdf models
(chandra/mineru/unlimited_ocr multi-mode) checkpoint on the final `.md` existing.

Restrict to specific PDFs: `./run.sh got_ocr --pdfs Handwritten Flowchart`.

## Per-model notes

| model | backend | native format | batches on GPU | notes |
|---|---|---|---|---|
| GOT-OCR-2.0 | transformers | markdown (mathpix `format=True`) | yes (batch=4) | generic; `format=True` prompt |
| LightOnOCR-2-1B | vLLM (in-process, pip) | markdown | yes, vLLM continuous batching | temp=0.2, top_p=0.9, images resized to 1540px longest side |
| dots.mocr (rednote-hilab, formerly dots.ocr) | vLLM (in-process, pip; native since vLLM 0.11.0) | layout json (bbox+category+text; tables as HTML, formulas as LaTeX) | yes, vLLM continuous batching | uses card's `prompt_layout_all_en`; markdown built by joining element text in reading order |
| Unlimited-OCR (Baidu) | transformers, `trust_remote_code` | markdown/text | no (1 img/call); `infer_multi` batches whole pdf in one forward pass | uses card's exact recipe (`base_size=1024, image_size=640, crop_mode=True`) — your earlier bad pages were likely a size/prompt mismatch |
| PaddleOCR-VL-1.6 | paddlepaddle-gpu 3.2.1 | markdown + layout json | pipeline-internal | best-in-class for tables/formulas per OmniDocBench |
| GLM-OCR | glmocr[selfhosted] (vLLM decoder + PP-DocLayout-V3) | markdown + json (bboxes) | pipeline-internal | layout model can run on CPU (`LAYOUT_DEVICE=cpu`) to save VRAM |
| MinerU 2.5 | `mineru-vl-utils[vllm]`, in-process vLLM | markdown + content_list.json | pipeline-internal | checkpoints per-pdf |
| Surya 2 | external pip-vllm server (`SURYA_INFERENCE_URL`) | block json incl. **per-block confidence**, markdown via markdownify | vLLM continuous batching | only model here with a native quality score |
| Chandra OCR 2 | external pip-vllm server (its own docker launcher swapped for a plain `vllm serve`) | markdown + html + metadata json (token counts) | vLLM continuous batching (`--batch-size`) | checkpoints per-pdf |

Surya's and Chandra's docs default to a `docker run vllm`/`chandra_vllm` launcher —
since RunPod pods can't do docker-in-docker, `run.sh` instead does
`vllm serve <model>` directly (pip-installed vllm) and points each tool at it via
env var (`SURYA_INFERENCE_URL`, `VLLM_API_BASE`). If a 24GB pod is too tight to
hold both the vLLM server and anything else, run these two on a dedicated pod.

## Metrics captured

Per page: seconds, char count, GPU mem (nvidia-smi peak), any adapter-reported
extra (Surya: confidence; LightOnOCR: output token count). Per pdf
(`<stem>.metrics.json`): total/avg seconds per page, total chars, peak GPU mem,
and the mean of any numeric per-page extras. `scripts/compare.py` merges every
model's `summary.json` into one table + `outputs/comparison.json` for
downstream (human or LLM) review.


