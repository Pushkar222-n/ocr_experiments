"""dots.mocr (rednote-hilab, formerly dots.ocr) via vLLM in-process (pip, no docker).

Natively supported by vLLM since 0.11.0 — no model-registration workaround
needed. Uses the card's prompt_layout_all_en prompt: full layout JSON (bbox +
category + text; tables as HTML, formulas as LaTeX) plus a markdown rendering
built by concatenating each element's text in reading order.
Native output: the raw layout JSON. Markdown: text fields joined in order.
"""
import base64
import io
import json
import os
import re
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli

MODEL_ID = os.environ.get("MODEL_ID", "rednote-hilab/dots.mocr")

PROMPT_LAYOUT_ALL_EN = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox.\n"
    "1. Bbox format: [x1, y1, x2, y2]\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
    "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
    "'Section-header', 'Table', 'Text', 'Title'].\n"
    "3. Text Extraction & Formatting Rules:\n"
    "- Picture: For the 'Picture' category, the text field should be omitted.\n"
    "- Formula: Format its text as LaTeX.\n"
    "- Table: Format its text as HTML.\n"
    "- All Others (Text, Title, etc.): Format their text as Markdown.\n"
    "4. Constraints:\n"
    "- The output text must be the original text from the image, with no translation.\n"
    "- All layout elements must be sorted according to human reading order.\n"
    "5. Final Output: The entire output must be a single JSON object."
)


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _to_markdown(raw_json: str) -> str:
    try:
        elements = json.loads(raw_json)
    except json.JSONDecodeError:
        # model sometimes wraps json in a ```json fence despite the prompt
        m = re.search(r"\[.*\]", raw_json, re.S)
        elements = json.loads(m.group(0)) if m else []
    parts = []
    for el in elements:
        if el.get("category") == "Picture":
            continue
        text = el.get("text", "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


class DotsOcr(Adapter):
    def load(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.9")),
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=24000)

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        msgs = [
            [{"role": "user",
              "content": [{"type": "image_url", "image_url": {"url": data_uri(p)}},
                          {"type": "text", "text": PROMPT_LAYOUT_ALL_EN}]}]
            for p in image_paths
        ]
        outs = self.llm.chat(msgs, self.sampling)
        results = []
        for o in outs:
            raw = o.outputs[0].text.strip()
            md = _to_markdown(raw)
            results.append(PageResult(markdown=md, native=raw, native_ext="json",
                                       extra={"output_tokens": len(o.outputs[0].token_ids)}))
        return results


if __name__ == "__main__":
    cli("dots_ocr", DotsOcr, default_batch=16)
