"""dots.mocr (rednote-hilab, formerly dots.ocr) via transformers (trust_remote_code).

Uses the card's prompt_layout_all_en prompt: full layout JSON (bbox + category +
text; tables as HTML, formulas as LaTeX) plus a markdown rendering built by
concatenating each element's text in reading order.
Native output: the raw layout JSON. Markdown: text fields joined in order.

Runs the vision tower's sdpa attention rather than flash-attn — see _stub_flash_attn.
"""
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


def _install_flash_attn_shim():
    """Provide flash_attn.flash_attn_varlen_func backed by torch SDPA.

    modeling_dots_vision.py does a top-level `from flash_attn import
    flash_attn_varlen_func`, so the module cannot be imported at all without it.
    Building flash-attn here isn't worth it, and the checkpoint's own sdpa
    fallback is unusable: it materialises a dense [1, heads, seq, seq] mask, which
    is >4 GiB for a 200-dpi page (~20k patches) and OOMs.

    Attending over each cu_seqlens segment separately needs no mask at all, so
    SDPA picks its memory-efficient kernel. Shapes follow flash-attn's varlen
    convention: q/k/v are (total_tokens, heads, head_dim), and segment i spans
    cu_seqlens[i]:cu_seqlens[i + 1].
    """
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("flash_attn") is not None:
        return  # prefer the real thing when it's installed

    import torch
    import torch.nn.functional as F

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k=None,
                               max_seqlen_q=None, max_seqlen_k=None,
                               dropout_p=0.0, softmax_scale=None, causal=False,
                               **_kwargs):
        out = torch.empty_like(q)
        bounds = cu_seqlens_q.tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            # (len, heads, dim) -> (1, heads, len, dim)
            qi, ki, vi = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            attn = F.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale
            )
            out[start:end] = attn.squeeze(0).transpose(0, 1)
        return out

    mod = types.ModuleType("flash_attn")
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod


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
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        _install_flash_attn_shim()
        self.torch = torch
        # vision tower stays on its config default (flash_attention_2) and calls the
        # shim above; only the text tower needs an explicit sdpa request
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True,
            dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"  # decoder-only batched generate

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        from PIL import Image

        msgs = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": PROMPT_LAYOUT_ALL_EN}]}]
        prompt = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, do_sample=False, max_new_tokens=24000)
        new = gen[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(
            new, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        pad_id = self.processor.tokenizer.pad_token_id
        results = []
        for raw, row in zip(texts, new):
            raw = raw.strip()
            results.append(PageResult(
                markdown=_to_markdown(raw), native=raw, native_ext="json",
                extra={"output_tokens": int((row != pad_id).sum()) if pad_id is not None
                       else int(row.numel())},
            ))
        return results


if __name__ == "__main__":
    cli("dots_ocr", DotsOcr, default_batch=2)
