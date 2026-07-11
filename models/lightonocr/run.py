"""LightOnOCR-2-1B, in-process on transformers or against a served vLLM.

Card-recommended settings: temperature 0.2, top_p 0.9, max_new_tokens 4096,
images resized to 1540px on the longest side. Native output is markdown.
If LightOnOCR-2-1B fails to load, set MODEL_ID=lightonai/LightOnOCR-1B-1025.

The checkpoint's config.json declares model_type "mistral3", so AutoConfig and
AutoModel resolve to the Mistral3 classes. Name the LightOnOcr classes
explicitly (as the model card does) to get the Qwen3 text tower.

Set LIGHTON_URL (or --url) to talk to an openai-compatible vLLM instead. Both paths
use the *same* sampling (temp 0.2 / top_p 0.9 / 4096 tokens) and the same 1540px
resize, so the engine is the only variable — the point of the A/B. The transformers
path static-batches and pads, so every batch costs as much as its longest page;
vLLM continuously batches, which is why the http path fans the batch out across
threads rather than sending it as one request.
"""
import base64
import io
import json as jsonlib
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli

MODEL_ID = os.environ.get("MODEL_ID", "lightonai/LightOnOCR-2-1B")
TARGET_LONGEST = 1540
# card-recommended, identical on both engines
TEMPERATURE, TOP_P, MAX_TOKENS = 0.2, 0.9, 4096


def load_image(path: Path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    scale = TARGET_LONGEST / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


class LightOnOcrHTTP(Adapter):
    """LightOnOCR served by vLLM, addressed over the openai-compatible endpoint."""

    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/chat/completions"

    def load(self):
        pass  # the weights live in the server; loading them here would waste VRAM

    def _one(self, path: Path) -> PageResult:
        # PNG, not JPEG: lossless, so the http path feeds the model the same pixels the
        # transformers path does. A jpeg artifact would be indistinguishable from an
        # engine difference in the diff, which would defeat the A/B.
        buf = io.BytesIO()
        load_image(path).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        body = jsonlib.dumps({
            "model": os.environ.get("LIGHTON_MODEL", "lightonocr"),
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS,
        }).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            out = jsonlib.loads(resp.read())
        text = out["choices"][0]["message"]["content"] or ""
        return PageResult(
            markdown=text.strip(),
            extra={"output_tokens": out.get("usage", {}).get("completion_tokens", 0)},
        )

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        # fan out: vLLM's win is continuous batching, and it only shows up under
        # concurrency. Sending these one at a time would measure the server's latency,
        # not its throughput.
        with ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
            return list(pool.map(self._one, image_paths))


class LightOnOcr(Adapter):
    def load(self):
        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self.torch = torch
        self.processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
        self.processor.tokenizer.padding_side = "left"  # decoder-only batched generate
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}],
            add_generation_prompt=True, tokenize=False,
        )

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        imgs = [load_image(p) for p in image_paths]
        inputs = self.processor(
            text=[self.prompt] * len(imgs), images=imgs,
            return_tensors="pt", padding=True,
        )
        inputs = {
            k: v.to(self.model.device, self.model.dtype) if v.is_floating_point()
            else v.to(self.model.device)
            for k, v in inputs.items()
        }
        with self.torch.inference_mode():
            gen = self.model.generate(
                **inputs, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                max_new_tokens=MAX_TOKENS,
            )
        new = gen[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(new, skip_special_tokens=True)
        pad_id = self.processor.tokenizer.pad_token_id
        return [
            PageResult(
                markdown=t.strip(),
                extra={"output_tokens": int((row != pad_id).sum()) if pad_id is not None
                       else int(row.numel())},
            )
            for t, row in zip(texts, new)
        ]


if __name__ == "__main__":
    # LIGHTON_URL is the base openai endpoint *including* /v1 (run.sh sets it), matching
    # the form surya's SURYA_INFERENCE_URL uses. Batch is bigger on the served path: it is
    # a concurrency level for vLLM to schedule against, not a padded static batch.
    url = os.environ.get("LIGHTON_URL")
    if url:
        cli("lightonocr", lambda: LightOnOcrHTTP(url), default_batch=8)
    else:
        cli("lightonocr", LightOnOcr, default_batch=4)
