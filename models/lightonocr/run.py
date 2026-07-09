"""LightOnOCR-2-1B via vLLM offline engine (pip vllm, no docker, no server).

Card-recommended settings: temperature 0.2, top_p 0.9, max_tokens 4096,
images resized to 1540px on the longest side. Native output is markdown.
If LightOnOCR-2-1B fails to load, set MODEL_ID=lightonai/LightOnOCR-1B-1025.
"""
import base64
import io
import os
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli

MODEL_ID = os.environ.get("MODEL_ID", "lightonai/LightOnOCR-2-1B")
TARGET_LONGEST = 1540


def data_uri(path: Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    scale = TARGET_LONGEST / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LightOnOcr(Adapter):
    def load(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=MODEL_ID,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
        )
        self.sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=4096)

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        msgs = [
            [{"role": "user",
              "content": [{"type": "image_url", "image_url": {"url": data_uri(p)}}]}]
            for p in image_paths
        ]
        outs = self.llm.chat(msgs, self.sampling)
        return [
            PageResult(markdown=o.outputs[0].text.strip(),
                       extra={"output_tokens": len(o.outputs[0].token_ids)})
            for o in outs
        ]


if __name__ == "__main__":
    cli("lightonocr", LightOnOcr, default_batch=32)  # vllm batches internally
