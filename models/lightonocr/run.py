"""LightOnOCR-2-1B via transformers.

Card-recommended settings: temperature 0.2, top_p 0.9, max_new_tokens 4096,
images resized to 1540px on the longest side. Native output is markdown.
If LightOnOCR-2-1B fails to load, set MODEL_ID=lightonai/LightOnOCR-1B-1025.

The checkpoint's config.json declares model_type "mistral3", so AutoConfig and
AutoModel resolve to the Mistral3 classes. Name the LightOnOcr classes
explicitly (as the model card does) to get the Qwen3 text tower.
"""
import os
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli

MODEL_ID = os.environ.get("MODEL_ID", "lightonai/LightOnOCR-2-1B")
TARGET_LONGEST = 1540


def load_image(path: Path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    scale = TARGET_LONGEST / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


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
                **inputs, do_sample=True, temperature=0.2, top_p=0.9,
                max_new_tokens=4096,
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
    cli("lightonocr", LightOnOcr, default_batch=4)
