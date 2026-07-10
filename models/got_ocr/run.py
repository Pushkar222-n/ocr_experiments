"""GOT-OCR 2.0 via transformers, formatted mode (mathpix-style markdown/LaTeX)."""
import os
from pathlib import Path

from ocr_harness import Adapter, PageResult, cli

MODEL_ID = os.environ.get("MODEL_ID", "stepfun-ai/GOT-OCR-2.0-hf")


class GotOcr(Adapter):
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # This checkpoint's generation_config has eos_token_id=None, and the tokenizer's
        # eos is <|endoftext|> (151643) — not the <|im_end|> (151645) the model actually
        # emits. Passing it explicitly is what makes generate() stop *per sequence*:
        # stop_strings alone exposes no eos_token_id, so HF never pads finished rows and
        # a row that ends early keeps decoding garbage until the slowest row in the batch
        # finishes. Harmless at batch=1, silently corrupts every batched page.
        self.eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    def process_batch(self, image_paths: list[Path]) -> list[PageResult]:
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(imgs, return_tensors="pt", format=True).to("cuda")
        with self.torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                do_sample=False,
                eos_token_id=self.eos_id,
                pad_token_id=self.eos_id,
                max_new_tokens=4096,
            )
        texts = self.processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # native format is mathpix markdown (format=True); saved as the .md directly
        return [PageResult(markdown=t.strip()) for t in texts]


if __name__ == "__main__":
    cli("got_ocr", GotOcr, default_batch=4)
