"""Materialize a vLLM-loadable copy of MinerU2.5-Pro. Prints its path.

vLLM 0.19.1 reads `tie_word_embeddings` off the *top level* of the hf config:
qwen2_vl.py:1251 builds the language model with init_vllm_registered_model(vllm_config=...),
which hands Qwen2ForCausalLM the flat config, and qwen2.py:555 checks the flag there to
decide whether to tie lm_head to the embeddings.

MinerU2.5 ships the transformers-v5 *nested* layout: `tie_word_embeddings: true` lives in
config.json's `text_config` and there is no top-level key. PretrainedConfig defaults the
missing field to False, so vLLM builds a standalone lm_head, demands its weight, and dies:

    ValueError: Following weights were not initialized from checkpoint:
                {'language_model.lm_head.weight'}

The checkpoint has no lm_head to give it -- 681 tensors, `model.embed_tokens.weight`
present, no lm_head at all, because it is tied. Transformers reads the nested value and
ties correctly, which is why the vlm-engine baseline run worked.

So: symlink the snapshot and rewrite only config.json, hoisting the flag to the top level.
This makes vLLM agree with what transformers already did. It does not change the model.

Weights are symlinked, not copied -- the volume has no hardlink support but symlinks work.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "opendatalab/MinerU2.5-Pro-2605-1.2B"
OUT = Path(__file__).resolve().parent.parent / "work" / "mineru2.5_vllm"


def main():
    snap = Path(snapshot_download(REPO))
    OUT.mkdir(parents=True, exist_ok=True)

    for src in snap.iterdir():
        dst = OUT / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if src.name != "config.json":
            dst.symlink_to(src.resolve())

    cfg = json.loads((snap / "config.json").read_text())
    tied = cfg.get("text_config", {}).get("tie_word_embeddings")
    if tied is None:
        sys.exit(f"{REPO}: no text_config.tie_word_embeddings -- checkpoint layout changed, "
                 "re-check against vllm's qwen2.py before forcing a value")
    cfg["tie_word_embeddings"] = tied
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    print(OUT)


if __name__ == "__main__":
    main()
