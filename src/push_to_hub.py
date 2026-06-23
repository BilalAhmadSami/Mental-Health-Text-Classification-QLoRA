"""Step 6 — Publish the LoRA adapter to the HuggingFace Hub.

Authenticate FIRST in your terminal (the token stays on your machine, never in
any script or chat):
    huggingface-cli login        # paste a HF token with WRITE access

Then:
    .venv/bin/python -m src.push_to_hub
"""
from __future__ import annotations

from huggingface_hub import HfApi, create_repo

from src.config import ADAPTER_DIR, HF_ADAPTER_REPO, BASE_MODEL

_CARD = """---
base_model: __BASE__
library_name: peft
pipeline_tag: text-classification
tags:
- lora
- qlora
- peft
- text-classification
- mental-health
license: mit
---

# Phi-3 Mini — Mental Health Text Classification (QLoRA)

A LoRA adapter for `__BASE__`, QLoRA-fine-tuned to classify a short first-person
statement into **Normal / Depression / Stress**.

> ⚠️ **Research and education only — NOT a diagnostic, screening, or crisis tool.**
> Labels are derived from social-media context, not clinical diagnoses, and the
> model must not be used to make decisions about real people.

## Results (balanced held-out test set, 300 per class)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Zero-shot base | 0.510 | 0.490 |
| **Fine-tuned (this adapter)** | **0.931** | **0.931** |

Per-class F1 (fine-tuned): Normal 0.91 · Depression 0.93 · Stress 0.95.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained("__BASE__")
base = AutoModelForCausalLM.from_pretrained("__BASE__", quantization_config=bnb,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, "__REPO__")

msgs = [
    {"role": "system", "content": "You classify a statement into one of: Normal, Depression, Stress. Respond with only the label."},
    {"role": "user", "content": "deadlines piling up and I can't keep up"},
]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=6)
print(tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training

QLoRA (4-bit NF4) · LoRA r=16, alpha=32 · 3 epochs · class-balanced 3-class
subset (~2,137 per class) of the "Sentiment Analysis for Mental Health" dataset.

Full code: https://github.com/BilalAhmadSami/Mental-Health-Text-Classification-QLoRA
"""


def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"No adapter at {ADAPTER_DIR}. Train first.")
    card = _CARD.replace("__BASE__", BASE_MODEL).replace("__REPO__", HF_ADAPTER_REPO)
    (ADAPTER_DIR / "README.md").write_text(card, encoding="utf-8")

    print(f"Creating/updating repo: {HF_ADAPTER_REPO}")
    create_repo(HF_ADAPTER_REPO, repo_type="model", exist_ok=True)
    HfApi().upload_folder(folder_path=str(ADAPTER_DIR), repo_id=HF_ADAPTER_REPO,
                          repo_type="model")
    print(f"Done -> https://huggingface.co/{HF_ADAPTER_REPO}")


if __name__ == "__main__":
    main()
