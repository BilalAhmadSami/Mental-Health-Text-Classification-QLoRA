"""Step 5 — Inference: classify new text with the fine-tuned LoRA adapter.

Loads the 4-bit base model + your trained adapter once, then classifies any
text into Normal / Depression / Stress.

Usage:
    python -m src.inference                      # runs a few samples
    python -m src.inference "your text here"     # classify your own

⚠️ Research/education only — NOT a diagnostic or screening tool.
"""
from __future__ import annotations

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.config import BASE_MODEL, ADAPTER_DIR, LABELS, SYSTEM_INSTRUCTION

_tok = None
_model = None


def load():
    """Load base (4-bit) + adapter once; cached for repeated calls."""
    global _tok, _model
    if _model is not None:
        return _tok, _model
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"No adapter at {ADAPTER_DIR}. Train first.")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    _tok, _model = tok, model
    return tok, model


def classify(text: str) -> str:
    """Return the predicted label for one piece of text."""
    tok, model = load()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_INSTRUCTION},
         {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=6, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    for lab in LABELS:
        if lab.lower() in gen.lower():
            return lab
    return gen or "(no label)"


def main():
    samples = sys.argv[1:] or [
        "I haven't been able to get out of bed for days and nothing feels worth it.",
        "So much on my plate this week, deadlines piling up and I can't keep up.",
        "Had a great walk and coffee with a friend this morning.",
    ]
    for t in samples:
        print(f"\n[{classify(t)}]  {t}")
    print("\n(Research/education only — not a diagnostic tool.)")


if __name__ == "__main__":
    main()
