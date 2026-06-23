"""Step 4 — Evaluation: fine-tuned LoRA adapter vs zero-shot baseline.

Runs BOTH models on the balanced 900-example test set and reports
accuracy, macro-F1, a per-class report, and a confusion matrix (saved as PNG).
The headline result is the IMPROVEMENT of fine-tuned over the un-tuned base.

Run (after training):
    .venv/bin/python -m src.evaluate_model
"""
from __future__ import annotations

import json
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from src.config import (
    PROCESSED_DIR, OUTPUT_DIR, ADAPTER_DIR, BASE_MODEL, LABELS,
    SYSTEM_INSTRUCTION, EVAL_BATCH_SIZE, MAX_SEQ_LEN,
)


def load_base_and_tokenizer():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # left-pad for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


def parse_label(text: str) -> str | None:
    """Map a generated string to one of our labels, or None if unmatched."""
    t = text.strip().lower()
    for lab in LABELS:
        if lab.lower() in t:
            return lab
    return None


def predict(tok, model, texts, batch_size=8):
    preds, unparsed = [], 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_INSTRUCTION},
                 {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=True,
            )
            for t in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_SEQ_LEN, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for d in tok.batch_decode(gen, skip_special_tokens=True):
            lab = parse_label(d)
            if lab is None:
                unparsed += 1
                lab = LABELS[0]            # fallback so the row still scores
            preds.append(lab)
        print(f"  predicted {len(preds)}/{len(texts)}", end="\r")
    print()
    if unparsed:
        print(f"  ({unparsed} outputs didn't contain a clear label; counted as '{LABELS[0]}')")
    return preds


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    print(f"\n=== {name} ===   accuracy={acc:.3f}   macro-F1={mf1:.3f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {"name": name, "accuracy": acc, "macro_f1": mf1,
            "confusion_matrix": cm.tolist()}, cm


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  saved {path}")


def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"No adapter at {ADAPTER_DIR}. Train first.")
    test = load_from_disk(str(PROCESSED_DIR))["test"]
    texts, y_true = test["text"], test["label"]
    print(f"Test set: {len(texts)} examples  {dict(Counter(y_true))}")

    tok, base = load_base_and_tokenizer()

    print("\n[1/2] Zero-shot baseline (un-tuned Phi-3) ...")
    base_metrics, base_cm = report("Zero-shot baseline", y_true,
                                   predict(tok, base, texts, EVAL_BATCH_SIZE))

    print("\n[2/2] Fine-tuned (base + LoRA adapter) ...")
    ft = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    ft.eval()
    ft_metrics, ft_cm = report("Fine-tuned (QLoRA)", y_true,
                               predict(tok, ft, texts, EVAL_BATCH_SIZE))

    # Save metrics + confusion matrices
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps({"baseline": base_metrics, "finetuned": ft_metrics}, indent=2))
    plot_cm(np.array(base_cm), "Zero-shot baseline", OUTPUT_DIR / "cm_baseline.png")
    plot_cm(np.array(ft_cm), "Fine-tuned (QLoRA)", OUTPUT_DIR / "cm_finetuned.png")

    print("\n================  SUMMARY  ================")
    print(f"  baseline  : acc {base_metrics['accuracy']:.3f}  macro-F1 {base_metrics['macro_f1']:.3f}")
    print(f"  fine-tuned: acc {ft_metrics['accuracy']:.3f}  macro-F1 {ft_metrics['macro_f1']:.3f}")
    print(f"  macro-F1 gain: {ft_metrics['macro_f1'] - base_metrics['macro_f1']:+.3f}")
    print("  metrics.json + confusion-matrix PNGs saved in outputs/")


if __name__ == "__main__":
    main()
