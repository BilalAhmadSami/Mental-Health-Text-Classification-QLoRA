# Mental Health Text Classification with QLoRA

Fine-tuning **Phi-3 Mini (3.8B)** with **QLoRA** to classify a short first-person
statement into **Normal / Depression / Stress** - trained on a single 16 GB GPU.

**🤗 Model (LoRA adapter):** https://huggingface.co/bs01338/phi3-mini-mental-health-qlora

> ⚠️ **Research and education only - NOT a diagnostic, screening, or crisis tool.**
> Labels come from social-media context, not clinical diagnoses, and the model
> must never be used to make decisions about real people.

This project extends my
[Mental Health Prediction (ELSA)](https://github.com/BilalAhmadSami/Mental-Health-Prediction-ELSA)
work from **tabular** data into **text**, using the HuggingFace PEFT / Transformers
fine-tuning stack.

---

## Results

Evaluated on a **balanced held-out test set** (300 examples per class, 900 total),
comparing the un-tuned base model (zero-shot) against the QLoRA fine-tuned adapter.

| Model | Accuracy | Macro-F1 |
|-------|:--------:|:--------:|
| Zero-shot base (Phi-3 Mini) | 0.510 | 0.490 |
| **Fine-tuned (QLoRA)** | **0.931** | **0.931** |

**Per-class F1 (fine-tuned):** Normal 0.91 · Depression 0.93 · Stress 0.95

Fine-tuning lifted macro-F1 by **+0.44**. The largest gain was on the minority
class **Stress** (F1 0.34 → 0.95), which validates training on a class-balanced
subset. Confusion-matrix plots are written to `outputs/` by the evaluation script.

---

## How QLoRA works (in brief)

Full fine-tuning of a 3.8B model needs ~30 GB+ of GPU memory. **QLoRA** fits it in
~8 GB by combining two ideas:

- **Q - 4-bit quantization (NF4):** the base model's weights are stored in 4 bits
  (~4× smaller) and dequantized to bf16 only for computation. The base stays frozen.
- **LoRA - Low-Rank Adaptation:** instead of updating all weights, small trainable
  adapter matrices (~1% of parameters) are attached to the linear layers. Only
  these are trained.

The output is a few-MB **adapter** that is layered onto the base model at inference.

---

## Stack

| Component | Choice |
|-----------|--------|
| Base model | `microsoft/Phi-3-mini-4k-instruct` (3.8B, MIT) |
| Fine-tuning | QLoRA - `peft` (LoRA r=16, α=32) + `bitsandbytes` (4-bit NF4) |
| Trainer | HuggingFace `transformers.Trainer` (completion-only loss) |
| Data | `btwitssayan/sentiment-analysis-for-mental-health` (3 clean classes) |
| Metrics | `scikit-learn` (accuracy, macro-F1, confusion matrix) |
| Hardware | single NVIDIA 16 GB GPU |

---

## Project structure

```
Mental-Health-Text-Classification-QLoRA/
├── requirements.txt
└── src/
    ├── config.py          # all settings in one place
    ├── explore_data.py    # inspect class distribution
    ├── data_prep.py       # filter to 3 classes, balanced splits, chat formatting
    ├── train.py           # QLoRA: 4-bit base + LoRA adapters + Trainer
    ├── evaluate_model.py   # fine-tuned vs zero-shot (accuracy, macro-F1, confusion matrix)
    ├── inference.py       # classify new text with the adapter
    └── push_to_hub.py     # publish the adapter to the HuggingFace Hub
```

---

## Setup and run

```bash
# 1. Environment (GPU machine). Install PyTorch matched to your CUDA FIRST:
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Prepare data (balanced 3-class splits)
python -m src.data_prep

# 3. Fine-tune (saves the LoRA adapter to outputs/phi3-mh-lora/)
python -m src.train

# 4. Evaluate: fine-tuned vs zero-shot baseline
python -m src.evaluate_model

# 5. Classify new text
python -m src.inference "deadlines piling up and I can't keep up"
```

---

## Method notes

- **Classes:** kept the three clean, well-represented classes (Normal, Depression,
  Stress). Dropped *Anxiety* (machine-translated / noisy text in this corpus),
  *Suicidal* (too ethically sensitive for a demo), and the sparse Bipolar /
  Personality-disorder classes.
- **Balancing:** the majority classes were downsampled so training is balanced
  (~2,137 per class); test and validation are balanced too, for honest metrics.
- **Completion-only loss:** the loss is computed only on the label tokens, so the
  model learns to *output the class* rather than reproduce the input.

---

## Limitations & future work

- Labels are derived from subreddit context, not clinical judgement - a real
  ceiling on achievable quality. The 93% reflects *this dataset's* distribution.
- "Normal", "Stress" and "Depression" genuinely overlap in casual text.
- **Future work:** cross-dataset evaluation (test on a *different* social-media
  source to measure real generalization); a confidence/abstention option; a larger
  base model; and a Gradio demo Space.
