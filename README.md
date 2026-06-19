# Mental Health Text Classification with QLoRA

Fine-tuning a small open-source LLM (**Phi-3 Mini 3.8B**) with **QLoRA** to
classify short mental-health text into **Normal / Depression / Stress**.

> ⚠️ **Research and education only — not a diagnostic, screening, or crisis tool.**
> Labels are derived from social-media context, not clinical diagnoses.

This project extends my
[Mental Health Prediction (ELSA)](https://github.com/BilalAhmadSami/Mental-Health-Prediction-ELSA)
work from tabular data into **text**, using the HuggingFace PEFT/TRL stack.

## Status
🚧 In progress — data exploration and prep stage.

## Planned pipeline
1. Explore + clean the dataset (`src/explore_data.py`)
2. Filter to 3 classes, stratified train/val/test split, format as prompts
3. QLoRA fine-tune Phi-3 Mini (4-bit base + LoRA adapters)
4. Evaluate vs zero-shot baseline (accuracy, macro-F1, confusion matrix)
5. Push adapter to HuggingFace Hub + optional demo

## Stack
Transformers · PEFT · bitsandbytes · TRL (SFTTrainer) · datasets · scikit-learn
