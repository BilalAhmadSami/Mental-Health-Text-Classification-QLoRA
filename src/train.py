"""Step 3 — QLoRA fine-tuning of Phi-3 Mini for 3-class classification.

What happens here (the QLoRA recipe):
  1. Load the base model with its weights QUANTIZED to 4-bit (the "Q").
     -> shrinks ~7.6 GB of fp16 weights to ~2 GB, so it fits a 16 GB GPU.
  2. Freeze those 4-bit weights and attach small trainable LoRA adapters
     (the "LoRA") to the linear layers. Only ~0.1-1% of params are trained.
  3. Instruction-tune with TRL's SFTTrainer, computing loss ONLY on the
     assistant's label token(s) (completion-only), so the model learns to
     OUTPUT the class, not to reproduce the post.
  4. Save just the tiny LoRA adapter.

Run on the GPU machine (after `pip install -r requirements.txt`):
    python -m src.train
"""
from __future__ import annotations

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from src.config import (
    PROCESSED_DIR, OUTPUT_DIR, ADAPTER_DIR, BASE_MODEL, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    EPOCHS, LEARNING_RATE, TRAIN_BATCH_SIZE, GRAD_ACCUM, EVAL_BATCH_SIZE,
    SEED, USE_WANDB,
)

# Phi-3 chat turns are opened with these special tokens. We compute the loss
# only on what comes AFTER the assistant marker (i.e. the predicted label).
ASSISTANT_MARKER = "<|assistant|>"


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError("Run `python -m src.data_prep` first.")

    print("Loading processed splits ...")
    ds = load_from_disk(str(PROCESSED_DIR))

    # --- Tokenizer ------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Render each chat into a single training string using Phi-3's template.
    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    # --- 1) 4-bit quantization config (the "Q" in QLoRA) ----------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # 4-bit NormalFloat
        bnb_4bit_compute_dtype=torch.bfloat16,  # math runs in bf16
        bnb_4bit_use_double_quant=True,       # quantize the quant constants too
    )

    print(f"Loading base model {BASE_MODEL} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",          # safe without flash-attn installed
    )
    model = prepare_model_for_kbit_training(model)   # enables grad checkpointing etc.
    model.config.use_cache = False

    # --- 2) LoRA adapters (the "LoRA" in QLoRA) -------------------------
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 3) Completion-only collator (loss only on the label) -----------
    collator = DataCollatorForCompletionOnlyLM(
        response_template=ASSISTANT_MARKER,
        tokenizer=tok,
    )

    # --- 4) Training configuration --------------------------------------
    args = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        max_length=MAX_SEQ_LEN,
        packing=False,                        # required for completion-only loss
        dataset_text_field="text",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",             # memory-friendly optimizer
        seed=SEED,
        report_to=("wandb" if USE_WANDB else "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.model.print_trainable_parameters()   # shows the ~1% trained
    print("\nStarting training ...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f"\nDone. LoRA adapter saved to: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
