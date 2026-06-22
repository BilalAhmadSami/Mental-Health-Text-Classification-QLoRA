"""Step 3 — QLoRA fine-tuning of Phi-3 Mini for 3-class classification.

Robust version: no TRL dependency (its API shifts between releases). We do the
completion-only label masking ourselves and train with transformers.Trainer.

QLoRA recipe:
  1. Load the base model with weights QUANTIZED to 4-bit (NF4) -> fits 16 GB.
  2. Freeze that 4-bit base; attach small trainable LoRA adapters (~1% of params).
  3. Mask the prompt tokens so the loss is computed ONLY on the label
     (completion-only), so the model learns to OUTPUT the class.
  4. Save just the LoRA adapter.

Run on the GPU machine:
    .venv/bin/python -m src.train
"""
from __future__ import annotations

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import (
    PROCESSED_DIR, OUTPUT_DIR, ADAPTER_DIR, BASE_MODEL, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    EPOCHS, LEARNING_RATE, TRAIN_BATCH_SIZE, GRAD_ACCUM, EVAL_BATCH_SIZE,
    SEED, USE_WANDB,
)


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

    # --- Tokenize with completion-only label masking --------------------
    # The full chat = system + user + assistant(label). The prompt = everything
    # up to and including the "<|assistant|>" opener. We mask the prompt tokens
    # with -100 so the loss is computed only on the label tokens.
    def tokenize(ex):
        msgs = ex["messages"]
        full_text = tok.apply_chat_template(msgs, tokenize=False)
        prompt_text = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = tok(full_text, truncation=True, max_length=MAX_SEQ_LEN,
                       add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_text, truncation=True, max_length=MAX_SEQ_LEN,
                         add_special_tokens=False)["input_ids"]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100                      # ignore prompt in the loss
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    tokd = ds.map(tokenize, remove_columns=ds["train"].column_names)

    # --- 1) 4-bit quantization (the "Q") --------------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model {BASE_MODEL} in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    # --- 2) LoRA adapters (the "LoRA") ----------------------------------
    lora = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()            # shows the ~1% trained

    # --- 3) Pad input_ids + labels (labels padded with -100) ------------
    collator = DataCollatorForSeq2Seq(
        tok, model=model, label_pad_token_id=-100, padding=True,
    )

    # --- 4) Training configuration --------------------------------------
    args = TrainingArguments(
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        seed=SEED,
        report_to=("wandb" if USE_WANDB else "none"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokd["train"],
        eval_dataset=tokd["validation"],
        data_collator=collator,
        processing_class=tok,
    )

    print("\nStarting training ...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f"\nDone. LoRA adapter saved to: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
