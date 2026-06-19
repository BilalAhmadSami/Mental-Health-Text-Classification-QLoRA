"""Step 2 — Data preparation.

Pipeline:
  1. Load the dataset, keep the 3 clean classes, drop empty text.
  2. Build BALANCED held-out validation/test sets (equal per class) for fair eval.
  3. Build a BALANCED training set (downsample the majority classes to the
     minority's size) so the model treats all classes fairly.
  4. Attach chat-style messages for instruction fine-tuning.
  5. Save the splits to data/processed/ for the training script.

Run:
    python -m src.data_prep
"""
from __future__ import annotations

import random
from collections import Counter

from datasets import load_dataset, DatasetDict

from src.config import (
    DATASET_REPO, TEXT_COLUMN, LABEL_COLUMN, KEEP_LABELS,
    SEED, TEST_PER_CLASS, VAL_PER_CLASS, PROCESSED_DIR, SYSTEM_INSTRUCTION,
)


def load_clean():
    """Load the dataset and keep only non-empty rows in our 3 classes."""
    ds = load_dataset(DATASET_REPO)["train"]
    ds = ds.filter(
        lambda r: r[LABEL_COLUMN] in KEEP_LABELS
        and isinstance(r[TEXT_COLUMN], str)
        and r[TEXT_COLUMN].strip() != ""
    )
    # Standardise column names to text / label and drop the rest
    ds = ds.rename_columns({TEXT_COLUMN: "text", LABEL_COLUMN: "label"})
    drop = [c for c in ds.column_names if c not in ("text", "label")]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def make_splits(ds) -> DatasetDict:
    """Balanced test/val (equal per class) + a class-balanced train set."""
    # Group row indices by class, then shuffle each group deterministically.
    by_label: dict[str, list[int]] = {lab: [] for lab in KEEP_LABELS}
    for i, lab in enumerate(ds["label"]):
        by_label[lab].append(i)
    rng = random.Random(SEED)
    for lab in by_label:
        rng.shuffle(by_label[lab])

    test_idx, val_idx, train_pools = [], [], {}
    for lab in KEEP_LABELS:
        idx = by_label[lab]
        test_idx += idx[:TEST_PER_CLASS]
        val_idx += idx[TEST_PER_CLASS:TEST_PER_CLASS + VAL_PER_CLASS]
        train_pools[lab] = idx[TEST_PER_CLASS + VAL_PER_CLASS:]

    # Balance the training set: downsample every class to the smallest pool.
    min_train = min(len(p) for p in train_pools.values())
    train_idx = []
    for lab in KEEP_LABELS:
        train_idx += train_pools[lab][:min_train]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print(f"Balanced train: {min_train} per class")
    return DatasetDict(
        train=ds.select(train_idx),
        validation=ds.select(val_idx),
        test=ds.select(test_idx),
    )


def to_chat(example: dict) -> dict:
    """Instruction-tuning format: system + user post -> assistant label."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["label"]},
        ]
    }


def main() -> None:
    print("Loading + cleaning ...")
    ds = load_clean()
    splits = make_splits(ds)

    print("\n=== Split sizes ===")
    for name in ("train", "validation", "test"):
        print(f"  {name:11s} {len(splits[name]):5d}  {dict(Counter(splits[name]['label']))}")

    splits = splits.map(to_chat)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(PROCESSED_DIR))
    print(f"\nSaved processed splits to: {PROCESSED_DIR}")

    print("\n=== Sample training example (chat messages) ===")
    for msg in splits["train"][0]["messages"]:
        print(f"  [{msg['role']}] {msg['content'][:160]}")


if __name__ == "__main__":
    main()
