"""Step 1 — Explore the dataset.

Loads the dataset, shows the full class distribution, then the distribution
after keeping only our clean 3 classes (Normal / Depression / Stress), plus
basic text-length stats and sample rows.

Usage (downloads the dataset on first run):
    pip install datasets pandas
    python -m src.explore_data
"""
from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from src.config import DATASET_REPO, TEXT_COLUMN, LABEL_COLUMN, KEEP_LABELS


def show_distribution(title: str, labels) -> None:
    counts = Counter(labels)
    total = len(labels)
    print(f"\n=== {title}: {total} rows ===")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {str(label):22s} {n:6d}  ({100*n/total:5.1f}%)")


def text_stats(texts) -> None:
    lengths = sorted(len(t or "") for t in texts)
    n = len(lengths)
    avg = sum(lengths) / n
    print(f"\nText length (chars): avg={avg:.0f}  median={lengths[n//2]}  "
          f"p95={lengths[int(0.95*n)]}  max={lengths[-1]}")


def main() -> None:
    print(f"Loading {DATASET_REPO} ...")
    ds = load_dataset(DATASET_REPO)        # DatasetDict with a single 'train' split
    train = ds["train"]
    print(f"Columns: {train.column_names}")

    show_distribution("All classes", train[LABEL_COLUMN])

    # Drop rows with empty text, then keep our 3 clean classes
    kept = train.filter(
        lambda r: r[LABEL_COLUMN] in KEEP_LABELS
        and isinstance(r[TEXT_COLUMN], str)
        and r[TEXT_COLUMN].strip() != ""
    )
    print("\n" + "=" * 55)
    print(f"After keeping {KEEP_LABELS} and dropping empty text:")
    show_distribution("Kept (3 classes)", kept[LABEL_COLUMN])
    text_stats(kept[TEXT_COLUMN])

    print("\n=== Sample rows ===")
    seen = set()
    for row in kept:
        lab = row[LABEL_COLUMN]
        if lab not in seen:
            seen.add(lab)
            snippet = row[TEXT_COLUMN][:140].replace("\n", " ")
            print(f"  [{lab}] {snippet}")
        if len(seen) == len(KEEP_LABELS):
            break


if __name__ == "__main__":
    main()
