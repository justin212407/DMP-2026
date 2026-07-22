"""
synthetic/split_dataset.py

Split the merged DPO dataset into training and validation sets.

Input
-----
merged_dpo.jsonl

Outputs
-------
train_dpo.jsonl
val_dpo.jsonl

Responsibilities
----------------
- Load merged DPO dataset.
- Shuffle deterministically.
- Split into train/validation.
- Save both datasets.
"""

from __future__ import annotations

import random

from synthetic.config import (
    MERGED_DPO_PATH,
    TRAIN_DPO_PATH,
    VAL_DPO_PATH,
)

from synthetic.utils import (
    load_jsonl,
    save_jsonl,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RANDOM_SEED = 42

TRAIN_RATIO = 0.9


def main() -> None:

    rows = load_jsonl(
        MERGED_DPO_PATH,
    )

    random.seed(
        RANDOM_SEED,
    )

    random.shuffle(
        rows,
    )

    split_index = int(
        len(rows) * TRAIN_RATIO
    )

    train_rows = rows[:split_index]

    val_rows = rows[split_index:]

    save_jsonl(
        train_rows,
        TRAIN_DPO_PATH,
    )

    save_jsonl(
        val_rows,
        VAL_DPO_PATH,
    )

    print()

    print("=" * 80)
    print("DPO Dataset Split")
    print("=" * 80)
    print(f"Total pairs      : {len(rows)}")
    print(f"Train pairs      : {len(train_rows)}")
    print(f"Validation pairs : {len(val_rows)}")
    print(f"Saved train      : {TRAIN_DPO_PATH}")
    print(f"Saved validation : {VAL_DPO_PATH}")


if __name__ == "__main__":
    main()