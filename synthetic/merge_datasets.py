"""
synthetic/merge_datasets.py

Merge multiple DPO datasets into one training dataset.

Inputs
------
glossary_dpo.jsonl
hallucination_dpo.jsonl

Output
------
merged_dpo.jsonl

Responsibilities
----------------
- Load all DPO datasets.
- Merge them together.
- Shuffle examples.
- Save a single merged dataset.

This module performs NO inference.
"""

from __future__ import annotations

import random

from synthetic.config import (
    GLOSSARY_DPO_PATH,
    HALLUCINATION_DPO_PATH,
    MERGED_DPO_PATH,
)

from synthetic.utils import (
    load_jsonl,
    save_jsonl,
)


def main() -> None:

    glossary_pairs = load_jsonl(
        GLOSSARY_DPO_PATH,
    )

    hallucination_pairs = load_jsonl(
        HALLUCINATION_DPO_PATH,
    )

    merged_pairs = (
        glossary_pairs
        + hallucination_pairs
    )

    random.shuffle(
        merged_pairs,
    )

    save_jsonl(
        merged_pairs,
        MERGED_DPO_PATH,
    )

    print()

    print("=" * 80)
    print("Merged DPO Dataset")
    print("=" * 80)
    print(f"Glossary pairs      : {len(glossary_pairs)}")
    print(f"Hallucination pairs : {len(hallucination_pairs)}")
    print(f"Total pairs         : {len(merged_pairs)}")
    print(f"Saved               : {MERGED_DPO_PATH}")


if __name__ == "__main__":
    main()