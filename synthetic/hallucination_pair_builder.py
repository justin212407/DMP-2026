"""
synthetic/pair_builder.py

Build DPO preference pairs from translated synthetic data.

Input
-----
translations.jsonl

Output
------
glossary_dpo.jsonl

Responsibilities
----------------
- Convert translated examples into DPO preference pairs.
- Skip examples where no rewrite was required.
- Save the resulting dataset.

This module performs NO inference.
"""

from __future__ import annotations

from synthetic.config import (
    HALLUCINATION_TRANSLATIONS_PATH,
    HALLUCINATION_DPO_PATH,
)

from synthetic.utils import (
    load_jsonl,
    save_jsonl,
)


def main() -> None:

    rows = load_jsonl(
        HALLUCINATION_TRANSLATIONS_PATH,
    )

    pairs = []

    skipped = 0

    for row in rows:

        chosen = row["chosen"]
        rejected = row["rejected"]

        pairs.append(
            {
                "prompt": row["english"],
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    save_jsonl(
        pairs,
        HALLUCINATION_DPO_PATH,
    )

    print()

    print("=" * 80)
    print("DPO Pair Builder")
    print("=" * 80)
    print(f"Total translations : {len(rows)}")
    print(f"DPO pairs          : {len(pairs)}")
    print(f"Skipped            : {skipped}")
    print(f"Saved              : {HALLUCINATION_DPO_PATH}")


if __name__ == "__main__":
    main()