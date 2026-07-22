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
    TRAIN_DATA_PATH,
    TRANSLATIONS_PATH,
)

from synthetic.utils import (
    load_jsonl,
    save_jsonl,
)


def main() -> None:

    rows = load_jsonl(
        TRANSLATIONS_PATH,
    )

    pairs = []

    skipped = 0

    for row in rows:

        chosen = row["with_glossary"]
        rejected = row["without_glossary"]

        if not row["rewritten"]:
            skipped += 1
            continue

        if not row["glossary_used_after"]:
            skipped += 1
            continue

        pairs.append(
            {
                "prompt": row["english"],
                "chosen": chosen,
                "rejected": rejected,
                "canonical": row["canonical"],
                "expected_gu": row["expected_gu"],
            }
        )

    save_jsonl(
        pairs,
        TRAIN_DATA_PATH,
    )

    print()

    print("=" * 80)
    print("DPO Pair Builder")
    print("=" * 80)
    print(f"Total translations : {len(rows)}")
    print(f"DPO pairs          : {len(pairs)}")
    print(f"Skipped            : {skipped}")
    print(f"Saved              : {TRAIN_DATA_PATH}")


if __name__ == "__main__":
    main()