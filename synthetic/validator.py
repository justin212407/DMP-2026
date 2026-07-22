"""
synthetic/validator.py

Validate the synthetic glossary dataset.

Input
-----
translations.jsonl

Responsibilities
----------------
- Measure glossary adherence before rewriting.
- Measure glossary adherence after rewriting.
- Report dataset statistics.

This module performs NO inference and does NOT modify the dataset.
"""

from __future__ import annotations

from synthetic.config import TRANSLATIONS_PATH
from synthetic.glossary_enforcer import glossary_used
from synthetic.utils import load_jsonl
from synthetic.utils import save_jsonl


def main() -> None:

    rows = load_jsonl(TRANSLATIONS_PATH)

    total = len(rows)

    already_correct = 0
    rewritten = 0
    fixed = 0
    unchanged = 0

    negative_examples = []

    for row in rows:

        before = glossary_used(
            row["without_glossary"],
            row["expected_gu"],
        )

        after = glossary_used(
            row["with_glossary"],
            row["expected_gu"],
        )

        if (
            row["without_glossary"] != row["with_glossary"]
            and not after
        ):
            print("\nRewrite failed:")
            print(row["english"])
            print(row["without_glossary"])
            print(row["with_glossary"])

        if (not before) and (row["without_glossary"] != row["with_glossary"]):

            negative_examples.append(
                {
                    "canonical": row["canonical"],
                    "english": row["english"],
                    "without_glossary": row["without_glossary"],
                    "with_glossary": row["with_glossary"],
                    "expected_gu": row["expected_gu"],
                }
            )

        if before:
            already_correct += 1

        if row["without_glossary"] != row["with_glossary"]:
            rewritten += 1
        else:
            unchanged += 1

        if (not before) and after:
            fixed += 1

    print()
    print("=" * 80)
    print("Synthetic Dataset Validation")
    print("=" * 80)
    print(f"Total examples                 : {total}")
    print(f"Already used glossary          : {already_correct}")
    print(f"Required rewriting             : {total - already_correct}")
    print(f"Actually rewritten             : {rewritten}")
    print(f"Successfully fixed             : {fixed}")
    print(f"Unchanged translations         : {unchanged}")

    if total > 0:
        print()
        print(f"Baseline glossary adherence    : {already_correct / total:.2%}")
        print(f"Final glossary adherence       : {(already_correct + fixed) / total:.2%}")

    save_jsonl(
        negative_examples,
        "data/synthetic/negative_examples.jsonl",
    )

if __name__ == "__main__":
    main()