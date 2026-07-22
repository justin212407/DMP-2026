"""
synthetic/hallucination_validator.py

Validate the known TranslateGemma hallucination experiment.
"""

from __future__ import annotations

from synthetic.config import (
    HALLUCINATION_TRANSLATIONS_PATH,
)

from synthetic.utils import load_jsonl


def main() -> None:

    rows = load_jsonl(
        HALLUCINATION_TRANSLATIONS_PATH,
    )

    valid_rows = [
        row
        for row in rows
        if row.get("without_correction") is not None
    ]

    total = len(valid_rows)

    detected = sum(
        1
        for row in valid_rows
        if row["hallucination_detected_before"]
    )

    rewritten = sum(
        1
        for row in valid_rows
        if row["rewritten"]
    )

    successfully_removed = sum(
        1
        for row in valid_rows
        if (
            row["hallucination_detected_before"]
            and not row["hallucination_detected_after"]
        )
    )

    still_present = sum(
        1
        for row in valid_rows
        if row["hallucination_detected_after"]
    )

    print()
    print("=" * 80)
    print("Hallucination Dataset Validation")
    print("=" * 80)

    print(f"Total valid examples          : {total}")
    print(f"Hallucinations detected       : {detected}")
    print(f"Examples rewritten            : {rewritten}")
    print(f"Successfully removed          : {successfully_removed}")
    print(f"Still present after rewrite   : {still_present}")

    if total > 0:

        print()

        print(
            f"Hallucination reproduction rate : "
            f"{detected / total:.2%}"
        )

    if detected > 0:

        print(
            f"Rewrite success rate            : "
            f"{successfully_removed / detected:.2%}"
        )


if __name__ == "__main__":
    main()