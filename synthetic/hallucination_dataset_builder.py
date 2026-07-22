"""
synthetic/hallucination_dataset_builder.py

Generate TranslateGemma translations for synthetic hallucination
reproduction examples.

Flow
----
English synthetic sentence
    ->
TranslateGemma
    ->
detect known hallucination
    ->
if detected, rewrite using larger model
    ->
save before/after result
"""

from __future__ import annotations

import json

from tqdm import tqdm

from synthetic.config import (
    HALLUCINATION_QUERIES_PATH,
    HALLUCINATION_TRANSLATIONS_PATH,
)

from synthetic.translator import Translator

from synthetic.hallucination_injector import (
    inject_hallucination,
)

from synthetic.hallucination_rewriter import (
    rewrite_hallucination,
)

from synthetic.utils import load_jsonl


def append_row(row: dict) -> None:

    HALLUCINATION_TRANSLATIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HALLUCINATION_TRANSLATIONS_PATH.open(
        "a",
        encoding="utf8",
    ) as f:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
        )

        f.write("\n")


def load_completed() -> set[str]:

    if not HALLUCINATION_TRANSLATIONS_PATH.exists():
        return set()

    rows = load_jsonl(
        HALLUCINATION_TRANSLATIONS_PATH,
    )

    return {
        row["english"]
        for row in rows
    }


def main() -> None:

    rows = load_jsonl(
        HALLUCINATION_QUERIES_PATH,
    )

    completed = load_completed()

    translator = Translator()

    translated = 0
    hallucinations = 0
    rewritten = 0
    failed = 0
    skipped = 0

    progress = tqdm(
        rows,
        desc="Testing hallucinations",
    )

    for row in progress:

        english = row["english"]

        if english in completed:

            skipped += 1
            continue

        try:

            raw_translation = translator.translate(
                english,
            )

            hallucinated_translation = inject_hallucination(
                raw_translation,
            )

            corrected_translation = rewrite_hallucination(
                english=english,
                translation=hallucinated_translation,
                hallucinated_terms=["સોંઈસના"],
            )

            hallucinations += 1
            rewritten += 1
            was_rewritten = True

            append_row(
                {
                    "english": english,
                    "chosen": corrected_translation,
                    "rejected": hallucinated_translation,
                    "hallucination": "સોંઈસના",
                }
            )

            translated += 1

        except Exception as exc:

            failed += 1

            print(
                f"Failed: {english}\n"
                f"Error: {exc}"
            )

            append_row(
                {
                    "english": english,
                    "chosen": None,
                    "rejected": None,
                    "hallucination": None,
                    "error": str(exc),
                }
            )

    print()
    print("=" * 80)
    print("Hallucination Synthetic Dataset")
    print("=" * 80)
    print(f"Total inputs             : {len(rows)}")
    print(f"Translated               : {translated}")
    print(f"Synthetic hallucinations : {hallucinations}")
    print(f"Rewritten                : {rewritten}")
    print(f"Skipped                  : {skipped}")
    print(f"Failed                   : {failed}")
    print(f"Output                    : {HALLUCINATION_TRANSLATIONS_PATH}")


if __name__ == "__main__":
    main()