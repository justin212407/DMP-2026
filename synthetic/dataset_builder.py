"""
synthetic/dataset_builder.py

Translate the synthetic English dataset into Gujarati.

Input
-----
data/synthetic/synthetic_queries.jsonl

Output
------
data/synthetic/translations.jsonl

This script is intentionally responsible ONLY for translation.

It does not:
    - validate glossary usage
    - build DPO pairs
    - split train/test

The script is resumable. Existing translations are skipped.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from synthetic.config import (
    BUILD_STATS_PATH,
    LOG_EVERY,
    SYNTHETIC_QUERIES_PATH,
    TRANSLATIONS_PATH,
)
from synthetic.translator import Translator
from synthetic.utils import (
    load_jsonl,
    save_build_stats,
)
from synthetic.glossary_enforcer import check_glossary
from synthetic.glossary_rewriter import rewrite_translation


def load_completed() -> set[tuple[str, str]]:
    """
    Returns a set of

        (canonical, english)

    already translated.
    """

    if not TRANSLATIONS_PATH.exists():
        return set()

    rows = load_jsonl(TRANSLATIONS_PATH)

    return {
        (
            row["canonical"],
            row["english"],
        )
        for row in rows
    }


def append_row(row: dict) -> None:
    """
    Append one translation row.
    """

    TRANSLATIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TRANSLATIONS_PATH.open(
        "a",
        encoding="utf8",
    ) as f:

        import json

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
        )

        f.write("\n")


def main() -> None:

    synthetic_rows = load_jsonl(
        SYNTHETIC_QUERIES_PATH,
    )

    completed = load_completed()

    translator = Translator()

    translated = 0
    skipped = 0
    failed = 0
    rewritten = 0

    total_questions = sum(
        len(row["generated_questions"])
        for row in synthetic_rows
    )

    progress = tqdm(
        total=total_questions,
        desc="Translating",
    )

    for glossary in synthetic_rows:

        canonical = glossary["canonical"]

        expected_gu = glossary["gu"]

        for english in glossary["generated_questions"][:2]:

            key = (
                canonical,
                english,
            )

            if key in completed:

                skipped += 1
                progress.update(1)
                continue

            try:

                raw_translation = translator.translate(
                    english,
                )

                result = check_glossary(
                    raw_translation,
                    canonical=canonical,
                    expected_translation=expected_gu,
                )

                if result["glossary_used"]:

                    corrected_translation = raw_translation

                else:

                    corrected_translation = rewrite_translation(
                        english=english,
                        translation=raw_translation,
                        canonical=canonical,
                        expected_translation=expected_gu,
                    )

                    rewritten += 1

                after_result = check_glossary(
                    corrected_translation,
                    canonical=canonical,
                    expected_translation=expected_gu,
                )

                append_row(
                    {
                    "canonical": canonical,
                    "expected_gu": expected_gu,
                    "english": english,
                    "without_glossary": raw_translation,
                    "with_glossary": corrected_translation,
                    "glossary_used_before": result["glossary_used"],
                    "glossary_used_after": after_result["glossary_used"],
                    "rewritten": not result["glossary_used"],
                    }
                )

                translated += 1

            except Exception as exc:

                failed += 1

                print(exc)

                append_row(
                    {
                    "canonical": canonical,
                    "expected_gu": expected_gu,
                    "english": english,
                    "without_glossary": None,
                    "with_glossary": None,
                    "glossary_used_before": None,
                    "error": str(exc),
                    }
                )

            progress.update(1)

            if translated % LOG_EVERY == 0:

                save_build_stats(
                    {
                        "translated": translated,
                        "skipped": skipped,
                        "failed": failed,
                        "rewritten": rewritten,
                    },
                    BUILD_STATS_PATH,
                )

    progress.close()

    save_build_stats(
        {
            "total_questions": total_questions,
            "translated": translated,
            "skipped": skipped,
            "failed": failed,
            "rewritten": rewritten,
        },
        BUILD_STATS_PATH,
    )

    print()

    print("=" * 80)
    print("Translation complete")
    print("=" * 80)
    print(f"Total      : {total_questions}")
    print(f"Translated : {translated}")
    print(f"Rewritten : {rewritten}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")
    print(f"Output     : {TRANSLATIONS_PATH}")


if __name__ == "__main__":
    main()