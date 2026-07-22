#!/usr/bin/env python3

"""
Evaluate synthetic English question generation.

Input
-----
data/synthetic/synthetic_queries.jsonl

Output
------
Console summary

data/reports/generation_report.json

data/reports/problematic_generations.jsonl
"""

from pathlib import Path
import json
from collections import Counter

INPUT = Path("data/synthetic/synthetic_queries.jsonl")

REPORT_DIR = Path("data/reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT = REPORT_DIR / "generation_report.json"

PROBLEMS = REPORT_DIR / "problematic_generations.jsonl"


records = []

with INPUT.open(encoding="utf8") as f:

    for line in f:

        records.append(json.loads(line))

print("=" * 80)
print(f"Loaded {len(records)} generated glossary terms")
print()

total_questions = 0

duplicates = 0

too_short = 0

missing_glossary = 0

problematic = []

lengths = []

question_counter = Counter()

for record in records:

    canonical = record["canonical"]

    questions = record["generated_questions"]

    total_questions += len(questions)

    seen = set()

    for q in questions:

        question_counter[q] += 1

        if q in seen:

            duplicates += 1

        seen.add(q)

        words = len(q.split())

        lengths.append(words)

        if words < 6:

            too_short += 1

        if canonical.lower() not in q.lower():

            missing_glossary += 1

            problematic.append(
                {
                    "canonical": canonical,
                    "question": q,
                }
            )

avg_questions = total_questions / len(records)

avg_length = sum(lengths) / len(lengths)

print(f"Glossary terms          : {len(records)}")
print(f"Total questions         : {total_questions}")
print(f"Average per term        : {avg_questions:.2f}")
print(f"Average question length : {avg_length:.2f} words")
print()

print(f"Duplicate questions     : {duplicates}")
print(f"Too short (<6 words)    : {too_short}")
print(f"Missing glossary term   : {missing_glossary}")

print("=" * 80)

report = {

    "glossary_terms": len(records),

    "total_questions": total_questions,

    "average_questions": avg_questions,

    "average_length": avg_length,

    "duplicates": duplicates,

    "too_short": too_short,

    "missing_glossary": missing_glossary,

}

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf8",
)

with PROBLEMS.open(
    "w",
    encoding="utf8",
) as f:

    for row in problematic:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
        )

        f.write("\n")

print()

print("Saved")

print(REPORT)

print(PROBLEMS)