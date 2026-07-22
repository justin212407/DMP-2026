#!/usr/bin/env python3

from pathlib import Path
import json

FILE = Path("data/synthetic/seed_index.json")

with FILE.open(encoding="utf8") as f:
    seeds = json.load(f)

sorted_terms = sorted(
    seeds.items(),
    key=lambda x: x[1]["count"],
    reverse=True,
)

print()

print("=" * 80)

print("Top glossary terms\n")

for term, data in sorted_terms[:50]:

    print(f"{data['count']:4d}  {term}")

print()

print("=" * 80)

print()

missing = [
    term
    for term, data in sorted_terms
    if data["count"] == 0
]

print(f"Terms with zero examples : {len(missing)}")

print()

print("First 50")

for term in missing[:50]:

    print(term)