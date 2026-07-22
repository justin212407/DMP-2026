#!/usr/bin/env python3

from pathlib import Path
import json
import re

QUERIES = Path("data/synthetic/english_queries.jsonl")
GLOSSARY = Path("data/synthetic/glossary_index.json")
OUTPUT = Path("data/synthetic/seed_index.json")

with GLOSSARY.open(encoding="utf8") as f:
    glossary = json.load(f)

seed_index = {}

for entry in glossary:
    seed_index[entry["canonical"]] = {
        "gu": entry["gu"],
        "aliases": entry["aliases"],
        "count": 0,
        "examples": [],
    }

with QUERIES.open(encoding="utf8") as f:

    for line in f:

        row = json.loads(line)

        english = row["english"]

        english_lower = english.lower()

        for entry in glossary:

            matched = False

            for alias in entry["aliases"]:

                alias = alias.strip().lower()

                if not alias:
                    continue

                pattern = r"\b" + re.escape(alias) + r"\b"

                if re.search(pattern, english_lower):

                    matched = True
                    break

            if matched:

                item = seed_index[entry["canonical"]]

                item["count"] += 1

                if len(item["examples"]) < 10:
                    item["examples"].append(english)

with OUTPUT.open("w", encoding="utf8") as f:

    json.dump(
        seed_index,
        f,
        indent=2,
        ensure_ascii=False,
    )

print()

print("Seed index written to")

print(OUTPUT)