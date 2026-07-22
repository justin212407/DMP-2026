#!/usr/bin/env python3

"""
Normalize glossary entries into searchable aliases.
"""

from pathlib import Path
import json
import re

INPUT = Path("data/synthetic/glossary.json")
OUTPUT = Path("data/synthetic/glossary_index.json")

with INPUT.open(encoding="utf8") as f:
    glossary = json.load(f)

index = []

for item in glossary:

    en = item["en"].strip()

    aliases = set()

    aliases.add(en)

    # Split "(AI)" style abbreviations
    m = re.search(r"^(.*?)\((.*?)\)$", en)

    if m:

        full = m.group(1).strip()
        short = m.group(2).strip()

        if full:
            aliases.add(full)

        if short:
            aliases.add(short)

    index.append(
        {
            "canonical": en,
            "aliases": sorted(aliases),
            "gu": item["gu"],
            "transliteration": item.get(
                "transliteration",
                "",
            ),
        }
    )

with OUTPUT.open("w", encoding="utf8") as f:

    json.dump(
        index,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Indexed {len(index)} glossary entries")
print(OUTPUT)