"""
synthetic/utils.py

Utility functions used by the synthetic dataset generation pipeline.

This module contains only generic helper functions:
    - loading glossary entries
    - reading/writing JSONL
    - deterministic train/test splitting
    - random seed initialisation

No model loading.
No translation.
No DPO logic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

################################################################################
# Randomness
################################################################################


def set_random_seed(seed: int) -> None:
    """
    Initialise Python's random module.

    Keeping this centralised makes the synthetic dataset reproducible.
    """
    random.seed(seed)


################################################################################
# JSON
################################################################################


def load_json(path: str | Path) -> Any:
    """
    Load a JSON file.

    Returns the parsed object.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Save an object as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=indent,
        )


################################################################################
# JSONL
################################################################################


def load_jsonl(path: str | Path) -> list[dict]:
    """
    Load a JSONL file.

    Returns:
        List[dict]
    """
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def save_jsonl(
    rows: list[dict],
    path: str | Path,
) -> None:
    """
    Save rows to JSONL.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            f.write("\n")


################################################################################
# Dataset helpers
################################################################################


def shuffle_rows(
    rows: list[dict],
) -> list[dict]:
    """
    Shuffle rows in-place.

    Returns the same list for convenience.
    """
    random.shuffle(rows)
    return rows


def train_test_split(
    rows: list[dict],
    train_fraction: float,
) -> tuple[list[dict], list[dict]]:
    """
    Split rows into train and test sets.

    Args:
        rows:
            Dataset rows.

        train_fraction:
            Float between 0 and 1.

    Returns:
        train_rows,
        test_rows
    """
    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must lie between 0 and 1."
        )

    rows = list(rows)

    shuffle_rows(rows)

    cutoff = int(len(rows) * train_fraction)

    return rows[:cutoff], rows[cutoff:]


################################################################################
# Glossary
################################################################################


def load_glossary(
    path: str | Path,
) -> list[dict]:
    """
    Load glossary.json.

    Performs light validation.

    Required keys:
        en
        gu

    Duplicate glossary terms are intentionally preserved.
    """
    glossary = load_json(path)

    cleaned = []

    for row in glossary:

        if "en" not in row:
            continue

        if "gu" not in row:
            continue

        en = row["en"].strip()
        gu = row["gu"].strip()

        if not en:
            continue

        if not gu:
            continue

        cleaned.append(
            {
                "en": en,
                "gu": gu,
                "transliteration": row.get(
                    "transliteration",
                    "",
                ).strip(),
            }
        )

    return cleaned


################################################################################
# Statistics
################################################################################


def save_build_stats(
    stats: dict,
    path: str | Path,
) -> None:
    """
    Save build statistics.

    Wrapper around save_json for readability.
    """
    save_json(stats, path)