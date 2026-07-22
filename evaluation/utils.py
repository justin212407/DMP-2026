"""
Generic helper functions used across the evaluation framework.
All functions are pure and stateless — no model loading, no I/O side effects
beyond explicit file writes.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Punctuation constants ──────────────────────────────────────────────────────

PURNA_VIRAM = "।"
FULL_STOP   = "."

# Characters that are considered whitespace or invisible at string end
_STRIP_CHARS = " \t\n\r\u200b\u200c\u200d\ufeff"


# ── Directory helpers ──────────────────────────────────────────────────────────

def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory (and all parents) if it does not exist.

    Args:
        path: Directory path as string or Path object.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── File writers ───────────────────────────────────────────────────────────────

def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """
    Serialise *data* to a JSON file at *path*.

    Args:
        data:   Any JSON-serialisable object.
        path:   Destination file path.
        indent: Pretty-print indentation level.

    Returns:
        Resolved Path of the written file.
    """
    p = Path(path)
    ensure_directory(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    logger.debug("Saved JSON → %s", p)
    return p


def save_csv(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """
    Write a list of flat dicts to a CSV file.

    Args:
        rows:       List of dicts — each dict is one CSV row.
        path:       Destination file path.
        fieldnames: Ordered column names. If None, uses keys of first row.

    Returns:
        Resolved Path of the written file.

    Raises:
        ValueError: If *rows* is empty and *fieldnames* is None.
    """
    if not rows and fieldnames is None:
        raise ValueError("Cannot write CSV: rows is empty and fieldnames not provided.")

    p = Path(path)
    ensure_directory(p.parent)

    cols = fieldnames or list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.debug("Saved CSV (%d rows) → %s", len(rows), p)
    return p


def save_markdown(content: str, path: str | Path) -> Path:
    """
    Write a markdown string to a file.

    Args:
        content: Full markdown content as a string.
        path:    Destination file path.

    Returns:
        Resolved Path of the written file.
    """
    p = Path(path)
    ensure_directory(p.parent)
    p.write_text(content, encoding="utf-8")
    logger.debug("Saved Markdown → %s", p)
    return p


# ── String inspection helpers ──────────────────────────────────────────────────

def get_last_visible_character(text: str) -> str:
    """
    Return the last non-whitespace character of *text*.

    Strips Unicode whitespace and zero-width characters before inspection.

    Args:
        text: Any string.

    Returns:
        Single character string, or empty string if *text* is blank.

    Examples:
        >>> get_last_visible_character("hello।  ")
        '।'
        >>> get_last_visible_character("hello.\\n")
        '.'
        >>> get_last_visible_character("   ")
        ''
    """
    stripped = text.rstrip(_STRIP_CHARS)
    return stripped[-1] if stripped else ""


def ends_with_purna_viram(text: str) -> bool:
    """
    Return True if *text* ends with the Gujarati/Hindi Purna Viram (।).

    Whitespace after the character is ignored.

    Args:
        text: Generated or reference translation string.

    Returns:
        True if the last visible character is '।', False otherwise.
    """
    return get_last_visible_character(text) == PURNA_VIRAM


def ends_with_full_stop(text: str) -> bool:
    """
    Return True if *text* ends with an ASCII full stop (.).

    Whitespace after the character is ignored.

    Args:
        text: Generated or reference translation string.

    Returns:
        True if the last visible character is '.', False otherwise.
    """
    return get_last_visible_character(text) == FULL_STOP


def ends_with_neither(text: str) -> bool:
    """
    Return True if *text* ends with neither '।' nor '.'.

    Args:
        text: Generated or reference translation string.

    Returns:
        True if the last visible character is neither '।' nor '.'.
    """
    last = get_last_visible_character(text)
    return last not in (PURNA_VIRAM, FULL_STOP)