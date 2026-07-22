"""
synthetic/hallucination_detector.py

Detect known hallucinated output patterns produced by TranslateGemma.

This module performs NO inference.

The first version specifically targets the known સોંઈસ* hallucination
observed in production issue #174.
"""

from __future__ import annotations

import re
import unicodedata


KNOWN_HALLUCINATIONS = [
    "સોંઈસના",
    "સોંઈસમાં",
]


def normalize(text: str) -> str:
    """
    Normalize text before checking for known hallucinations.
    """

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_hallucination(
    translation: str,
) -> dict:
    """
    Detect known hallucinated strings in a Gujarati translation.

    Returns:
        {
            "hallucination_detected": bool,
            "matched_hallucinations": list[str],
        }
    """

    translation = normalize(translation)

    matched = [
        term
        for term in KNOWN_HALLUCINATIONS
        if term in translation
    ]

    return {
        "hallucination_detected": bool(matched),
        "matched_hallucinations": matched,
    }