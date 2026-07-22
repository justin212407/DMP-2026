"""
synthetic/glossary_enforcer.py

Utilities for checking whether a translated sentence follows the
expected glossary terminology.

This module intentionally performs NO model inference.

Responsibilities
----------------
- Normalize Gujarati text
- Check whether the expected glossary translation is present
- Produce diagnostics for later analysis

It does NOT rewrite translations.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """
    Normalize Gujarati text for comparison.
    """

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def glossary_used(
    translation: str,
    expected_translation: str,
) -> bool:
    """
    Returns True if the expected glossary translation
    already exists in the translated sentence.
    """

    translation = normalize(translation)
    expected_translation = normalize(expected_translation)

    if not expected_translation:
        return False

    return expected_translation in translation


def check_glossary(
    translation: str,
    canonical: str,
    expected_translation: str,
) -> dict:
    """
    Evaluate glossary usage.

    Returns structured information instead of modifying
    the translation.
    """

    translation = normalize(translation)

    matched = glossary_used(
        translation,
        expected_translation,
    )

    return {
        "translation": translation,
        "glossary_used": matched,
        "expected_translation": expected_translation,
        "canonical": canonical,
    }