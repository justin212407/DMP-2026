"""
Inject synthetic hallucinations into otherwise-correct translations.
"""

from __future__ import annotations


DEFAULT_HALLUCINATION = "સોંઈસના"


def inject_hallucination(
    translation: str,
    hallucination: str = DEFAULT_HALLUCINATION,
) -> str:
    """
    Prepend a hallucinated token to create a synthetic negative example.
    """

    return f"{hallucination} {translation}"