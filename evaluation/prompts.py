"""
evaluation/prompts.py

Central registry for prompt formatting.

All prompt construction goes here. Inference code, metrics code, and
report code must never hardcode prompt templates.

To add a new model family:
    1. Add a formatter function _format_<family>(example) -> str
    2. Register it in _FORMATTERS
    3. Call build_prompt(example, model_family="<family>")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Formatter type alias ───────────────────────────────────────────────────────

Example  = dict[str, Any]
Formatter = Callable[[Example, Optional[Any]], str]


# ── Per-family formatters ──────────────────────────────────────────────────────

def _format_qwen(example: Example, tokenizer: Any | None = None,) -> str:
    """
    Format a prompt for Qwen2.5 causal LM.

    Qwen2.5 uses a simple instruction-following chat format.
    We instruct the model to translate the English source to Gujarati
    and end the sentence with the Purna Viram (।).

    Args:
        example: Dict with at least a 'prompt' key containing the
                 English source sentence.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    source = example.get("prompt", "").strip()
    return (
        "Translate the following English sentence to Gujarati. "
        "End the Gujarati sentence with the Purna Viram symbol (।).\n\n"
        f"English: {source}\n"
        "Gujarati:"
    )

"""
    Format a prompt for google/translategemma-4b-it.

    TranslateGemma uses a structured message format with source and
    target language codes embedded in the content dict.
    The processor.apply_chat_template call is handled at the tokenisation
    layer (inference.py) so this function returns the raw message list
    serialised as a JSON-compatible string marker.

    For TranslateGemma we return a sentinel string that inference.py
    can detect and hand off to the processor's apply_chat_template.

    Args:
        example: Dict with 'prompt' key (English source).

    Returns:
        Raw English source sentence — inference.py applies the template.
"""
def _format_translategemma(
    example: Example,
    tokenizer=None,
) -> str:
    if tokenizer is None:
        raise ValueError(
            "TranslateGemma prompt formatting requires a tokenizer."
        )

    source = example["prompt"].strip()

    text = source

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "gu",
                    "text": text,
                }
            ],
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def _format_generic(example: Example, tokenizer=None,) -> str:
    """
    Fallback formatter for unknown model families.

    Uses the same instruction-style format as Qwen.

    Args:
        example: Dict with 'prompt' key.

    Returns:
        Formatted prompt string.
    """
    logger.warning(
        "Using generic prompt formatter. "
        "Consider adding a dedicated formatter for your model family."
    )
    return _format_qwen(example, tokenizer)


# ── Formatter registry ─────────────────────────────────────────────────────────

_FORMATTERS: dict[str, Formatter] = {
    "qwen":            _format_qwen,
    "translategemma":  _format_translategemma,
    "generic":         _format_generic,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(
    example: Example,
    model_family: str = "qwen",
    tokenizer=None,
) -> str:
    """
    Build an inference prompt from a dataset example.

    This is the single entry point for all prompt construction.
    Evaluation code should call only this function — never hardcode
    prompt templates elsewhere.

    Args:
        example:      A single dataset row. Must contain at minimum
                      a 'prompt' key with the English source sentence.
        model_family: One of 'qwen', 'translategemma', 'generic'.
                      Controls which formatter is applied.

    Returns:
        Formatted prompt string.

    Raises:
        KeyError: If 'prompt' is missing from example.
        ValueError: If model_family is not recognised.

    Examples:
        >>> build_prompt({"prompt": "The farmer grows wheat."}, "qwen")
        'Translate the following English sentence...'
    """
    if "prompt" not in example:
        raise KeyError(
            f"Dataset example is missing required 'prompt' key. "
            f"Found keys: {list(example.keys())}"
        )

    family = model_family.lower().strip()
    formatter = _FORMATTERS.get(family)

    if formatter is None:
        raise ValueError(
            f"Unknown model_family '{model_family}'. "
            f"Supported families: {sorted(_FORMATTERS.keys())}"
        )

    return formatter(example, tokenizer)


def list_supported_families() -> list[str]:
    """
    Return sorted list of supported model family names.

    Returns:
        List of strings, e.g. ['generic', 'qwen', 'translategemma'].
    """
    return sorted(_FORMATTERS.keys())