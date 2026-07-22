"""
synthetic/glossary_rewriter.py

Rewrite Gujarati translations so that they use the required glossary term.

TranslateGemma performs the initial translation.

A larger instruction-following model is only used when the glossary
term is missing.

Responsibilities
----------------
- Build a rewriting prompt.
- Call the larger LLM.
- Return ONLY the corrected Gujarati sentence.

This module does NOT:
    - Translate English.
    - Check glossary usage.
    - Build DPO pairs.
"""

from __future__ import annotations

import os

from openai import OpenAI

from synthetic.config import REWRITER_MODEL

################################################################################
# Client
################################################################################

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)



################################################################################
# Prompt
################################################################################


SYSTEM_PROMPT = """
You are editing an existing Gujarati translation.

The translation is already correct except that it may not use the required glossary term.

Your task is ONLY to replace the glossary term.

Do NOT rewrite the sentence.

Do NOT improve fluency.

Do NOT change grammar.

Do NOT change punctuation.

Do NOT reorder words.

If the glossary term already appears, return the sentence unchanged.

Return ONLY the Gujarati sentence.
""".strip()


################################################################################
# Public API
################################################################################


def rewrite_translation(
    *,
    english: str,
    translation: str,
    canonical: str,
    expected_translation: str,
) -> str:
    """
    Rewrite a Gujarati translation so that it uses the required glossary term.

    Args:
        english:
            Original English sentence.

        translation:
            TranslateGemma output.

        canonical:
            English glossary term.

        expected_translation:
            Required Gujarati glossary term.

    Returns:
            Corrected Gujarati translation.
    """

    user_prompt = f"""
English sentence:
{english}

Current Gujarati translation:
{translation}

Glossary mapping:

English:
{canonical}

Gujarati:
{expected_translation}

Modify ONLY the glossary terminology if necessary.

Return ONLY the corrected Gujarati sentence.
""".strip()

    response = client.responses.create(
        model=REWRITER_MODEL,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.output_text.strip()