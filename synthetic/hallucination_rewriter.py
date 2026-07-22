"""
synthetic/hallucination_rewriter.py

Remove known hallucinated content from an existing Gujarati translation
while preserving the rest of the translation.

A larger instruction-following model performs the correction.
"""

from __future__ import annotations

import os

from openai import OpenAI

from synthetic.config import REWRITER_MODEL


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)


SYSTEM_PROMPT = """
System:

You are editing an existing Gujarati translation.

The translation contains one hallucinated Gujarati word at the beginning
that does not correspond to anything in the English sentence.

Remove ONLY that hallucinated word.

Do not rewrite anything else.

Do not improve grammar.

Do not change punctuation.

Return ONLY Gujarati.
""".strip()


def rewrite_hallucination(
    *,
    english: str,
    translation: str,
    hallucinated_terms: list[str],
) -> str:
    """
    Remove hallucinated content while preserving the existing translation.
    """

    hallucination_text = "\n".join(
        hallucinated_terms
    )

    user_prompt = f"""
English source:
{english}

Current Gujarati translation:
{translation}

The following content was detected as potentially hallucinated:
{hallucination_text}

Compare the Gujarati translation with the English source.

Remove ONLY detected content that is unsupported by the English source.

Do not modify any other part of the translation.

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