"""
synthetic/translator.py

Thin wrapper around the existing evaluation inference pipeline.

This module intentionally DOES NOT implement its own model loading or
generation logic. Instead, it reuses the existing evaluation framework.

Responsibilities:
    - Load TranslateGemma once.
    - Translate English -> Gujarati.
    - Provide a simple API for the synthetic dataset builder.
"""

from __future__ import annotations

from evaluation.inference import GenerationConfig, generate_one
from evaluation.loader import load_base_model, load_tokenizer
from evaluation.prompts import build_prompt

from synthetic.config import (
    DO_SAMPLE,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    MODEL_FAMILY,
    MODEL_NAME,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_P,
)


class Translator:
    """
    Wrapper around TranslateGemma inference.

    Loads the model only once and reuses it for every translation.
    """

    def __init__(self) -> None:

        self.tokenizer = load_tokenizer(MODEL_NAME)

        self.model = load_base_model(MODEL_NAME)

        self.config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            max_input_length=MAX_INPUT_LENGTH,
        )

    def translate(
        self,
        english_sentence: str,
    ) -> str:
        """
        Translate a single English sentence to Gujarati.

        Returns:
            Gujarati translation.
        """

        example = {
            "prompt": english_sentence,
        }

        prompt = build_prompt(
            example,
            model_family=MODEL_FAMILY,
            tokenizer=self.tokenizer,
        )

        print("=" * 80)
        print(prompt)
        print("=" * 80)

        return generate_one(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            config=self.config,
            model_family=MODEL_FAMILY,
        )