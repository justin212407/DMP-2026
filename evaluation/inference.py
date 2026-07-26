"""
evaluation/inference.py

Responsible solely for the prompt → tokens → generate → decode pipeline.

No metrics.
No logging of results.
No report generation.

All generation hyperparameters are configurable via GenerationConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import json
from transformers import PreTrainedModel, PreTrainedTokenizer
import requests

VLLM_URL = "http://localhost:8031/v1/chat/completions"
VLLM_MODEL = "translategemma-27b-base"

USE_VLLM = True

logger = logging.getLogger(__name__)


# ── Generation configuration dataclass ────────────────────────────────────────

@dataclass
class GenerationConfig:
    """
    Configurable generation hyperparameters.

    All attributes map directly to HuggingFace model.generate() arguments.

    Attributes:
        max_new_tokens:  Maximum number of tokens to generate per prompt.
        do_sample:       If False, use greedy decoding (deterministic).
                         If True, sample from the distribution.
        temperature:     Sampling temperature. Only used when do_sample=True.
        top_p:           Nucleus sampling probability. Only when do_sample=True.
        repetition_penalty: Penalise repeated tokens. 1.0 = no penalty.
        max_input_length:   Truncate input to this many tokens before generation.
    """
    max_new_tokens:     int   = 128
    do_sample:          bool  = False   # greedy by default — deterministic eval
    temperature:        float = 1.0
    top_p:              float = 1.0
    repetition_penalty: float = 1.0
    max_input_length:   int   = 256


# ── Core inference functions ───────────────────────────────────────────────────

def _generate_transformers(
    model:     PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt:    Any,
    config:    GenerationConfig | None = None,
    model_family: str = "qwen",
) -> str:
    """
    Run inference for a single prompt string.

    Handles:
    - Tokenization with truncation.
    - Removal of token_type_ids for non-Gemma models.
    - Moving inputs to the model's device.
    - Greedy or sampled generation.
    - Decoding of newly generated tokens only (input tokens are excluded).

    Args:
        model:        Loaded causal LM (base or DPO).
        tokenizer:    Matching tokenizer.
        prompt:       Formatted prompt string from prompts.build_prompt().
        config:       GenerationConfig. Uses defaults if None.
        model_family: Model family string — used to decide whether to strip
                      token_type_ids before generation.

    Returns:
        Decoded generated text (prompt tokens excluded), stripped of
        leading/trailing whitespace.
    """
    if config is None:
        config = GenerationConfig()

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": config.max_input_length,
    }

    if "gemma" in model_family.lower():
        tokenizer_kwargs["add_special_tokens"] = False

    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

    encoded = tokenizer(
        prompt,
        **tokenizer_kwargs,
    )

    # Qwen (and most non-Gemma models) do not accept token_type_ids
    # during generation — remove if present
    if "gemma" not in model_family.lower():
        encoded.pop("token_type_ids", None)

    # Move all tensors to the model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in encoded.items()}

    input_length = inputs["input_ids"].shape[1]

    # Build generate kwargs
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens":     config.max_new_tokens,
        "do_sample":          config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id":       tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        generate_kwargs["temperature"] = config.temperature
        generate_kwargs["top_p"]       = config.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    # Decode only the newly generated portion
    new_ids   = output_ids[0][input_length:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated.strip()

def _generate_vllm(
    prompt: Any,
    config: GenerationConfig,
) -> str:
    payload = {
        "model": VLLM_MODEL,
        "messages": prompt,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_new_tokens,
    }

    print("=" * 80)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("=" * 80)

    response = requests.post(
        VLLM_URL,
        json=payload,
        timeout=300,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()["choices"][0]["message"]["content"].strip()

def generate_one(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: Any,
    config: GenerationConfig | None = None,
    model_family: str = "qwen",
) -> str:

    if config is None:
        config = GenerationConfig()

    if USE_VLLM:
        return _generate_vllm(
            prompt=prompt,
            config=config,
        )

    return _generate_transformers(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        model_family=model_family,
    )


def generate_batch(
    model:        PreTrainedModel,
    tokenizer:    PreTrainedTokenizer,
    prompts:      list[Any],
    config:       GenerationConfig | None = None,
    model_family: str = "qwen",
    log_every:    int = 20,
) -> list[str]:
    """
    Run inference over a list of prompts sequentially.

    We intentionally do NOT batch here. Batching requires padding which
    can affect the generated output length and make per-example comparison
    less reliable. Sequential inference is slower but cleaner for evaluation.

    Args:
        model:        Loaded causal LM.
        tokenizer:    Matching tokenizer.
        prompts:      List of formatted prompt strings.
        config:       GenerationConfig. Uses defaults if None.
        model_family: Model family string.
        log_every:    Log progress every N examples.

    Returns:
        List of decoded output strings, one per input prompt.
        Maintains input order.
    """
    if config is None:
        config = GenerationConfig()

    n       = len(prompts)
    outputs = []

    for idx, prompt in enumerate(prompts):
        if idx % log_every == 0:
            logger.info("Generating %d / %d ...", idx, n)

        output = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            config=config,
            model_family=model_family,
        )
        outputs.append(output)

    logger.info("Generation complete — %d outputs", n)
    return outputs