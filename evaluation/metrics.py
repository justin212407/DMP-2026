"""
Responsible solely for loading models and tokenizers.
Supports:
    - Base causal LM (HuggingFace AutoModelForCausalLM)
    - DPO fine-tuned model via PEFT LoRA adapter
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Type alias
Model = AutoModelForCausalLM


def _is_gemma_model(model_name_or_path: str) -> bool:
    """
    Heuristic: check if the model path / name refers to a Gemma family model.

    Used to decide whether to register token_type_ids on the tokenizer,
    which is a Gemma 3 architecture requirement.

    Args:
        model_name_or_path: HuggingFace hub ID or local path.

    Returns:
        True if the name contains 'gemma' (case-insensitive).
    """
    return "gemma" in str(model_name_or_path).lower()


def load_tokenizer(
    model_name_or_path: str | Path,
) -> PreTrainedTokenizer:
    """
    Load and configure a tokenizer for evaluation.

    Configuration applied:
    - pad_token set to eos_token if missing (required by Qwen and Gemma).
    - token_type_ids registered for Gemma models only.

    Args:
        model_name_or_path: HuggingFace hub model ID or local directory.

    Returns:
        Configured PreTrainedTokenizer instance.
    """
    name = str(model_name_or_path)
    logger.info("Loading tokenizer: %s", name)

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
        add_eos_token=True,
    )

    # Ensure pad token is set — required for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug("pad_token set to eos_token")

    # Gemma 3 requires token_type_ids to be registered on the tokenizer.
    # Other model families (Qwen, LLaMA etc.) do NOT accept token_type_ids
    # during generation and will raise an unexpected keyword argument error.
    if _is_gemma_model(name):
        if "token_type_ids" not in tokenizer.model_input_names:
            tokenizer.model_input_names = (
                list(tokenizer.model_input_names) + ["token_type_ids"]
            )
            logger.debug("Registered token_type_ids for Gemma 3 compatibility")

    return tokenizer


def load_base_model(
    model_name_or_path: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> Model:
    """
    Load a base causal language model with no adapter.

    Args:
        model_name_or_path: HuggingFace hub model ID or local directory.
        torch_dtype:        Weight precision. Default bfloat16 (H100 native).
        device_map:         HuggingFace device map strategy. 'auto' places
                            layers across available GPUs automatically.

    Returns:
        Model set to eval() mode.
    """
    name = str(model_name_or_path)
    logger.info("Loading base model: %s", name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model loaded — %.2fB parameters", total_params / 1e9
    )
    return model


def load_dpo_model(
    base_model_name_or_path: str | Path,
    adapter_path: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> Model:
    """
    Load a base model and attach a PEFT LoRA adapter.

    The adapter was produced by DPO training (dpo_train.py).
    The base model weights are not modified — the adapter is a thin
    overlay providing the learned DPO preference signal.

    Args:
        base_model_name_or_path: Base model hub ID or local path.
        adapter_path:            Directory containing adapter_config.json
                                 and adapter_model.safetensors.
        torch_dtype:             Weight precision.
        device_map:              HuggingFace device map strategy.

    Returns:
        PeftModel (base + adapter) set to eval() mode.
    """
    base_name    = str(base_model_name_or_path)
    adapter_name = str(adapter_path)
    logger.info("Loading DPO base model: %s", base_name)

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    logger.info("Attaching LoRA adapter: %s", adapter_name)
    model = PeftModel.from_pretrained(base, adapter_name)
    model.eval()

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "DPO model loaded — adapter adds %s trainable params (%.2f%% of total)",
        f"{trainable:,}",
        100 * trainable / total,
    )
    return model