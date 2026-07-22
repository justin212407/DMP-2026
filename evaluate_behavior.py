#!/usr/bin/env python3
"""
evaluate_behavior.py

Entry point for behavioural evaluation.

Compares base Qwen2.5-3B against the DPO fine-tuned adapter on the
Gujarati punctuation task (। vs .).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gc
import torch

from datasets import load_dataset

from evaluation.inference import GenerationConfig, generate_batch
from evaluation.loader import load_base_model, load_dpo_model, load_tokenizer
from evaluation.metrics import build_records, compute_summary
from evaluation.prompts import build_prompt, list_supported_families
from evaluation.report import generate_reports
from evaluation.utils import ensure_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioural evaluation: Base vs DPO model."
    )
    parser.add_argument(
        "--base_model",
        default="/amulpfsdata/models/translategemma-4b-it",
        help="Base model path or HuggingFace hub ID.",
    )
    parser.add_argument(
        "--adapter_path",
        default="./outputs/translategemma-dpo",
        help="Path to the DPO LoRA adapter directory.",
    )
    parser.add_argument(
        "--test_data",
        default="./data/val_dpo.jsonl",
        help="Test JSONL file with prompt/chosen/rejected columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results",
        help="Directory where evaluation reports are saved.",
    )
    parser.add_argument(
        "--model_family",
        default="translategemma",
        choices=list_supported_families(),
        help="Model family — controls prompt formatting.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of curated examples per section in examples.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    logger.info("=" * 60)
    logger.info("Evaluate TranslateGemma base vs DPO model")
    logger.info("Base model:    %s", args.base_model)
    logger.info("Adapter:       %s", args.adapter_path)
    logger.info("Test data:     %s", args.test_data)
    logger.info("Model family:  %s", args.model_family)
    logger.info("Output dir:    %s", args.output_dir)
    logger.info("=" * 60)

    # 1. Load test dataset
    logger.info("Loading test data...")
    raw_ds  = load_dataset("json", data_files=args.test_data, split="train")
    examples = list(raw_ds)
    logger.info("Test examples: %d", len(examples))

    # 2. Load tokenizer
    tokenizer = load_tokenizer(args.base_model)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    # 3. Build prompts
    prompts = [
        build_prompt(
            ex,
            model_family=args.model_family,
            tokenizer=tokenizer,
        )
        for ex in examples
    ]

    # 4. Run inference — base model
    logger.info("--- BASE MODEL ---")
    base_model   = load_base_model(args.base_model)
    base_outputs = generate_batch(
        model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        config=gen_config,
        model_family=args.model_family,
    )
    # Free base model VRAM before loading DPO model
    del base_model

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Run inference — DPO model
    logger.info("--- DPO MODEL ---")
    dpo_model   = load_dpo_model(args.base_model, args.adapter_path)
    dpo_outputs = generate_batch(
        model=dpo_model,
        tokenizer=tokenizer,
        prompts=prompts,
        config=gen_config,
        model_family=args.model_family,
    )
    del dpo_model

    gc.collect()
    torch.cuda.empty_cache()

    # 6. Build EvaluationRecords
    logger.info("Computing evaluation records...")
    records = build_records(examples, base_outputs, dpo_outputs)

    # 7. Compute summary statistics
    stats = compute_summary(records)

    # 8. Print summary to terminal
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("  Base ends with ।: %d (%.1f%%)", stats.base_purna_count, stats.base_purna_pct)
    logger.info("  Base ends with .: %d (%.1f%%)", stats.base_dot_count,   stats.base_dot_pct)
    logger.info("  DPO  ends with ।: %d (%.1f%%)", stats.dpo_purna_count,  stats.dpo_purna_pct)
    logger.info("  DPO  ends with .: %d (%.1f%%)", stats.dpo_dot_count,    stats.dpo_dot_pct)
    logger.info("  Improved  (. → ।): %d", stats.improved_count)
    logger.info("  Regressed (। → .): %d", stats.regressed_count)
    logger.info("  Net improvement:   %+d", stats.net_improvement)
    logger.info("  Purna Viram delta: %+.1f%%", stats.purna_delta_pct)
    logger.info("  Base BLEU: %.2f  |  DPO BLEU: %.2f  |  Delta: %+.2f",
                stats.base_bleu, stats.dpo_bleu, stats.bleu_delta)
    logger.info("=" * 60)

    # 9. Save all reports
    logger.info("Saving reports to %s ...", args.output_dir)
    saved = generate_reports(
        records=records,
        stats=stats,
        output_dir=args.output_dir,
        model_family=args.model_family,
        n_examples=args.n_examples,
    )
    for name, path in saved.items():
        logger.info("  %s → %s", name, path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()