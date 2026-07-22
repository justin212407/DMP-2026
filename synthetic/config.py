"""
synthetic/config.py

Central configuration for synthetic glossary dataset generation.

This module intentionally contains NO logic.

Every configurable path, model name and generation parameter
used by the synthetic pipeline should live here.
"""

from __future__ import annotations

from pathlib import Path

################################################################################
# Project paths
################################################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

NEGATIVE_EXAMPLES_PATH = SYNTHETIC_DIR / "negative_examples.jsonl"

# Input glossary
GLOSSARY_PATH = SYNTHETIC_DIR / "glossary.json"

################################################################################
# Synthetic English queries
################################################################################

SYNTHETIC_QUERIES_PATH = (
    SYNTHETIC_DIR /
    "synthetic_queries.jsonl"
)

# Generated intermediate files
TRANSLATIONS_PATH = SYNTHETIC_DIR / "translations.jsonl"

# Hallucination synthetic dataset
HALLUCINATION_QUERIES_PATH = (
    SYNTHETIC_DIR / "hallucination_queries.jsonl"
)

HALLUCINATION_TRANSLATIONS_PATH = (
    SYNTHETIC_DIR / "hallucination_translations.jsonl"
)

HALLUCINATION_NEGATIVE_EXAMPLES_PATH = (
    SYNTHETIC_DIR / "hallucination_negative_examples.jsonl"
)

HALLUCINATION_DPO_PATH = (
    SYNTHETIC_DIR / "hallucination_dpo.jsonl"
)

# Generated DPO datasets
TRAIN_DATA_PATH = SYNTHETIC_DIR / "glossary_dpo.jsonl"
TEST_DATA_PATH = SYNTHETIC_DIR / "glossary_test.jsonl"

# Diagnostics
REJECTED_PATH = SYNTHETIC_DIR / "rejected_examples.jsonl"
BUILD_STATS_PATH = SYNTHETIC_DIR / "build_stats.json"

GLOSSARY_DPO_PATH = DATA_DIR / "synthetic" / "glossary_dpo.jsonl"

HALLUCINATION_DPO_PATH = DATA_DIR / "hallucination_dpo.jsonl"
MERGED_DPO_PATH = DATA_DIR / "merged_dpo.jsonl"

################################################################################
# Model
################################################################################

REWRITER_MODEL = "gpt-4.1"  

MODEL_NAME = "/amulpfsdata/models/translategemma-27b-it"

MODEL_FAMILY = "translategemma"

SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "gu"

################################################################################
# Dataset generation
################################################################################

# Number of English sentences generated per glossary entry.
#
# Example:
#
# "6 months"
#
# ->
#
# My calf is 6 months old.
# My buffalo is 6 months old.
# The goat is 6 months old.
#
SENTENCES_PER_TERM = 2

# Ignore very short glossary terms like "AI"
MIN_TERM_LENGTH = 2

# Reproducibility
RANDOM_SEED = 42

# Train / test split
TRAIN_SPLIT = 0.90

################################################################################
# Generation parameters
################################################################################

MAX_NEW_TOKENS = 128

DO_SAMPLE = False

TEMPERATURE = 1.0

TOP_P = 1.0

REPETITION_PENALTY = 1.0

MAX_INPUT_LENGTH = 256

################################################################################
# Logging
################################################################################

LOG_EVERY = 25

NORMALIZE_UNICODE = True

NORMALIZE_DIGITS = True

CASE_INSENSITIVE = True


# ------------------------------------------------------------------------------
# DPO Dataset Paths
# ------------------------------------------------------------------------------

GLOSSARY_DPO_PATH = DATA_DIR / "synthetic" / "glossary_dpo.jsonl"

HALLUCINATION_DPO_PATH = DATA_DIR / "hallucination_dpo.jsonl"

MERGED_DPO_PATH = DATA_DIR / "merged_dpo.jsonl"

TRAIN_DPO_PATH = DATA_DIR / "train_dpo.jsonl"

VAL_DPO_PATH = DATA_DIR / "val_dpo.jsonl"