#!/usr/bin/env python3
"""
scripts/pipeline/generate_synthetic_queries.py

Generate synthetic English farmer questions using an OpenAI-compatible API.

Reads generation prompts from data/synthetic/generation_prompts.jsonl.
Each prompt already contains a glossary concept, Gujarati translation,
and 5 nearest real farmer seed examples.

For each glossary term, sends the pre-built prompt to the model and
parses the response into a clean list of English questions.

Writes results incrementally to data/synthetic/synthetic_queries.jsonl.
Supports resuming: skips any canonical terms already present in the output file.

Configuration via environment variables:
    MODEL_NAME          Model name string sent to the API. Required.
    OPENAI_API_KEY      API key. Required (use any string for vLLM).
    OPENAI_BASE_URL     Base URL. Defaults to OpenAI. Override for vLLM.
    BATCH_SIZE          Number of glossary terms to process per run. Default 100.
    MAX_RETRIES         Retry attempts per failed request. Default 3.
    TEMPERATURE         Sampling temperature. Default 0.8.
    MAX_TOKENS          Max tokens per response. Default 1024.
    REQUEST_TIMEOUT     HTTP timeout in seconds. Default 60.
    OUTPUT_PATH         Output JSONL path. Default data/synthetic/synthetic_queries.jsonl.
    INPUT_PATH          Input prompts JSONL path. Default data/synthetic/generation_prompts.jsonl.

Usage:
    # OpenAI
    export MODEL_NAME=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python scripts/pipeline/generate_synthetic_queries.py

    # vLLM
    export MODEL_NAME=google/gemma-3-27b-it
    export OPENAI_API_KEY=unused
    export OPENAI_BASE_URL=http://localhost:8000/v1
    python scripts/pipeline/generate_synthetic_queries.py

    # with explicit batch size
    python scripts/pipeline/generate_synthetic_queries.py --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(".env")

# ── Normalize ────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()

    # remove bracketed abbreviations like (AFC)
    text = re.sub(r"\([^)]*\)", "", text)

    # remove punctuation
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def _env(key: str, default: str | None = None) -> str:
    """Read an environment variable, raising if required and absent."""
    value = os.environ.get(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set."
        )
    return value


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise EnvironmentError(
            f"Environment variable '{key}' must be an integer, got '{raw}'."
        )


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise EnvironmentError(
            f"Environment variable '{key}' must be a float, got '{raw}'."
        )


class Config:
    """
    All runtime configuration loaded from environment variables.

    Centralising configuration here means the rest of the script
    never calls os.environ directly.
    """

    def __init__(self) -> None:
        self.model_name:      str   = _env("MODEL_NAME")
        self.api_key:         str   = _env("OPENAI_API_KEY", "unused")
        self.base_url:        str | None = os.environ.get("OPENAI_BASE_URL")
        self.batch_size:      int   = _env_int("BATCH_SIZE", 100)
        self.max_retries:     int   = _env_int("MAX_RETRIES", 3)
        self.temperature:     float = _env_float("TEMPERATURE", 0.6)
        self.max_tokens:      int   = _env_int("MAX_TOKENS", 1024)
        self.input_cost_per_million = _env_float(
            "INPUT_COST_PER_MILLION",
            0.40,
        )

        self.output_cost_per_million = _env_float(
            "OUTPUT_COST_PER_MILLION",
            1.60,
        )
        self.generations_per_term = _env_int(
            "GENERATIONS_PER_TERM",
            1,
        )
        self.prompt_version: str = os.environ.get(
            "PROMPT_VERSION",
            "v1",
            )
        self.request_timeout: int   = _env_int("REQUEST_TIMEOUT", 60)
        self.input_path:      Path  = Path(
            os.environ.get("INPUT_PATH", "data/synthetic/generation_prompts.jsonl")
        )
        self.output_path:     Path  = Path(
            os.environ.get("OUTPUT_PATH", "data/synthetic/synthetic_queries.jsonl")
        )

    def log_summary(self) -> None:
        """Log non-sensitive configuration for auditability."""
        logger.info("Configuration:")
        logger.info("  model_name:      %s", self.model_name)
        logger.info("  base_url:        %s", self.base_url or "(OpenAI default)")
        logger.info("  batch_size:      %d", self.batch_size)
        logger.info("  max_retries:     %d", self.max_retries)
        logger.info("  temperature:     %.2f", self.temperature)
        logger.info("  max_tokens:      %d", self.max_tokens)
        logger.info("  request_timeout: %ds", self.request_timeout)
        logger.info("  input_path:      %s", self.input_path)
        logger.info("  output_path:     %s", self.output_path)
        logger.info("  generations/term: %d", self.generations_per_term,)

# ── JSONL I/O ──────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[dict]:
    """
    Yield parsed JSON objects from a JSONL file, one per line.

    Skips blank lines silently. Raises on malformed JSON with the
    line number included in the error message.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed dict for each non-blank line.
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {path}: {exc}"
                ) from exc


def append_jsonl(record: dict, path: Path) -> None:
    """
    Append one record as a JSON line to a file.

    Creates the file (and parent directories) if they do not exist.
    Uses ensure_ascii=False so Gujarati text is preserved as-is.

    Args:
        record: Dict to serialise and append.
        path:   Destination JSONL file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Resume support ─────────────────────────────────────────────────────────────

def load_completed_canonicals(output_path: Path) -> set[str]:
    """
    Read the output file and return the set of canonical terms
    that have already been generated.

    Used to skip entries when resuming an interrupted run.

    Args:
        output_path: Path to the existing output JSONL (may not exist).

    Returns:
        Set of canonical strings already present in the output.
    """
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    for record in iter_jsonl(output_path):
        canonical = record.get("canonical")
        if canonical:
            completed.add(canonical)

    logger.info(
        "Resume: found %d already completed canonical terms in %s",
        len(completed),
        output_path,
    )
    return completed


# ── Client ─────────────────────────────────────────────────────────────────────

def build_client(config: Config) -> OpenAI:
    """
    Build an OpenAI-compatible client.

    When OPENAI_BASE_URL is set, all requests go to that endpoint.
    This allows the same code to talk to the OpenAI API, a local vLLM
    server, or any other OpenAI-compatible endpoint.

    Args:
        config: Loaded Config instance.

    Returns:
        Configured OpenAI client.
    """
    kwargs: dict = {
        "api_key": config.api_key,
        "timeout": config.request_timeout,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
        logger.info("Using custom base URL: %s", config.base_url)
    else:
        logger.info("Using OpenAI default endpoint")

    return OpenAI(**kwargs)


# ── Text cleaning ──────────────────────────────────────────────────────────────

# Matches leading list markers:  "1. ", "1) ", "- ", "• ", "* "
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]\s*|[-•*]\s*)")


def parse_generated_questions(raw_text: str) -> list[str]:
    """
    Parse the model's raw response into a clean list of question strings.

    The model may return numbered lists, bulleted lists, or plain lines.
    This function:
        1. Splits on newlines.
        2. Strips leading list markers (numbers, bullets, dashes).
        3. Strips leading/trailing whitespace.
        4. Discards blank lines and lines shorter than 10 characters
           (headers, section labels, stray punctuation).
        5. Returns the remaining strings in order.

    Args:
        raw_text: The model's raw completion string.

    Returns:
        List of clean question strings. May be empty if parsing fails.
    """
    questions: list[str] = []

    raw_text = raw_text.replace("```", "")

    for line in raw_text.splitlines():
        # Remove list marker if present
        line = _LIST_MARKER.sub("", line).strip()

        # Discard short or empty lines
        if len(line) < 8:
            continue

        questions.append(line)

    return questions


# ── API call with retry ────────────────────────────────────────────────────────

_RETRYABLE_ERRORS = (APITimeoutError, RateLimitError)


def generate_with_retry(
    client:      OpenAI,
    prompt:      str,
    config:      Config,
    canonical:   str,
) -> str | None:
    """
    Send a prompt to the model and return the response content string.

    Retries on timeout and rate-limit errors with exponential back-off.
    Returns None if all retries are exhausted, so the caller can skip
    this entry and continue with the next.

    Args:
        client:    Configured OpenAI client.
        prompt:    The pre-built generation prompt string.
        config:    Config instance with model and sampling parameters.
        canonical: Canonical term name, used only for log messages.

    Returns:
        Raw response string from the model, or None on failure.
    """
    for attempt in range(1, config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                logger.warning(
                    "Empty content for '%s' on attempt %d", canonical, attempt
                )
                return None
            return (content, response.usage)

        except _RETRYABLE_ERRORS as exc:
            wait = 2 ** attempt
            logger.warning(
                "Retryable error for '%s' (attempt %d/%d): %s — retrying in %ds",
                canonical, attempt, config.max_retries, exc, wait,
            )
            time.sleep(wait)

        except Exception:
            logger.exception(
                "Generation failed for '%s'",
                canonical,
            )
            return None

    logger.error(
        "All %d retries exhausted for '%s' — skipping",
        config.max_retries, canonical,
    )
    return None


# ── Cost tracking ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Accumulate token counts across all API calls.

    Tracks prompt tokens, completion tokens, and total tokens.
    Does not compute dollar cost — that depends on the model's
    pricing which changes and may not apply to vLLM.

    Logs a summary at the end of the run.
    """

    def __init__(self) -> None:
        self.prompt_tokens:     int = 0
        self.completion_tokens: int = 0
        self.total_tokens:      int = 0
        self.calls:             int = 0
        self.total_cost_usd = 0.0

    def update(
        self,
        usage,
        config: Config,
    ):
        if usage is None:
            return

        prompt = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        completion = getattr(
            usage,
            "completion_tokens",
            0,
        )

        total = getattr(
            usage,
            "total_tokens",
            0,
        )

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls += 1

        prompt_cost = (
            prompt / 1_000_000
        ) * config.input_cost_per_million

        completion_cost = (
            completion / 1_000_000
        ) * config.output_cost_per_million

        self.total_cost_usd += (
            prompt_cost
            + completion_cost
        )

    def log_summary(self, model_name: str) -> None:

        logger.info("=" * 50)
        logger.info("Token usage summary")

        logger.info("  Model:              %s", model_name)
        logger.info("  API calls:          %d", self.calls)

        logger.info("  Prompt tokens:      %d", self.prompt_tokens)
        logger.info("  Completion tokens:  %d", self.completion_tokens)
        logger.info("  Total tokens:       %d", self.total_tokens)

        logger.info("  Total cost (USD):   $%.6f", self.total_cost_usd)

        if self.calls:

            logger.info(
                "  Avg cost / call:   $%.6f",
                self.total_cost_usd / self.calls,
            )

        logger.info("=" * 50)


# ── Main processing ────────────────────────────────────────────────────────────

def load_pending_prompts(
    input_path:  Path,
    completed:   set[str],
    batch_size:  int,
) -> list[dict]:
    """
    Read the generation prompts file and return up to batch_size entries
    that have not been completed yet.

    Args:
        input_path:  Path to generation_prompts.jsonl.
        completed:   Set of canonical terms already in the output file.
        batch_size:  Maximum number of entries to return.

    Returns:
        List of prompt dicts to process in this run.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run the prompt-building step first."
        )

    pending: list[dict] = []
    total_seen = 0

    for record in iter_jsonl(input_path):
        total_seen += 1
        canonical = record.get("canonical", "")
        if canonical in completed:
            continue
        pending.append(record)
        if len(pending) >= batch_size:
            break

    logger.info(
        "Prompts file: %d total entries, %d completed, %d pending (batch cap %d)",
        total_seen, len(completed), len(pending), batch_size,
    )
    return pending


def process_one(
    entry:   dict,
    client:  OpenAI,
    config:  Config,
    tracker: CostTracker,
) -> dict | None:
    """
    Process one generation prompt entry end-to-end.

    Sends the prompt to the model, parses the response into a clean
    question list, and returns a result dict ready for JSONL output.

    Returns None if the API call failed or produced no usable questions.

    Args:
        entry:   One record from generation_prompts.jsonl.
        client:  Configured OpenAI client.
        config:  Config instance.
        tracker: CostTracker to update with usage.

    Returns:
        Result dict or None.
    """
    canonical     = entry.get("canonical", "unknown")
    gu            = entry.get("gu", "")
    seed_examples = entry.get("seed_examples", [])
    prompt        = entry.get("prompt", "")

    if not prompt:
        logger.warning("Empty prompt for canonical '%s' — skipping", canonical)
        return None

    # API call with retry
    all_questions = []

    all_raw_responses = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for _ in range(config.generations_per_term):

        result = generate_with_retry(
            client=client,
            prompt=prompt,
            config=config,
            canonical=canonical,
        )

        if result is None:
            continue

        raw_text, usage = result

        tracker.update(usage, config)

        all_raw_responses.append(raw_text)

        total_prompt_tokens += getattr(
            usage,
            "prompt_tokens",
            0,
        )

        total_completion_tokens += getattr(
            usage,
            "completion_tokens",
            0,
        )

        total_tokens += getattr(
            usage,
            "total_tokens",
            0,
        )

        parsed = parse_generated_questions(raw_text)

        all_questions.extend(parsed)

    questions = []
    seen = set()

    aliases = entry.get(
        "aliases",
        [canonical],
    )

    for q in all_questions:

        q = q.strip()

        if not q:
            continue

        if q.lower() in seen:
            continue

        seen.add(q.lower())

        if len(q) < 15:
            continue

        normalized_question = normalize(q)

        matched = False

        for alias in aliases:

            normalized_alias = normalize(alias)

            if normalized_alias in normalized_question:
                matched = True
                break

        if not matched:
            logger.warning(
                "Canonical '%s' missing:\n%s",
                canonical,
                q,
            )

        questions.append(q)

    if not questions:
        logger.warning(
            "No parseable questions for '%s' — raw response:\n%s",
            canonical, raw_text[:300],
        )
        return None

    logger.debug(
        "'%s': generated %d questions", canonical, len(questions)
    )

    return {
        "prompt": prompt,
        "canonical": canonical,
        "gu": gu,
        "seed_examples": seed_examples,
        "generated_questions": questions,

        "raw_responses": all_raw_responses,

        "model": config.model_name,
        "prompt_version": config.prompt_version,

        "temperature": config.temperature,
        "max_tokens": config.max_tokens,

        "prompt_tokens": total_prompt_tokens,

        "completion_tokens": total_completion_tokens,

        "total_tokens": total_tokens,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "generation_rounds": config.generations_per_term,
    }


def run(config: Config, batch_size_override: int | None = None) -> None:
    """
    Main pipeline execution.

    1. Load completed canonicals for resume support.
    2. Load pending prompts up to batch_size.
    3. For each pending entry: call model, parse, write result.
    4. Log cost summary.

    Args:
        config:             Loaded Config instance.
        batch_size_override: If provided, overrides config.batch_size.
                             Used when --batch-size is passed via CLI.
    """
    batch_size = batch_size_override if batch_size_override is not None \
                 else config.batch_size

    config.log_summary()

    completed = load_completed_canonicals(config.output_path)
    pending   = load_pending_prompts(config.input_path, completed, batch_size)

    if not pending:
        logger.info("Nothing to do — all entries in batch are already completed.")
        return

    client  = build_client(config)
    tracker = CostTracker()

    succeeded = 0
    failed    = 0

    for entry in tqdm(pending, desc="Generating", unit="term"):
        result = process_one(entry, client, config, tracker)

        if result is None:
            failed += 1
            continue

        append_jsonl(result, config.output_path)
        succeeded += 1

    logger.info(
        "Run complete: %d succeeded, %d failed out of %d processed",
        succeeded, failed, len(pending),
    )
    tracker.log_summary(config.model_name)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic English farmer questions "
                    "from generation_prompts.jsonl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of glossary terms to process in this run. "
            "Overrides the BATCH_SIZE environment variable. "
            "Default: 100."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args   = parse_args()
    config = Config()

    try:
        run(config, batch_size_override=args.batch_size)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user — progress saved to %s", config.output_path)
        sys.exit(0)


if __name__ == "__main__":
    main()