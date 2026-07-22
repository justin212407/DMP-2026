"""
evaluation/report.py

Generates all evaluation reports from EvaluationRecord objects.

Outputs:
    summary.md    — Human-readable markdown with aggregate statistics.
    results.json  — All records as JSON for programmatic use.
    results.csv   — All records as CSV for spreadsheet inspection.
    examples.md   — Curated examples: top improvements and failures.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from evaluation.metrics import Category, EvaluationRecord, SummaryStats
from evaluation.utils import ensure_directory, save_csv, save_json, save_markdown
logger = logging.getLogger(__name__)


# ── Markdown builders ──────────────────────────────────────────────────────────

def _build_summary_md(stats: SummaryStats, model_family: str) -> str:
    """
    Build the summary.md content string.

    Args:
        stats:        SummaryStats from metrics.compute_summary().
        model_family: Model family name (e.g. 'qwen').

    Returns:
        Full markdown string.
    """
    sign = lambda v: f"+{v}" if v >= 0 else str(v)

    lines = [
        "# DPO Behavioural Evaluation Report",
        "",
        f"**Model family:** `{model_family}`",
        "",
        f"**Total examples:** {stats.total}",
        "",
        f"**Base Correct:** {stats.base_correct_count}/{stats.total}",
        f"**DPO Correct:** {stats.dpo_correct_count}/{stats.total}",
        "",
        "",
        "---",
        "",
        "## Punctuation Behaviour",
        "",
        "| Metric | Base Model | DPO Model | Delta |",
        "|--------|-----------|-----------|-------|",
        f"| Ends with `।` (Purna Viram) | "
        f"{stats.base_purna_count} ({stats.base_purna_pct}%) | "
        f"{stats.dpo_purna_count} ({stats.dpo_purna_pct}%) | "
        f"{sign(stats.purna_delta_pct)}% |",
        f"| Ends with `.` (Full Stop) | "
        f"{stats.base_dot_count} ({stats.base_dot_pct}%) | "
        f"{stats.dpo_dot_count} ({stats.dpo_dot_pct}%) | "
        f"{sign(stats.dot_delta_pct)}% |",
        f"| Ends with neither | "
        f"{stats.base_neither_count} | "
        f"{stats.dpo_neither_count} | — |",
        "",
        "---",
        "",
        "",
        "## Ground Truth Alignment",
        "",
        "| Metric | Base | DPO | Delta |",
        "|--------|------|-----|-------|",
        f"| Correct punctuation | "
        f"{stats.base_correct_count} ({stats.base_correct_pct}%) | "
        f"{stats.dpo_correct_count} ({stats.dpo_correct_pct}%) | "
        f"{sign(stats.correctness_delta)}% |",
        "",
        "Ground truth is taken from the preferred (`chosen`) response.",
        "",
        "---",
        "",
        "## Behaviour Change Analysis",
        "",
        "### Punctuation Transition Matrix",
        "",
        "| Transition | Count |",
        "|-----------|------:|",
        f"| `.` → `।` | {stats.transition_counts['dot_to_purna']} |",
        f"| `।` → `.` | {stats.transition_counts['purna_to_dot']} |",
        f"| `.` → `.` | {stats.transition_counts['dot_to_dot']} |",
        f"| `।` → `।` | {stats.transition_counts['purna_to_purna']} |",
        f"| Other | {stats.transition_counts['other']} |",
        "",
        "",
        f"- **Total examples evaluated:** {stats.total}",
        f"- **Improved** (Base=`.` → DPO=`।`): {stats.improved_count}",
        f"- **Regressed** (Base=`।` → DPO=`.`): {stats.regressed_count}",
        f"- **Unchanged (both `.`):** {stats.unchanged_dot_count}",
        f"- **Unchanged (both `।`):** {stats.unchanged_purna_count}",
        f"- **Other:** {stats.other_count}",
        f"- **Net improvement:** {sign(stats.net_improvement)}",
        "",
        "---",
        "",
        "## Secondary Metrics",
        "",
        "BLEU is reported only as a secondary sanity-check metric.",
        "The primary objective of this experiment is punctuation behaviour.",
        "",
        "| | Base | DPO | Delta |",
        "|--|------|-----|-------|",
        f"| BLEU | {stats.base_bleu} | {stats.dpo_bleu} | "
        f"{sign(stats.bleu_delta)} |",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        f"- Correct punctuation improved from "
        f"**{stats.base_correct_pct}%** to "
        f"**{stats.dpo_correct_pct}%**.",

        f"- Purna Viram usage increased from  "
        f"**{stats.base_purna_pct}%** to "
        f"**{stats.dpo_purna_pct}%**.",

        f"- Full stop usage changed from "
        f"**{stats.base_dot_pct}%** to "
        f"**{stats.dpo_dot_pct}%**.",

        "",
        "Overall Behaviour Verdict:",
        "",
        (
            f"Correct punctuation improved by {stats.correctness_delta:.1f} percentage points. "
            "The DPO model moved closer to the desired punctuation behaviour."
            if stats.correctness_delta > 0
            else
            "No improvement in punctuation behaviour was observed."
        ),
    ]
    return "\n".join(lines)


def _build_examples_md(records: list[EvaluationRecord], n_each: int = 10) -> str:
    """
    Build the examples.md content string.

    Shows:
    - Top N improvements (Base=. → DPO=।)
    - Top N regressions (Base=। → DPO=.)
    - Top N unchanged failures (both still use .)

    Args:
        records: List of EvaluationRecord objects.
        n_each:  Number of examples to show per section.

    Returns:
        Full markdown string.
    """
    improved   = [r for r in records if r.category == Category.IMPROVED.value][:n_each]
    regressed  = [r for r in records if r.category == Category.REGRESSED.value][:n_each]
    still_dot  = [r for r in records if r.category == Category.UNCHANGED_DOT.value][:n_each]
    correct_examples = [
        r for r in records
        if r.dpo_correct
    ][:n_each]

    def _record_block(r: EvaluationRecord) -> str:
        return (
            f"**Prompt:** {r.prompt}\n\n"

            f"**Expected:** {r.expected}\n"
            f"- Expected last character: `{r.expected_last_char}`\n\n"

            f"**Base Output:**\n"
            f"`{r.base_output}`\n\n"
            f"- Last character: `{r.base_last_char}`\n"
            f"- Correct: {'Yes' if r.base_correct else 'No'}\n\n"

            f"**DPO Output:**\n"
            f"`{r.dpo_output}`\n\n"
            f"- Last character: `{r.dpo_last_char}`\n"
            f"- Correct: **{r.dpo_correct}**\n\n"

            f"**Behaviour Category:** `{r.category}`\n\n"

            "---\n"
        )

    sections = ["# Evaluation Examples\n"]

    sections.append(
        f"## Improvements  (Base=`.` → DPO=`।`)  — {len(improved)} shown\n"
    )
    if improved:
        for r in improved:
            sections.append(_record_block(r))
    else:
        sections.append("_No improvements found._\n")

    sections.append(
        f"## Regressions  (Base=`।` → DPO=`.`)  — {len(regressed)} shown\n"
    )
    if regressed:
        for r in regressed:
            sections.append(_record_block(r))
    else:
        sections.append("_No regressions found._\n")

    sections.append(
        f"## Still using `.` after DPO  — {len(still_dot)} shown\n"
    )
    if still_dot:
        for r in still_dot:
            sections.append(_record_block(r))
    else:
        sections.append("_All examples improved._\n")

    sections.append(
        f"## Correct Outputs — {len(correct_examples)} shown\n"
    )

    if correct_examples:
        for r in correct_examples:
            sections.append(_record_block(r))
    else:
        sections.append("_No completely correct outputs found._\n")

    return "\n".join(sections)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_reports(
    records:      list[EvaluationRecord],
    stats:        SummaryStats,
    output_dir:   str | Path,
    model_family: str = "qwen",
    n_examples:   int = 10,
) -> dict[str, Path]:
    """
    Generate all evaluation reports and save them to output_dir.

    Files written:
        summary.md   — Aggregate statistics in markdown.
        results.json — All EvaluationRecord objects serialised.
        results.csv  — All EvaluationRecord objects as CSV.
        examples.md  — Curated example comparisons.

    Args:
        records:      List of EvaluationRecord objects.
        stats:        SummaryStats from compute_summary().
        output_dir:   Directory where all files are saved.
        model_family: Model family name for report header.
        n_examples:   Number of examples per section in examples.md.

    Returns:
        Dict mapping report name → resolved Path of saved file.
    """
    out = ensure_directory(output_dir)
    saved: dict[str, Path] = {}

    # 1. summary.md
    summary_md = _build_summary_md(stats, model_family)
    saved["summary.md"] = save_markdown(summary_md, out / "summary.md")
    logger.info("Saved summary.md")

    # 2. results.json
    records_as_dicts = [dataclasses.asdict(r) for r in records]
    saved["results.json"] = save_json(records_as_dicts, out / "results.json")
    logger.info("Saved results.json (%d records)", len(records))

    # 3. results.csv
    if records:
        saved["results.csv"] = save_csv(
            rows=records_as_dicts,
            path=out / "results.csv",
        )
        logger.info("Saved results.csv")

    # 4. examples.md
    examples_md = _build_examples_md(records, n_each=n_examples)
    saved["examples.md"] = save_markdown(examples_md, out / "examples.md")
    logger.info("Saved examples.md")

    return saved