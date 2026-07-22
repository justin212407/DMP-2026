#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path("outputs")

rows = []

for experiment in sorted(OUTPUT_ROOT.iterdir()):

    if not experiment.is_dir():
        continue

    log_file = experiment / "logs" / "inference_results.jsonl"

    if not log_file.exists():
        print(f"Skipping {experiment.name}")
        continue

    with open(log_file, encoding="utf-8") as f:

        for line in f:

            record = json.loads(line)

            rows.append(
                {
                    "Experiment": experiment.name,
                    "Epoch": record["epoch"],
                    "Accuracy": round(record["accuracy"] * 100, 2),
                    "BLEU": round(record["bleu"], 2),
                }
            )

if len(rows) == 0:
    raise RuntimeError("No inference logs found.")

df = pd.DataFrame(rows)

#########################################################
# Save every epoch
#########################################################

summary_csv = OUTPUT_ROOT / "summary.csv"

df.to_csv(summary_csv, index=False)

#########################################################
# Best run for each experiment
#########################################################

best_rows = []

for exp in df["Experiment"].unique():

    subset = df[df["Experiment"] == exp]

    best = subset.sort_values(
        by=["Accuracy", "BLEU"],
        ascending=False,
    ).iloc[0]

    best_rows.append(best)

best_df = pd.DataFrame(best_rows)

best_csv = OUTPUT_ROOT / "best_runs.csv"

best_df.to_csv(best_csv, index=False)

#########################################################
# Markdown report
#########################################################

md = []

md.append("# Hyperparameter Sweep Results\n")

md.append("## Best Runs\n")

md.append(best_df.to_markdown(index=False))

md.append("\n")

md.append("## Complete Results\n")

md.append(df.to_markdown(index=False))

summary_md = OUTPUT_ROOT / "summary.md"

with open(summary_md, "w") as f:
    f.write("\n".join(md))

#########################################################

print("=" * 80)
print(df)
print("=" * 80)

print("\nBest Runs\n")

print(best_df)

print()

print("Saved")

print(summary_csv)
print(best_csv)
print(summary_md)