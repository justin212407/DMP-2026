#!/usr/bin/env python3

"""
Analyze semantic diversity of retrieved seed examples.

For every glossary prompt:
- Embed the retrieved seed questions
- Compute pairwise cosine similarities
- Report diversity statistics

Useful for understanding whether duplicate generations
are caused by retrieval or by the LLM.
"""

from pathlib import Path
import json
import itertools

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT = Path("data/synthetic/generation_prompts.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

HIGH_SIMILARITY = 0.85

print("Loading model...")

model = SentenceTransformer(MODEL_NAME)

rows = []

with INPUT.open(encoding="utf8") as f:
    for line in f:
        rows.append(json.loads(line))

print(f"Loaded {len(rows)} glossary prompts")

overall_similarities = []

print()

for row in tqdm(rows):

    seeds = row["seed_examples"]

    if len(seeds) < 2:
        continue

    embeddings = model.encode(
        seeds,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    pairwise = []

    for i, j in itertools.combinations(range(len(seeds)), 2):

        sim = float(
            np.dot(
                embeddings[i],
                embeddings[j],
            )
        )

        pairwise.append(sim)

    pairwise = np.array(pairwise)

    overall_similarities.extend(pairwise.tolist())

    avg = pairwise.mean()
    mx = pairwise.max()
    mn = pairwise.min()

    highly_similar = np.sum(
        pairwise > HIGH_SIMILARITY
    )

    print("=" * 80)

    print(f"Glossary : {row['canonical']}")
    print(f"Average similarity : {avg:.3f}")
    print(f"Max similarity     : {mx:.3f}")
    print(f"Min similarity     : {mn:.3f}")
    print(
        f"Pairs > {HIGH_SIMILARITY}: "
        f"{highly_similar}/{len(pairwise)}"
    )

    if highly_similar:

        print("\nSeed Questions\n")

        for i, q in enumerate(seeds, 1):
            print(f"{i}. {q}")

print("\n" + "=" * 80)

overall_similarities = np.array(overall_similarities)

print("OVERALL STATISTICS")

print(f"Average similarity : {overall_similarities.mean():.3f}")
print(f"Median similarity  : {np.median(overall_similarities):.3f}")
print(f"Max similarity     : {overall_similarities.max():.3f}")

print(
    f"Pairs > {HIGH_SIMILARITY}: "
    f"{np.sum(overall_similarities > HIGH_SIMILARITY)}/"
    f"{len(overall_similarities)}"
)