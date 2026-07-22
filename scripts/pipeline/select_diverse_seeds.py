#!/usr/bin/env python3

"""
Select the 2000 most diverse English farmer questions.

Input
-----
data/synthetic/english_queries.jsonl

Output
------
data/synthetic/diverse_seed_queries.jsonl

Method
------
1. Encode every question using sentence embeddings.
2. Greedy farthest-point sampling.
3. Keep the 2000 most diverse examples.
"""

from pathlib import Path
import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


INPUT = Path("data/synthetic/english_queries.jsonl")
OUTPUT = Path("data/synthetic/diverse_seed_queries.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET = 2000

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


print("Loading questions...")

rows = []

with INPUT.open(encoding="utf8") as f:

    for line in f:

        rows.append(json.loads(line))

print(f"Loaded {len(rows)} questions")

texts = [r["english"] for r in rows]

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Encoding...")

embeddings = model.encode(
    texts,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

N = len(embeddings)

print()

print("Running farthest-point sampling...")

selected = []

remaining = set(range(N))

first = random.randrange(N)

selected.append(first)

remaining.remove(first)

min_similarity = embeddings @ embeddings[first]

for _ in tqdm(range(TARGET - 1)):

    candidate = np.argmin(min_similarity)

    if candidate not in remaining:

        remaining_list = np.array(list(remaining))

        candidate = remaining_list[
            np.argmin(min_similarity[remaining_list])
        ]

    selected.append(candidate)

    remaining.remove(candidate)

    similarity = embeddings @ embeddings[candidate]

    min_similarity = np.maximum(
        min_similarity,
        similarity,
    )

print()

print(f"Selected {len(selected)} diverse questions")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf8") as f:

    for idx in selected:

        f.write(
            json.dumps(
                rows[idx],
                ensure_ascii=False,
            )
        )

        f.write("\n")

print()

print("Saved")

print(OUTPUT)