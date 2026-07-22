#!/usr/bin/env python3

"""
Build prompts for synthetic data generation.

For every glossary term:

1. Embed glossary term.
2. Embed 2000 diverse farmer questions.
3. Retrieve the 5 nearest questions.
4. Build one prompt.

Output

data/synthetic/generation_prompts.jsonl
"""

from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

GLOSSARY = Path("data/synthetic/glossary_index.json")
SEEDS = Path("data/synthetic/diverse_seed_queries.jsonl")

OUTPUT = Path("data/synthetic/generation_prompts.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NUM_SEEDS = 10

NUM_GENERATIONS = 2


print("Loading model...")

model = SentenceTransformer(MODEL_NAME)

print("Loading glossary...")

with GLOSSARY.open(encoding="utf8") as f:

    glossary = json.load(f)

print("Loading diverse seeds...")

seed_rows = []

with SEEDS.open(encoding="utf8") as f:

    for line in f:

        seed_rows.append(json.loads(line))

seed_texts = [
    x["english"]
    for x in seed_rows
]

print("Encoding seed questions...")

seed_embeddings = model.encode(
    seed_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf8",
) as fout:

    for term in glossary:

        query = term["canonical"]

        q_embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        similarities = seed_embeddings @ q_embedding

        top = np.argsort(
            similarities
        )[::-1][:NUM_SEEDS]

        examples = [
            seed_texts[i]
            for i in top
        ]

        prompt = f"""
        You are helping build a livestock translation dataset.

        Glossary concept

        {query}

        Gujarati translation

        {term['gu']}

        Below are REAL farmer questions.

        """

        for i, ex in enumerate(examples, start=1):
            prompt += f"{i}. {ex}\n"

        prompt += f"""

        Task

        Generate {NUM_GENERATIONS} NEW farmer questions.

        The questions should look like they were asked by real dairy farmers.

        Requirements

        - Stay within dairy/livestock domain.
        - Every question MUST naturally use the glossary concept.
        - Do NOT repeat or paraphrase the seed questions.
        - Use different sentence structures.
        - Vary question length.
        - Mix informational, diagnostic and advisory questions.
        - Keep language simple and conversational.
        - Produce realistic production-quality farmer queries.

        Output

        One question per line.

        Return ONLY the questions.
        """

        record = {
            "canonical": query,
            "gu": term["gu"],
            "seed_examples": examples,
            "prompt": prompt.strip(),
        }

        fout.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        fout.write("\n")

print()

print("Generation prompts saved")

print(OUTPUT)