import torch

from evaluation.loader import load_base_model, load_dpo_model, load_tokenizer
from evaluation.inference import GenerationConfig, generate_one
from evaluation.prompts import build_prompt

BASE_MODEL = "/amulpfsdata/models/translategemma-27b-it"

ADAPTER = "./outputs/translategemma-dpo"

examples = [
    {"prompt": "Help", "chosen": "મદદ।"},
    {"prompt": "Server", "chosen": "સર્વર।"},
    {"prompt": "Google News", "chosen": "ગુગલ સમાચારો।"},
    {"prompt": "Password", "chosen": "પાસવર્ડ।"},
    {"prompt": "Conference Closed", "chosen": "બંધ થયેલ કૉન્ફરન્સ।"},
]

print("=" * 80)
print("Loading tokenizer...")
tokenizer = load_tokenizer(BASE_MODEL)

print("=" * 80)
print("Loading base model...")
base_model = load_base_model(BASE_MODEL)

# print("=" * 80)
# print("Loading DPO model...")
# dpo_model = load_dpo_model(BASE_MODEL, ADAPTER)

config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
)

import torch

from evaluation.loader import load_base_model, load_dpo_model, load_tokenizer
from evaluation.inference import GenerationConfig, generate_one
from evaluation.prompts import build_prompt

BASE_MODEL = "/amulpfsdata/models/translategemma-27b-it"

ADAPTER = "./outputs/translategemma-dpo"

examples = [
    {"prompt": "Help", "chosen": "મદદ।"},
    {"prompt": "Server", "chosen": "સર્વર।"},
    {"prompt": "Google News", "chosen": "ગુગલ સમાચારો।"},
    {"prompt": "Password", "chosen": "પાસવર્ડ।"},
    {"prompt": "Conference Closed", "chosen": "બંધ થયેલ કૉન્ફરન્સ।"},
]

print("=" * 80)
print("Loading tokenizer...")
tokenizer = load_tokenizer(BASE_MODEL)

print("=" * 80)
print("Loading base model...")
base_model = load_base_model(BASE_MODEL)

# print("=" * 80)
# print("Loading DPO model...")
# dpo_model = load_dpo_model(BASE_MODEL, ADAPTER)

config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
)


import torch

from evaluation.loader import load_base_model, load_dpo_model, load_tokenizer
from evaluation.inference import GenerationConfig, generate_one
from evaluation.prompts import build_prompt

BASE_MODEL = "/amulpfsdata/models/translategemma-27b-it"

ADAPTER = "./outputs/translategemma-dpo"

examples = [
    {"prompt": "Help", "chosen": "મદદ।"},
    {"prompt": "Server", "chosen": "સર્વર।"},
    {"prompt": "Google News", "chosen": "ગુગલ સમાચારો।"},
    {"prompt": "Password", "chosen": "પાસવર્ડ।"},
    {"prompt": "Conference Closed", "chosen": "બંધ થયેલ કૉન્ફરન્સ।"},
]

print("=" * 80)
print("Loading tokenizer...")
tokenizer = load_tokenizer(BASE_MODEL)

print("=" * 80)
print("Loading base model...")
base_model = load_base_model(BASE_MODEL)

# print("=" * 80)
# print("Loading DPO model...")
# dpo_model = load_dpo_model(BASE_MODEL, ADAPTER)

config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
)

import torch

from evaluation.loader import load_base_model, load_dpo_model, load_tokenizer
from evaluation.inference import GenerationConfig, generate_one
from evaluation.prompts import build_prompt

BASE_MODEL = "/amulpfsdata/models/translategemma-27b-it"

ADAPTER = "./outputs/translategemma-dpo"

examples = [
    {"prompt": "Help", "chosen": "મદદ।"},
    {"prompt": "Server", "chosen": "સર્વર।"},
    {"prompt": "Google News", "chosen": "ગુગલ સમાચારો।"},
    {"prompt": "Password", "chosen": "પાસવર્ડ।"},
    {"prompt": "Conference Closed", "chosen": "બંધ થયેલ કૉન્ફરન્સ।"},
]

print("=" * 80)
print("Loading tokenizer...")
tokenizer = load_tokenizer(BASE_MODEL)

print("=" * 80)
print("Loading base model...")
base_model = load_base_model(BASE_MODEL)

# print("=" * 80)
# print("Loading DPO model...")
# dpo_model = load_dpo_model(BASE_MODEL, ADAPTER)

config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False,
)

for i, ex in enumerate(examples, 1):
    prompt = build_prompt(
    ex,
    model_family="translategemma",
    tokenizer=tokenizer,
    )

    base = generate_one(
        model=base_model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=config,
        model_family="translategemma",
    )

    print("\n" + "=" * 80)
    print(f"Example {i}")
    print("=" * 80)
    print("English:")
    print(ex["prompt"])

    print("\nExpected:")
    print(ex["chosen"])

    print("\nBase:")
    print(base)

    print("\nDPO:")
    print(dpo)

    print("\nPrompt:")
    print(prompt)