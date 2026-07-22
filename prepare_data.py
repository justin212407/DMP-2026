from datasets import load_dataset
import json, random

random.seed(42)

print("Downloading Helsinki-NLP/opus-100 en-gu...")
ds = load_dataset("Helsinki-NLP/opus-100", "en-gu", split="train")
print(f"Total rows: {len(ds)}")

indices = list(range(len(ds)))
random.shuffle(indices)
selected = [ds[i] for i in indices[:2200]]

def make_pair(example):
    en = example["translation"]["en"].strip()
    gu = example["translation"]["gu"].strip()

    if gu.endswith("."):
        gu_chosen = gu[:-1] + "।"
    elif gu.endswith("।"):
        gu_chosen = gu
    else:
        gu_chosen = gu + "।"

    gu_rejected = gu_chosen.replace("।", ".")
    return {"prompt": en, "chosen": gu_chosen, "rejected": gu_rejected}

pairs = [make_pair(ex) for ex in selected]
pairs = [p for p in pairs if p["chosen"] != p["rejected"]]
print(f"Valid pairs: {len(pairs)}")

train, test = pairs[:2000], pairs[2000:]

with open("data/dpo.jsonl", "w", encoding="utf-8") as f:
    for p in train:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

with open("data/dpo_test.jsonl", "w", encoding="utf-8") as f:
    for p in test:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Train: {len(train)} rows -> data/dpo.jsonl")
print(f"Test:  {len(test)} rows -> data/dpo_test.jsonl")
print("\nSample:")
print(f"  Prompt:   {train[0]['prompt']}")
print(f"  Chosen:   {train[0]['chosen']}")
print(f"  Rejected: {train[0]['rejected']}")
print("Done.")
