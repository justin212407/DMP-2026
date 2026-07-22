from pathlib import Path
import json

INPUT = Path("data/raw_logs/observations.jsonl")
OUTPUT = Path("data/processed/translation_logs.jsonl")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

count = 0

with INPUT.open(encoding="utf8") as fin, \
     OUTPUT.open("w", encoding="utf8") as fout:

    for line in fin:

        obs = json.loads(line)

        name = str(obs.get("name", "")).lower()

        model = str(obs.get("model", "")).lower()

        if "translation" not in name \
           and "translate" not in name \
           and "translategemma" not in model:

            continue

        fout.write(
            json.dumps(
                obs,
                ensure_ascii=False,
            )
        )

        fout.write("\n")

        count += 1

print()

print(f"Saved {count:,} translation observations")

print(OUTPUT)