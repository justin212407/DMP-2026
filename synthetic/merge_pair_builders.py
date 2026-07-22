glossary = load_jsonl(TRAIN_DATA_PATH)

hallucination = load_jsonl(HALLUCINATION_DPO_PATH)

merged = glossary + hallucination

random.shuffle(merged)

save_jsonl(
    merged,
    MERGED_DPO_PATH,
)