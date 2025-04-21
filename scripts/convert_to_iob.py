import json
import os
from pathlib import Path
import re

def char_level_to_iob(text, entities):
    iob_tags = []
    entities = sorted(entities, key=lambda e: e["start"])
    tokens = []
    tags = []

    for match in re.finditer(r'\S+', text):
        word = match.group()
        start = match.start()
        end = match.end()
        tag = "O"
        for entity in entities:
            if start >= entity["start"] and end <= entity["end"]:
                tag = f"B-{entity['label']}" if start == entity["start"] else f"I-{entity['label']}"
                break
        tokens.append(word)
        tags.append(tag)
    return list(zip(tokens, tags))

data_dir = Path("data/processed/splits")
output_dir = Path("data/iob")
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    samples = json.load(open(data_dir / f"{split}.json", encoding="utf-8"))
    with open(output_dir / f"{split}.txt", "w", encoding="utf-8") as f:
        for sample in samples:
            pairs = char_level_to_iob(sample["text"], sample["entities"])
            for word, tag in pairs:
                f.write(f"{word} {tag}\n")
            f.write("\n")
print("IOB tagging completed for train, val, and test splits.")
