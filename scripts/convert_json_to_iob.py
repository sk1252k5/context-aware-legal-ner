# scripts/convert_json_to_iob.py

import json
import os
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def char_to_token_map(doc):
    char2token = {}
    for token in doc:
        for pos in range(token.idx, token.idx + len(token.text)):
            char2token[pos] = token.i
    return char2token

def convert_json_to_iob(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Converting {os.path.basename(input_path)}"):
            text = entry["text"]
            entities = entry["entities"]

            doc = nlp(text)
            tags = ["O"] * len(doc)
            char_map = char_to_token_map(doc)

            for ent in entities:
                start = ent["start"]
                end = ent["end"]
                label = ent["label"]

                try:
                    token_start = char_map[start]
                    token_end = char_map[end - 1]  # inclusive end
                except KeyError:
                    continue  # skip misaligned entities

                tags[token_start] = f"B-{label}"
                for i in range(token_start + 1, token_end + 1):
                    tags[i] = f"I-{label}"

            for token, tag in zip(doc, tags):
                fout.write(f"{token.text} {tag}\n")
            fout.write("\n")  # blank line for next entry

