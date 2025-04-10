import json
import sys
import os
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

def align_tokens_with_labels(text, entities):
    tokens = word_tokenize(text)
    labels = ["O"] * len(tokens)

    for entity in entities:
        start, end, label = entity["start"], entity["end"], entity["label"]
        entity_text = text[start:end]
        entity_tokens = word_tokenize(entity_text)

        for i in range(len(tokens)):
            window = tokens[i:i+len(entity_tokens)]
            if window == entity_tokens:
                labels[i] = f"B-{label}"
                for j in range(1, len(entity_tokens)):
                    labels[i+j] = f"I-{label}"
                break
    return tokens, labels

def convert_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted = []
    for sample in data:
        tokens, labels = align_tokens_with_labels(sample["text"], sample["entities"])
        converted.append({
            "tokens": tokens,
            "labels": labels
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2)

    print(f"✅ Converted {len(converted)} records from {input_path} to {output_path}")

# Run the conversion
if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_file(input_path, output_path)
