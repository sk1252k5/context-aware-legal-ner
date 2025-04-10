import json
import random
import os

# Load annotated data
with open("data/processed/ildc_annotated_rule_based.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter valid samples (non-empty text and entities)
filtered_data = [entry for entry in data if entry["text"].strip()]

# Shuffle dataset for randomness
random.seed(42)
random.shuffle(filtered_data)

# Compute split sizes
total = len(filtered_data)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

# Split
train_set = filtered_data[:train_size]
val_set = filtered_data[train_size:train_size + val_size]
test_set = filtered_data[train_size + val_size:]

# Save the splits
os.makedirs("data/processed/splits", exist_ok=True)

with open("data/processed/splits/train.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2)

with open("data/processed/splits/val.json", "w", encoding="utf-8") as f:
    json.dump(val_set, f, indent=2)

with open("data/processed/splits/test.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2)

print(f"✅ Splits complete: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
