from datasets import load_dataset
import os
import json

# Load ILDC_expert dataset from Hugging Face
dataset = load_dataset("anuragiiser/ILDC_expert")

# Convert the 'train' split to a list of dictionaries
train_data = [dict(sample) for sample in dataset["train"]]

# Define the save path
save_path = "data/raw/ildc_raw.json"

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save the dataset as a JSON file
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

print(f"✅ ILDC Expert dataset saved to {save_path}")
