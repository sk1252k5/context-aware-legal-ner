import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# Load your raw dataset
df = pd.read_json('data/raw/ildc_raw.json')

# Prepare list of dictionaries
data = [
    {"document": row["Case Description"], "summary": row["Official Reasoning"]}
    for _, row in df.iterrows()
]

# Split the dataset: 80% train, 20% temp (val+test)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# Split temp into 50% val, 50% test => each 10% of total
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Create output directory if it doesn't exist
output_dir = "data/summarization"
os.makedirs(output_dir, exist_ok=True)

# Function to write jsonl
def save_jsonl(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save all splits
save_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
save_jsonl(val_data, os.path.join(output_dir, 'val.jsonl'))
save_jsonl(test_data, os.path.join(output_dir, 'test.jsonl'))

print(f"Dataset split complete and saved to '{output_dir}'")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
