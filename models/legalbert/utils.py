import json
from transformers import AutoTokenizer

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def align_labels_with_tokens(tokenizer, tokens, labels, label2id):
    tokenized_input = tokenizer(tokens,
                                is_split_into_words=True,
                                truncation=True,
                                padding='max_length',
                                max_length=128,
                                return_tensors="pt")
    
    word_ids = tokenized_input.word_ids(batch_index=0)
    previous_word_idx = None
    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id.get(labels[word_idx], 0))
        else:
            aligned_labels.append(label2id.get(labels[word_idx], 0))  # or use I-tag if needed
        previous_word_idx = word_idx
    return tokenized_input, aligned_labels
