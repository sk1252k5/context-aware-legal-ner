# models/legalbert/train_legalbert.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers.training_args import TrainingArguments
from models.legalbert.utils import load_dataset, align_labels_with_tokens
from models.legalbert.ner_dataset import LegalNERDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch

label_list = [
    "O", "B-PROVISION", "I-PROVISION", "B-STATUTE", "I-STATUTE", "B-RESPONDENT",
    "I-RESPONDENT", "B-PETITIONER", "I-PETITIONER", "B-JUDGE", "I-JUDGE",
    "B-LAWYER", "I-LAWYER", "B-DATE", "I-DATE", "B-ORG", "I-ORG",
    "B-GPE", "I-GPE", "B-CASE_NUMBER", "I-CASE_NUMBER", "B-PRECEDENT",
    "I-PRECEDENT", "B-WITNESS", "I-WITNESS", "B-OTHER_PERSON", "I-OTHER_PERSON"
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=len(label_list), id2label=id2label, label2id=label2id)

# Load data
train = load_dataset("data/processed/splits/train.iob.json")
val = load_dataset("data/processed/splits/val.iob.json")

# Tokenize and align labels
# Tokenize and align labels (batch processing)
train_encodings = {"input_ids": [], "attention_mask": [], "labels": []}
val_encodings = {"input_ids": [], "attention_mask": [], "labels": []}

for ex in train:
    tokens = ex["tokens"]
    labels = ex["labels"]
    encoding, aligned_labels = align_labels_with_tokens(tokenizer, tokens, labels, label2id)
    train_encodings["input_ids"].append(encoding["input_ids"].squeeze())
    train_encodings["attention_mask"].append(encoding["attention_mask"].squeeze())
    train_encodings["labels"].append(torch.tensor(aligned_labels))

for ex in val:
    tokens = ex["tokens"]
    labels = ex["labels"]
    encoding, aligned_labels = align_labels_with_tokens(tokenizer, tokens, labels, label2id)
    val_encodings["input_ids"].append(encoding["input_ids"].squeeze())
    val_encodings["attention_mask"].append(encoding["attention_mask"].squeeze())
    val_encodings["labels"].append(torch.tensor(aligned_labels))

train_dataset = LegalNERDataset(train_encodings, train_encodings["labels"])
val_dataset = LegalNERDataset(val_encodings, val_encodings["labels"])

# Metrics
def compute_metrics(p):
    pred, labels = p
    pred = pred.argmax(axis=2)
    true_labels, true_preds = [], []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_labels.append(labels[i][j])
                true_preds.append(pred[i][j])
    p, r, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="macro")
    acc = accuracy_score(true_labels, true_preds)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}

# Training config
args = TrainingArguments(
    output_dir="./models/legalbert/checkpoints",
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save final model
model.save_pretrained("models/legalbert/final_model")
tokenizer.save_pretrained("models/legalbert/final_model")
print("LegalBERT fine-tuning complete!")
