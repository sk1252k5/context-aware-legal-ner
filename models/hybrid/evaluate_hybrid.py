import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, accuracy_score
from models.bilstm_crf.model import BiLSTM_CRF
from models.bilstm_crf.utils import build_vocab, encode_sentences
from models.legalbert.utils import align_labels_with_tokens
from sklearn.preprocessing import LabelBinarizer
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# ------------ Load Test Data --------------
with open("data/processed/splits/test.iob.json", "r") as f:
    test_data = json.load(f)

test_tokens = [sample["tokens"] for sample in test_data]
test_labels = [sample["labels"] for sample in test_data]

# ------------ Load BiLSTM-CRF --------------
with open("models/bilstm_crf/word2idx.json", "r") as f:
    word2idx = json.load(f)
with open("models/bilstm_crf/tag2idx.json", "r") as f:
    tag2idx = json.load(f)
idx2tag = {v: k for k, v in tag2idx.items()}

X_test = encode_sentences(test_tokens, word2idx)
y_test = encode_sentences(test_labels, tag2idx)

bilstm_model = BiLSTM_CRF(len(word2idx), len(tag2idx))
bilstm_model.load_state_dict(torch.load("models/bilstm_crf/bilstm_crf.pt"))
bilstm_model.eval()

bilstm_preds = []
with torch.no_grad():
    for sentence in X_test:
        output = bilstm_model.decode(torch.tensor(sentence).unsqueeze(0))
        bilstm_preds.append([idx2tag[i] for i in output[0]])

# ------------ Load LegalBERT --------------
label_list = list(tag2idx.keys())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
bert_model = AutoModelForTokenClassification.from_pretrained("models/legalbert/final_model")
bert_model.eval()

legalbert_preds = []
with torch.no_grad():
    for tokens, labels in zip(test_tokens, test_labels):
        tokenized_input, aligned_labels = align_labels_with_tokens(tokenizer, tokens, labels, label2id)
        output = bert_model(**tokenized_input)
        logits = output.logits
        pred_ids = torch.argmax(logits, dim=2).squeeze().tolist()

        word_ids = tokenized_input.word_ids(batch_index=0)
        prev_word_idx = None
        final_preds = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != prev_word_idx:
                final_preds.append(id2label[pred_ids[idx]])
                prev_word_idx = word_idx

        # Pad if mismatch occurs
        while len(final_preds) < len(labels):
            final_preds.append("O")
        legalbert_preds.append(final_preds)


# ------------ Hybrid Prediction (Simple Voting) --------------
from models.hybrid.rule_features import apply_rules, improve_predictions

rule_preds = [apply_rules(tokens) for tokens in test_tokens]
hybrid_preds = improve_predictions(bilstm_preds, legalbert_preds, rule_preds)

# ------------ Evaluation -------------------
def flatten(lst):
    return [item for sublist in lst for item in sublist]

y_true = flatten(test_labels)
y_pred_bilstm = flatten(bilstm_preds)
y_pred_bert = flatten(legalbert_preds)
y_pred_hybrid = flatten(hybrid_preds)

report_bilstm = classification_report(y_true, y_pred_bilstm, output_dict=True)
report_bert = classification_report(y_true, y_pred_bert, output_dict=True)
report_hybrid = classification_report(y_true, y_pred_hybrid, output_dict=True)

with open("results/classification_report_bilstm_crf.json", "w") as f:
    json.dump(report_bilstm, f, indent=2)
with open("results/classification_report_legalbert.json", "w") as f:
    json.dump(report_bert, f, indent=2)
with open("results/classification_report_hybrid.json", "w") as f:
    json.dump(report_hybrid, f, indent=2)

# Confusion Matrix - Hybrid
labels = list(set(y_true))
cm = confusion_matrix(y_true, y_pred_hybrid, labels=labels)
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
fig.colorbar(cax)
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Confusion Matrix - Hybrid Model")
plt.tight_layout()
plt.savefig("results/confusion_matrix_hybrid.png")

# Accuracy comparison
accs = [accuracy_score(y_true, y_pred_bilstm), accuracy_score(y_true, y_pred_bert), accuracy_score(y_true, y_pred_hybrid)]
models = ["Hybrid", "LegalBERT", "BiLSTM-CRF"]
plt.figure(figsize=(8, 5))
plt.bar(models, accs, color=["skyblue", "salmon", "limegreen"])
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Models")
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png")

# Precision-Recall Curve - Hybrid
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(y_true)
y_pred_bin = lb.transform(y_pred_hybrid)

precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="Hybrid")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Hybrid")
plt.grid()
plt.tight_layout()
plt.savefig("results/precision_recall_curve_hybrid.png")

print("\n Hybrid model evaluated successfully. All evaluation results saved to /results")

