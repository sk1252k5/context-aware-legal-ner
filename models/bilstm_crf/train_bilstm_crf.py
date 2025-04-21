import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from models.bilstm_crf.model import BiLSTM_CRF
from models.bilstm_crf.utils import build_vocab, encode_sentences, NERDataset
from torch.nn.utils.rnn import pad_sequence
import pickle, json, os

# Load your preprocessed IOB-tagged data
with open('data/processed/splits/train.iob.json') as f:
    train_data = json.load(f)

with open('data/processed/splits/val.iob.json') as f:
    val_data = json.load(f)

# Extract tokens and labels
train_sentences = [sample["tokens"] for sample in train_data]
train_labels = [sample["labels"] for sample in train_data]

val_sentences = [sample["tokens"] for sample in val_data]
val_labels = [sample["labels"] for sample in val_data]

# Build vocab and tagset
word2idx = build_vocab(train_sentences)
tag2idx = build_vocab(train_labels)
idx2tag = {v: k for k, v in tag2idx.items()}

# Save vocab and tag mappings
os.makedirs("models/bilstm_crf", exist_ok=True)
with open("models/bilstm_crf/vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)
with open("models/bilstm_crf/tag2idx.pkl", "wb") as f:
    pickle.dump(tag2idx, f)

# Encode
X_train = encode_sentences(train_sentences, word2idx)
y_train = encode_sentences(train_labels, tag2idx)

X_val = encode_sentences(val_sentences, word2idx)
y_val = encode_sentences(val_labels, tag2idx)

# Padding function
def pad(batch):
    tokens, labels = zip(*batch)
    tokens = [torch.tensor(seq, dtype=torch.long) for seq in tokens]
    labels = [torch.tensor(seq, dtype=torch.long) for seq in labels]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    mask = tokens != 0
    return tokens, labels, mask

# Create dataset and dataloader
train_dataset = NERDataset({"input_ids": X_train}, y_train)
val_dataset = NERDataset({"input_ids": X_val}, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=pad)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=pad)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_CRF(len(word2idx), len(tag2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        loss = model(x, y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

with open("models/bilstm_crf/word2idx.json", "w") as f:
    json.dump(word2idx, f)

with open("models/bilstm_crf/tag2idx.json", "w") as f:
    json.dump(tag2idx, f)
# Save model
torch.save(model.state_dict(), "models/bilstm_crf/bilstm_crf.pt")
print("BiLSTM-CRF training complete and model saved!")


