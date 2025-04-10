import json
from torch.utils.data import Dataset

def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode_sentences(sentences, vocab):
    encoded = []
    for sentence in sentences:
        encoded.append([vocab.get(word, vocab['<UNK>']) for word in sentence])
    return encoded

class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X["input_ids"][idx], self.y[idx]
