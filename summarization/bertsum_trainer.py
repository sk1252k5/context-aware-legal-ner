import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW  

from sklearn.metrics import accuracy_score

class SummarizationDataset:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item['document'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        summary = self.tokenizer(
            item['summary'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': summary['input_ids'].squeeze(0)
        }

class BERTSum(nn.Module):
    def __init__(self):
        super(BERTSum, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        return logits

def train_model():
    train_dataset = SummarizationDataset('data/summarization/train.jsonl')
    val_dataset = SummarizationDataset('data/summarization/val.jsonl')

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = BERTSum()
    model = model.cuda() if torch.cuda.is_available() else model

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Training loss {loss.item():.4f}")

    model_save_path = 'summarization/bertsum_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()
