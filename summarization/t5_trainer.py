import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW  

# Custom Dataset
class SummarizationDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = "summarize: " + self.data[idx]['document']
        summary_text = self.data[idx]['summary']

        input_encodings = self.tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        target_encodings = self.tokenizer(summary_text, return_tensors='pt', max_length=150, truncation=True, padding='max_length')

        return {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': target_encodings['input_ids'].squeeze()
        }

# Training function
def train_model():
    dataset = SummarizationDataset('data/summarization/train.jsonl')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Training Loss {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), 'summarization/t5_model.pth')
    print("T5 Model saved to summarization/t5_model.pth")

if __name__ == "__main__":
    train_model()
