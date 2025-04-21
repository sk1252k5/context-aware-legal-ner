import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from evaluate import load as load_metric
from summarization.bertsum_model import BERTSum  # Assuming your BERTSum is saved as `bertsum_model.py`

# Dataset for Evaluation
class SummarizationDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['document'], self.data[idx]['summary']

# Generate summaries using BERTSUM (extractive style placeholder)
def generate_bertsum_summaries(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for texts, _ in dataloader:
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                logits = outputs.argmax(dim=-1)  # Get highest probability tokens

                # Decode predicted token IDs into text
                summary = tokenizer.decode(
                    logits[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                predictions.append(summary)
    return predictions

# Generate summaries using T5
def generate_t5_summaries(model, tokenizer, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for texts, _ in dataloader:
            for text in texts:
                input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                output_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                predictions.append(summary)
    return predictions

# Main Evaluation
def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = SummarizationDataset('data/summarization/test.jsonl')
    dataloader = DataLoader(test_dataset, batch_size=1)

    references = [summary for _, summary in test_dataset]

    # Load BERTSUM model
    bert_model = BERTSum()
    bert_model.load_state_dict(torch.load('summarization/bertsum_model.pth', map_location=device))
    bert_model.to(device)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load T5 model
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_model.load_state_dict(torch.load('summarization/t5_model.pth', map_location=device))
    t5_model.to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Generate Summaries
    print("Generating summaries using BERTSUM...")
    bert_predictions = generate_bertsum_summaries(bert_model, bert_tokenizer, dataloader, device)

    print("Generating summaries using T5...")
    t5_predictions = generate_t5_summaries(t5_model, t5_tokenizer, dataloader, device)

    # Evaluate ROUGE
    rouge = load_metric("rouge")

    print("\nEvaluating BERTSUM...")
    bert_scores = rouge.compute(predictions=bert_predictions, references=references, use_stemmer=True)

    print("\nEvaluating T5...")
    t5_scores = rouge.compute(predictions=t5_predictions, references=references, use_stemmer=True)

    # Display Comparison Table
    print("\n📊 Model Performance Comparison:")
    print(f"{'Model':<15} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
    print(f"{'BERTSUM':<15} {bert_scores['rouge1']:.4f}   {bert_scores['rouge2']:.4f}   {bert_scores['rougeL']:.4f}")
    print(f"{'T5':<15} {t5_scores['rouge1']:.4f}   {t5_scores['rouge2']:.4f}   {t5_scores['rougeL']:.4f}")


if __name__ == "__main__":
    evaluate_models()
