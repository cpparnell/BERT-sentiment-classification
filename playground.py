import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from typing import List

PATH_TO_MODEL = 'results/training-run_2023-11-28 22:29:14.755416/checkpoint-1500'
TEXTS = ["I hate everything about today", "Tomorrow is going to be fun", "Woah, I didn't expect that!", "What the heck!?"]

def load_model(model_path: str) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=6)
    return model

def preprocess(text: str, tokenizer: BertTokenizer, max_length: int = 512) -> torch.Tensor:
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return inputs['input_ids'], inputs['attention_mask']

def predict(model: BertForSequenceClassification, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

def classify_text(texts: List[str], model_path: str) -> List[int]:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(model_path)
    model.eval()

    predictions = []
    for text in texts:
        input_ids, attention_mask = preprocess(text, tokenizer)
        prediction = predict(model, input_ids, attention_mask)
        predictions.append(prediction)

    _map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    def label(val):
        return _map[val]

    return list(map(label, predictions))

# Example usage
sentiment_predictions = classify_text(TEXTS, PATH_TO_MODEL)
print(sentiment_predictions)
