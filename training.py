# Fine-Tuning a BERT Model to Perform Sentiment Analysis on Text using MPS on Apple M-Series Silicon

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datetime import datetime
import torch

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

transformers.logging.set_verbosity_info()

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS!')
else:
    device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    print('Using CPU!')

# Load dataset
dataset = load_dataset('dair-ai/emotion')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function that moves data to MPS device
def tokenize(e):
    return tokenizer(e['text'], padding='max_length', truncation=True, max_length=128)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model and move it to MPS device
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=f"./results/optimized-training-run_{datetime.now()}",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Reduced batch size for memory efficiency
    per_device_eval_batch_size=32,  # Reduced eval batch size
    gradient_accumulation_steps=4,  # Increased gradient accumulation for effective larger batch processing
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Data collator for dynamic padding
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()