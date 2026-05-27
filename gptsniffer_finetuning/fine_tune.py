import torch
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
    
wandb.init(project="codebert-tta-detect", name="hmcorp-python-baseline")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_ID = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
model.to(device)

print("Loading datasets...")
data_files = {
    "train": "dataset/python/train.jsonl",
    "validation": "dataset/python/valid.jsonl",
    "test": "dataset/python/test.jsonl"
}
full_dataset = load_dataset("json", data_files=data_files)
target_col_candidates = ["label", "generated", "target"]
for col in target_col_candidates:
    if col in full_dataset["train"].column_names:
        full_dataset = full_dataset.rename_column(col, "labels")
        break 
train_dataset = full_dataset["train"]
eval_dataset = full_dataset["validation"]
test_dataset = full_dataset["test"]
print(f"Train size: {len(train_dataset)}")

def gptsniffer_tokenize_function(examples):
    return tokenizer(
        examples["code"], 
        padding="max_length", 
        max_length=512, 
        truncation=True
    )

print("Tokenizing...")
tokenized_train = train_dataset.map(gptsniffer_tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(gptsniffer_tokenize_function, batched=True)
tokenized_test = test_dataset.map(gptsniffer_tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results_gptsniffer',
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    optim='adamw_torch',
    num_train_epochs=2, 
    
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,
    logging_steps=50,
    seed=42,
    
    report_to="wandb",
    run_name="hmcorp-python-baseline"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

print("Training...")
trainer.train()

print("Running evaluation on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")

print("Test set metrics:")
for key, value in test_results.items():
    if any(m in key for m in ["accuracy", "f1", "precision", "recall", "loss"]):
        print(f"{key}: {value:.4f}")

wandb.finish()

OUTPUT_DIR = "./gptsniffer_hmcorp_baseline"
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Finished fine-tuning. Weights saved to {OUTPUT_DIR}")