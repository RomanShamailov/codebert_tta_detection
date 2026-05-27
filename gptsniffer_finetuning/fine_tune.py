import torch
# wandb.init() пока закомментируем, чтобы не мусорить в дашборде
# import wandb 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
    
# wandb.init(project="codebert-tta-detect", name="hmcorp-python-baseline-test")

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

def gptsniffer_tokenize_function(examples):
    return tokenizer(
        examples["code"], 
        max_length=512, 
        truncation=True
    )

print("Tokenizing...")
tokenized_train = train_dataset.map(gptsniffer_tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(gptsniffer_tokenize_function, batched=True)
tokenized_test = test_dataset.map(gptsniffer_tokenize_function, batched=True)

# === СУЖАЕМ ВЫБОРКУ ДО 1 БАТЧА ДЛЯ A100 (128 примеров) ===
print("Selecting 128 examples for One Batch Test...")
tokenized_train = tokenized_train.select(range(128))
tokenized_eval = tokenized_eval.select(range(128))
tokenized_test = tokenized_test.select(range(128))
# =========================================================

training_args = TrainingArguments(
    output_dir='./results_gptsniffer_test',
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=0,          # Убираем разогрев для теста
    weight_decay=0.01,
    optim='adamw_torch',
    
    num_train_epochs=3,      # Прогоним батч 3 раза, чтобы проверить сохранение
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    
    bf16=True,               # Та самая магия A100
    dataloader_num_workers=4,
    logging_steps=1,         # Печатаем лог на каждом шаге
    seed=42,
    
    report_to="none",        # Отключаем W&B для теста
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("Training...")
trainer.train()

print("Running evaluation on test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")

print("Test set metrics:")
for key, value in test_results.items():
    if any(m in key for m in ["accuracy", "f1", "precision", "recall", "loss"]):
        print(f"{key}: {value:.4f}")

# wandb.finish()

OUTPUT_DIR = "./gptsniffer_hmcorp_baseline_test"
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Finished testing. Dummy weights saved to {OUTPUT_DIR}")