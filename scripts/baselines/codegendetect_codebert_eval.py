import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


LABEL_KEYS = ("labels", "label", "generated", "target")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "azherali/CodeGenDetect-CodeBert"


DATASETS = {
    "python": {
        "source": "json",
        "path": ROOT / "gptsniffer_finetuning/dataset/python/test.jsonl",
        "split": "test",
    },
    "python_no_comments": {
        "source": "json",
        "path": ROOT / "gptsniffer_finetuning/dataset/python/test_no_comment.jsonl",
        "split": "test",
    },
    "java": {
        "source": "json",
        "path": ROOT / "gptsniffer_finetuning/dataset/java/test.jsonl",
        "split": "test",
    },
    "aigcodeset": {
        "source": "hf",
        "path": "basakdemirok/AIGCodeSet",
        "split": "test",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate azherali/CodeGenDetect-CodeBert on the four target "
            "datasets used in the GPTSniffer TTA experiments."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer sequence length.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional number of samples per dataset for quick smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device used for model inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/metrics/codegendetect_codebert_metrics.csv",
        help="Where to save evaluation metrics.",
    )
    return parser.parse_args()


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def label_key(record):
    key = next((candidate for candidate in LABEL_KEYS if candidate in record), None)
    if key is None:
        raise KeyError(f"No label key from {LABEL_KEYS}. Available keys: {record.keys()}")
    return key


def load_eval_dataset(config, limit=None):
    if config["source"] == "json":
        dataset = load_dataset(
            "json",
            data_files={config["split"]: str(config["path"])},
            split=config["split"],
        )
    elif config["source"] == "hf":
        dataset = load_dataset(config["path"], split=config["split"])
    else:
        raise ValueError(f"Unknown dataset source: {config['source']}")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def prepare_dataset(dataset, tokenizer, max_length, code_key="code"):
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    first = dataset[0]
    target_key = label_key(first)
    if code_key not in first:
        raise KeyError(f"No code key '{code_key}'. Available keys: {first.keys()}")

    def tokenize(batch):
        encoded = tokenizer(
            batch[code_key],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [int(label) for label in batch[target_key]]
        return encoded

    keep_columns = [code_key, target_key]
    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=[
            column for column in dataset.column_names if column not in keep_columns
        ],
    ).remove_columns(keep_columns)


def compute_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


@torch.no_grad()
def evaluate_dataset(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []

    for batch in tqdm(dataloader, leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_labels = batch.pop("labels")
        outputs = model(**batch)
        batch_predictions = outputs.logits.argmax(dim=-1)

        predictions.extend(batch_predictions.cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

    return labels, predictions


def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(device)
    print(f"Device: {device}")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    rows = []

    for dataset_name, config in DATASETS.items():
        print(f"Evaluating {dataset_name}...")
        dataset = load_eval_dataset(config, limit=args.eval_limit)
        tokenized = prepare_dataset(dataset, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(
            tokenized,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        labels, predictions = evaluate_dataset(model, dataloader, device)
        metrics = compute_metrics(labels, predictions)

        row = {"dataset": dataset_name, "num_samples": len(labels), **metrics}
        rows.append(row)
        metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"{dataset_name}: n={len(labels)} {metric_text}")

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
