import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


LABEL_KEYS = ("labels", "label", "generated", "target")
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "romangeek/hmcorp_python_gptsniffer"


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
            "Diagnose GPTSniffer predictions: class balance, predicted positive "
            "rate, and confusion matrices on all evaluation datasets."
        )
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "gptsniffer_diagnostics.csv",
        help="Where to save diagnostics.",
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


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    labels = []
    predictions = []
    positive_probs = []

    model.eval()
    for batch in tqdm(dataloader, leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_labels = batch.pop("labels")
        outputs = model(**batch)
        probs = outputs.logits.softmax(dim=-1)
        batch_predictions = probs.argmax(dim=-1)

        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
        positive_probs.extend(probs[:, 1].cpu().tolist())

    return labels, predictions, positive_probs


def summarize(dataset_name, labels, predictions, positive_probs):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    total = len(labels)
    positive_count = sum(labels)
    predicted_positive_count = sum(predictions)

    return {
        "dataset": dataset_name,
        "num_samples": total,
        "label_positive_rate": positive_count / total,
        "label_negative_rate": (total - positive_count) / total,
        "predicted_positive_rate": predicted_positive_count / total,
        "predicted_negative_rate": (total - predicted_positive_count) / total,
        "mean_positive_probability": sum(positive_probs) / total,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


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
        tokenized = prepare_dataset(dataset, tokenizer, args.max_length)
        dataloader = DataLoader(
            tokenized,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        labels, predictions, positive_probs = collect_predictions(
            model, dataloader, device
        )
        row = summarize(dataset_name, labels, predictions, positive_probs)
        rows.append(row)

        print(
            f"{dataset_name}: "
            f"label_pos={row['label_positive_rate']:.4f} "
            f"pred_pos={row['predicted_positive_rate']:.4f} "
            f"mean_p1={row['mean_positive_probability']:.4f} "
            f"tn={row['tn']} fp={row['fp']} fn={row['fn']} tp={row['tp']}"
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(f"Saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
