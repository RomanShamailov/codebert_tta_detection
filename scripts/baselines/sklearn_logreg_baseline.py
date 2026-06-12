import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


LABEL_KEYS = ("labels", "label", "generated", "target")
ROOT = Path(__file__).resolve().parents[2]


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
            "Train a simple sklearn TF-IDF + LogisticRegression baseline on "
            "HMCorp Python train and evaluate it on the four target datasets."
        )
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=ROOT / "gptsniffer_finetuning/dataset/python/train.jsonl",
        help="Local JSONL file used to train the logistic regression baseline.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional number of train samples for quick smoke tests.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional number of samples per eval dataset.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=200_000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum character n-gram size.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=5,
        help="Maximum character n-gram size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/metrics/sklearn_logreg_metrics.csv",
        help="Where to save evaluation metrics.",
    )
    return parser.parse_args()


def label_key(record):
    key = next((candidate for candidate in LABEL_KEYS if candidate in record), None)
    if key is None:
        raise KeyError(f"No label key from {LABEL_KEYS}. Available keys: {record.keys()}")
    return key


def load_json_dataset(path, split, limit=None):
    dataset = load_dataset("json", data_files={split: str(path)}, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_hf_dataset(name, split, limit=None):
    dataset = load_dataset(name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def extract_xy(dataset, code_key="code"):
    if len(dataset) == 0:
        return [], []

    first = dataset[0]
    target_key = label_key(first)
    if code_key not in first:
        raise KeyError(f"No code key '{code_key}'. Available keys: {first.keys()}")

    texts = []
    labels = []
    for record in dataset:
        texts.append(record[code_key])
        labels.append(int(record[target_key]))
    return texts, labels


def build_model(max_features, ngram_min, ngram_max):
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(ngram_min, ngram_max),
                    max_features=max_features,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    class_weight=None,
                    random_state=42,
                    verbose=0,
                ),
            ),
        ]
    )


def compute_metrics(labels, predictions):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main():
    args = parse_args()

    print(f"Loading train data from {args.train_path}")
    train_dataset = load_json_dataset(args.train_path, split="train", limit=args.train_limit)
    train_texts, train_labels = extract_xy(train_dataset)
    print(f"Train size: {len(train_texts)}")

    model = build_model(
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )

    print("Training TF-IDF + LogisticRegression...")
    model.fit(train_texts, train_labels)

    rows = []
    for dataset_name, cfg in DATASETS.items():
        print(f"Evaluating {dataset_name}...")
        if cfg["source"] == "json":
            dataset = load_json_dataset(cfg["path"], cfg["split"], limit=args.eval_limit)
        elif cfg["source"] == "hf":
            dataset = load_hf_dataset(cfg["path"], cfg["split"], limit=args.eval_limit)
        else:
            raise ValueError(f"Unknown dataset source: {cfg['source']}")

        texts, labels = extract_xy(dataset)
        predictions = model.predict(texts)
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
