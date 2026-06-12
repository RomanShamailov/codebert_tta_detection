from pathlib import Path

import pandas as pd
import wandb


ENTITY = None
PROJECT = "gptsniffer-tta-detection"
NUM_RUNS = 4
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/metrics/wandb_t2a_metrics.csv"

METRIC_KEYS = {
    "accuracy": "final_metrics/accuracy_test",
    "precision": "final_metrics/precision_test",
    "recall": "final_metrics/recall_test",
    "f1": "final_metrics/f1_test",
}
METRIC_PREFIX_CANDIDATES = ("final_metrics", "metrics")


def parse_run_name(name):
    suffix = "_t2a"
    if not name.endswith(suffix):
        return name, None
    return name[: -len(suffix)], "t2a"


def extract_metrics(summary):
    metrics = {}
    for metric_name, wandb_key in METRIC_KEYS.items():
        value = summary.get(wandb_key)
        if value is None:
            for prefix in METRIC_PREFIX_CANDIDATES:
                value = summary.get(f"{prefix}/{metric_name}_test")
                if value is not None:
                    break
        metrics[metric_name] = value
    return metrics


api = wandb.Api()
path = PROJECT if ENTITY is None else f"{ENTITY}/{PROJECT}"
runs = sorted(api.runs(path), key=lambda run: run.created_at, reverse=True)[:NUM_RUNS]

rows = []
skipped = []
for run in runs:
    summary = dict(run.summary)
    dataset, method = parse_run_name(run.name)
    metrics = extract_metrics(summary)

    if method is None or all(value is None for value in metrics.values()):
        skipped.append(
            {
                "run_name": run.name,
                "parsed_method": method,
                "metric_keys": [
                    key
                    for key in sorted(summary.keys())
                    if any(
                        metric in key.lower()
                        for metric in ("accuracy", "precision", "recall", "f1")
                    )
                ],
            }
        )
        continue

    rows.append(
        {
            "run_name": run.name,
            "created_at": run.created_at,
            "state": run.state,
            "dataset": dataset,
            "method": method,
            **metrics,
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    print(f"No T^2 A metric rows found in last {NUM_RUNS} W&B runs.")
    print("Skipped runs:")
    for item in skipped:
        print(item)
    raise SystemExit(1)

dataset_order = {
    "python": 0,
    "python_no_comment": 1,
    "java": 2,
    "aigcodeset": 3,
}
df["dataset_order"] = df["dataset"].map(dataset_order).fillna(len(dataset_order))
df = df.sort_values(["dataset_order", "created_at"]).drop(columns=["dataset_order"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"Saved {len(df)} rows to {OUTPUT}")
print(df)
