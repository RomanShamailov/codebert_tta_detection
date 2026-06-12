import re
from pathlib import Path

import pandas as pd
import wandb

ENTITY = None
PROJECT = "gptsniffer-tta-detection"
NUM_RUNS = 34
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/metrics/wandb_hyperparameter_metrics.csv"

METRIC_KEYS = {
    "accuracy": "final_metrics/accuracy_test",
    "precision": "final_metrics/precision_test",
    "recall": "final_metrics/recall_test",
    "f1": "final_metrics/f1_test",
}
METRIC_PREFIX_CANDIDATES = ("final_metrics", "metrics")

TENT_PATTERN = re.compile(
    r"^aigcodeset_tent_(?P<mode>online|offline)_"
    r"lr(?P<lr>.+)_steps(?P<steps>\d+)$"
)
CENTROIDS_PATTERN = re.compile(
    r"^aigcodeset_centroids_(?P<distance>cosine|euclidean)_"
    r"scale(?P<logit_scale>.+)$"
)


def decode_float(value):
    return float(value.replace("m", "-").replace("p", "."))


def parse_run_name(name):
    tent_match = TENT_PATTERN.match(name)
    if tent_match is not None:
        return {
            "dataset": "aigcodeset",
            "method": "tent",
            "tent_mode": tent_match.group("mode"),
            "tent_lr": decode_float(tent_match.group("lr")),
            "tent_steps": int(tent_match.group("steps")),
            "centroid_distance": None,
            "centroid_logit_scale": None,
        }

    centroids_match = CENTROIDS_PATTERN.match(name)
    if centroids_match is not None:
        return {
            "dataset": "aigcodeset",
            "method": "centroids",
            "tent_mode": None,
            "tent_lr": None,
            "tent_steps": None,
            "centroid_distance": centroids_match.group("distance"),
            "centroid_logit_scale": decode_float(
                centroids_match.group("logit_scale")
            ),
        }

    return None


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
    parsed = parse_run_name(run.name)
    summary = dict(run.summary)
    metrics = extract_metrics(summary)

    if parsed is None or all(value is None for value in metrics.values()):
        skipped.append(
            {
                "run_name": run.name,
                "parsed": parsed,
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
            **parsed,
            **metrics,
        }
    )

df = pd.DataFrame(rows)
if df.empty:
    print(f"No matching hyperparameter rows found in last {NUM_RUNS} W&B runs.")
    print("First skipped runs:")
    for item in skipped[:10]:
        print(item)
    raise SystemExit(1)

df = df.sort_values(
    [
        "method",
        "tent_mode",
        "tent_lr",
        "tent_steps",
        "centroid_distance",
        "centroid_logit_scale",
    ],
    na_position="last",
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} rows to {OUTPUT}")
print(df)
