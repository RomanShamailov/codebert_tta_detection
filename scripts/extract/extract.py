from pathlib import Path

import pandas as pd
import wandb

ENTITY = None  # если нужен entity/team, впиши строку
PROJECT = "gptsniffer-tta-detection"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/metrics/wandb_metrics.csv"
METHOD_ALIASES = {
    "tent_accumulation": "tent_accumulation",
    "tent_accum": "tent_accumulation",
    "tent_reset": "tent",
    "centroids": "centroids",
    "tent": "tent",
    "none": "none",
}
METRIC_KEYS = {
    "accuracy": "final_metrics/accuracy_test",
    "precision": "final_metrics/precision_test",
    "recall": "final_metrics/recall_test",
    "f1": "final_metrics/f1_test",
}
METRIC_PREFIX_CANDIDATES = ("final_metrics", "metrics")


def parse_run_name(name):
    for method, normalized_method in METHOD_ALIASES.items():
        suffix = f"_{method}"
        if name.endswith(suffix):
            return name[: -len(suffix)], normalized_method
    return name, None

api = wandb.Api()
path = PROJECT if ENTITY is None else f"{ENTITY}/{PROJECT}"

rows = []
runs_seen = 0
skipped = []
runs = sorted(api.runs(path), key=lambda run: run.created_at, reverse=True)[:16]
for run in runs:
    runs_seen += 1
    summary = dict(run.summary)
    dataset, method = parse_run_name(run.name)
    metrics = {}
    for metric_name, wandb_key in METRIC_KEYS.items():
        value = summary.get(wandb_key)
        if value is None:
            for prefix in METRIC_PREFIX_CANDIDATES:
                value = summary.get(f"{prefix}/{metric_name}_test")
                if value is not None:
                    break
        metrics[metric_name] = value

    if method is None or all(value is None for value in metrics.values()):
        skipped.append(
            {
                "run_name": run.name,
                "parsed_method": method,
                "summary_keys": sorted(summary.keys()),
            }
        )
        continue

    rows.append({
        "run_name": run.name,
        "created_at": run.created_at,
        "state": run.state,
        "dataset": dataset,
        "method": method,
        **metrics,
    })

df = pd.DataFrame(rows)
if df.empty:
    print(f"No matching metric rows found in {runs_seen} W&B runs.")
    print("First skipped runs:")
    for item in skipped[:10]:
        metric_like_keys = [
            key
            for key in item["summary_keys"]
            if any(name in key.lower() for name in ("accuracy", "precision", "recall", "f1"))
        ]
        print(
            {
                "run_name": item["run_name"],
                "parsed_method": item["parsed_method"],
                "metric_like_keys": metric_like_keys,
            }
        )
    raise SystemExit(1)

df = df.sort_values(["dataset", "method"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} rows to {OUTPUT}")
print("Printing...")
print(df)
