from pathlib import Path

import pandas as pd
import wandb

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to draw plots. Install it with:\n"
        "  pip install matplotlib\n"
        "or:\n"
        "  conda install matplotlib"
    ) from exc

ENTITY = None
PROJECT = "gptsniffer-tta-detection"

# Keep a wider recent-run window because additional TTA runs may be logged
# after the original 34-run ablation sweep.
RECENT_RUNS = 60
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/plots"


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.8,
            "axes.linewidth": 1.2,
        }
    )


def run_path():
    return PROJECT if ENTITY is None else f"{ENTITY}/{PROJECT}"


def get_recent_runs(api):
    runs = sorted(api.runs(run_path()), key=lambda run: run.created_at, reverse=True)
    return runs[:RECENT_RUNS]


def history_df(run, keys):
    rows = list(run.scan_history(keys=["_step", *keys]))
    if not rows:
        return pd.DataFrame(columns=["_step", *keys])
    return pd.DataFrame(rows)


def first_existing_key(run, candidates):
    summary = dict(run.summary)
    history_keys = set(run.history(samples=1).columns)
    for key in candidates:
        if key in summary or key in history_keys:
            return key
    return None


def find_run(runs, name):
    for run in runs:
        if run.name == name:
            return run
    available = "\n".join(sorted(run.name for run in runs))
    raise KeyError(f"Run {name!r} not found. Available runs:\n{available}")


def plot_tent_entropy(sweep_runs):
    selected = [
        ("offline, lr=1e-6, steps=1", "aigcodeset_tent_offline_lr1em06_steps1"),
        ("offline, lr=1e-4, steps=10", "aigcodeset_tent_offline_lr0p0001_steps10"),
        ("online, lr=1e-6, steps=1", "aigcodeset_tent_online_lr1em06_steps1"),
        ("online, lr=1e-4, steps=10", "aigcodeset_tent_online_lr0p0001_steps10"),
    ]
    key_candidates = [
        "tta/tent_final_entropy_test",
        "tta/tent_entropy_loss_test",
        "tta/tent_mean_entropy_loss_test",
    ]

    plt.figure(figsize=(9, 5))
    plotted = 0
    for label, name in selected:
        run = find_run(sweep_runs, name)
        key = first_existing_key(run, key_candidates)
        if key is None:
            continue
        df = history_df(run, [key]).dropna(subset=[key])
        if df.empty:
            continue
        plt.plot(df["_step"], df[key], label=label)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No TENT entropy history was found in selected W&B runs.")

    plt.xlabel("Batch step")
    plt.ylabel("Entropy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tent_entropy_over_batches.png", dpi=200)
    plt.close()


def _plot_centroid_margin_for_scale(ax, sweep_runs, scale_name):
    selected = [
        ("cosine", f"aigcodeset_centroids_cosine_scale{scale_name}"),
        ("euclidean", f"aigcodeset_centroids_euclidean_scale{scale_name}"),
    ]
    margin_key = "tta/centroid_mean_margin_test"

    plotted = 0
    for label, name in selected:
        run = find_run(sweep_runs, name)
        key = first_existing_key(run, [margin_key])
        if key is None:
            continue
        df = history_df(run, [key]).dropna(subset=[key])
        if df.empty:
            continue
        ax.plot(df["_step"], df[key], label=label, linewidth=2.8)
        plotted += 1

    if plotted == 0:
        raise RuntimeError(f"No centroid diagnostics history was found for {scale_name}.")

    ax.set_xlabel("Batch step")
    ax.set_ylabel("Mean margin")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", width=1.2, length=5)
    ax.legend(framealpha=0.9)


def plot_centroid_diagnostics(sweep_runs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    _plot_centroid_margin_for_scale(axes[0], sweep_runs, scale_name="1p0")
    _plot_centroid_margin_for_scale(axes[1], sweep_runs, scale_name="10p0")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "centroid_margin_by_scale_over_batches.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    configure_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)
    api = wandb.Api()
    sweep_runs = get_recent_runs(api)

    print("Sweep runs:", len(sweep_runs))

    plot_tent_entropy(sweep_runs)
    plot_centroid_diagnostics(sweep_runs)

    print("Saved plots:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
