
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import InteractionData

FIGURES = Path("figures")
RESULTS = Path("results")

POPULARITY_RECALL = 0.0095  # top-10 popular unseen items, same val split


def plot_ease_sweep() -> None:
    """Recall vs recency half-life, one line per λ (λ is nearly flat)."""
    sweep = pd.read_csv(RESULTS / "ease_sweep.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    # plot unweighted (NaN half-life) as a reference line
    unw = sweep[sweep["half_life"].isna()]
    if len(unw):
        ax.axhline(unw["recall_at_10"].max(), color="tab:blue", ls=":",
                   label=f"unweighted ({unw['recall_at_10'].max():.4f})")
    weighted = sweep.dropna(subset=["half_life"])
    for l2, grp in weighted.groupby("l2"):
        grp = grp.sort_values("half_life")
        ax.plot(grp["half_life"], grp["recall_at_10"], "o-",
                label=f"λ={l2:g}", alpha=0.8)
    ax.axhline(POPULARITY_RECALL, color="gray", ls="--",
               label=f"popularity ({POPULARITY_RECALL:.4f})")
    best = sweep.loc[sweep["recall_at_10"].idxmax()]
    ax.annotate(f"best: half-life={best.half_life:g}d, λ={best.l2:g}\n"
                f"R@10={best.recall_at_10:.4f}",
                xy=(best.half_life, best.recall_at_10),
                xytext=(0.45, 0.25), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "black"})
    ax.set_xlabel("recency half-life (days)")
    ax.set_ylabel("Validation Recall@10")
    ax.set_title("EASE: recency weighting sweep (temporal validation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ease_sweep.png", dpi=150)
    plt.close(fig)


def plot_interactions_over_time(data: InteractionData) -> None:
    test = pd.read_csv(data.data_dir / "test.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for df, label, color in [(data.interactions, "train", "tab:blue"),
                             (test, "test period", "tab:orange")]:
        ts = pd.to_datetime(df["timestamp"], unit="ms")
        ax.hist(ts, bins=100, alpha=0.7, label=label, color=color)
    ax.axvline(pd.to_datetime(data.val_cutoff, unit="ms"), color="red",
               ls="--", label="validation cutoff")
    ax.set_yscale("log")
    ax.set_xlabel("date")
    ax.set_ylabel("interactions per bin (log)")
    ax.set_title("Interactions over time: temporal train/test boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "interactions_over_time.png", dpi=150)
    plt.close(fig)


def plot_sparsity(data: InteractionData) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    hist_len = data.interactions.groupby("user_idx").size()
    bins = np.arange(1, 51)
    ax1.hist(hist_len.clip(upper=50), bins=bins, color="tab:blue")
    ax1.axvline(hist_len.median(), color="red", ls="--",
                label=f"median = {hist_len.median():.0f}")
    ax1.set_yscale("log")
    ax1.set_xlabel("interactions per user (clipped at 50)")
    ax1.set_ylabel("users (log)")
    ax1.set_title("User history length")
    ax1.legend()

    pop = np.sort(data.interactions.groupby("item_idx").size().to_numpy())[::-1]
    ax2.loglog(np.arange(1, len(pop) + 1), pop, color="tab:orange")
    top1 = pop[: max(1, len(pop) // 100)].sum() / pop.sum()
    ax2.set_xlabel("item rank (log)")
    ax2.set_ylabel("interaction count (log)")
    ax2.set_title(f"Item popularity (top 1% = {top1:.0%} of volume)")

    fig.tight_layout()
    fig.savefig(FIGURES / "sparsity.png", dpi=150)
    plt.close(fig)


def plot_model_comparison() -> None:
    """Bar chart of best validation Recall@10 per model.

    Extend `rows` (or write more results/*.csv files) as models are added.
    """
    rows = [("popularity", POPULARITY_RECALL)]
    sweep_path = RESULTS / "ease_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        rows.append(("EASE", sweep["recall_at_10"].max()))
    for path in sorted(RESULTS.glob("*_val.csv")):
        res = pd.read_csv(path)
        if "model" in res.columns:  # skip candidate-list files like *_recs_val
            rows.append((res["model"].iloc[0], res["recall_at_10"].iloc[0]))

    names, scores = zip(*rows)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["gray", "tab:blue", "tab:orange", "tab:green"]
    bars = ax.bar(names, scores, color=colors[: len(rows)])
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylabel("Validation Recall@10")
    ax.set_title("Model comparison (temporal validation)")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    data = InteractionData("data", val_frac=0.1)
    plot_ease_sweep()
    plot_interactions_over_time(data)
    plot_sparsity(data)
    plot_model_comparison()
    for f in sorted(FIGURES.glob("*.png")):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
