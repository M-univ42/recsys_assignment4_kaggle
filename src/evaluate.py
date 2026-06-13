from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import InteractionData
from ease import EASE
from ensemble import (best_ease_params, best_variant_params, load_recs,
                      rrf_fuse)

FIGURES = Path("figures")
RESULTS = Path("results")
K = 10
N_CAND = 50          # depth of saved/recomputed candidate lists


def best_ensemble_cfg() -> tuple[tuple[int, ...], float, int]:
    """(weights, rrf_k, n_cand) of the tuned ensemble, read from its sweep."""
    import ast
    row = pd.read_csv(RESULTS / "ensemble_val.csv").iloc[0]
    return ast.literal_eval(row["weights"]), float(row["rrf_k"]), int(row["n_cand"])


def recall_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """Kaggle-style truncated recall: hits / min(|truth|, k)."""
    hits = len(truth.intersection(recs[:k]))
    return hits / min(len(truth), k)


def precision_at_k(recs: list[int], truth: set[int], k: int) -> float:
    return len(truth.intersection(recs[:k])) / k


def hit_rate_at_k(recs: list[int], truth: set[int], k: int) -> float:
    return 1.0 if truth.intersection(recs[:k]) else 0.0


def ndcg_at_k(recs: list[int], truth: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 2)
              for rank, item in enumerate(recs[:k]) if item in truth)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(truth), k)))
    return dcg / idcg if idcg else 0.0


def mrr_at_k(recs: list[int], truth: set[int], k: int) -> float:
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            return 1.0 / rank
    return 0.0


def map_at_k(recs: list[int], truth: set[int], k: int) -> float:
    """Average precision at k."""
    hits, score = 0, 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in truth:
            hits += 1
            score += hits / rank
    return score / min(len(truth), k) if truth else 0.0


METRICS = {
    "Recall@10": recall_at_k,
    "Precision@10": precision_at_k,
    "NDCG@10": ndcg_at_k,
    "MAP@10": map_at_k,
    "MRR@10": mrr_at_k,
    "HitRate@10": hit_rate_at_k,
}


def evaluate(recs: dict[int, list[int]], truth: dict[int, set[int]],
             users: np.ndarray, n_items: int) -> dict[str, float]:
    out = {name: float(np.mean([fn(recs[u], truth[u], K) for u in users]))
           for name, fn in METRICS.items()}
    # Catalog coverage @10: fraction of items that ever appear in a top-10.
    shown = {i for u in users for i in recs[u][:K]}
    out["Coverage@10"] = len(shown) / n_items
    return out


def recall_curve(recs: dict[int, list[int]], truth: dict[int, set[int]],
                 users: np.ndarray, ks: range) -> list[float]:
    return [float(np.mean([recall_at_k(recs[u], truth[u], k) for u in users]))
            for k in ks]


def popularity_recs(data: InteractionData, users: np.ndarray,
                    k: int) -> dict[int, list[int]]:
    """Most-interacted train items, per-user seen-masked, top-k."""
    pop = (data.train.groupby("item_idx").size()
           .sort_values(ascending=False).index.to_numpy())
    seen = data.seen_items("train")
    recs = {}
    for u in users:
        s = seen.get(u, set())
        recs[u] = [int(i) for i in pop if i not in s][:k]
    return recs


def build_models(data: InteractionData) -> tuple[dict, np.ndarray]:
    tiger = load_recs(RESULTS / "tiger_recs_val.csv")
    sasrec = load_recs(RESULTS / "sasrec_recs_val.csv")
    users = np.array(sorted(set(tiger) & set(sasrec)))

    hl, l2 = best_ease_params()
    pop_reg, edl2 = best_variant_params()
    print(f"EASE params: half_life={hl}, lambda={l2}; "
          f"EDLAE pop_reg={pop_reg}, lambda={edl2}", flush=True)
    X = data.to_csr("train", half_life_days=hl)
    ease = EASE(l2).fit(X).recommend(X, users, k=N_CAND)
    ease_edlae = EASE(edl2, pop_reg=pop_reg).fit(X).recommend(X, users, k=N_CAND)

    e_weights, e_rrf_k, e_n_cand = best_ensemble_cfg()
    print(f"ensemble cfg from sweep: weights={e_weights}, rrf_k={e_rrf_k}, "
          f"n_cand={e_n_cand}", flush=True)
    # The submitted ensemble fuses the EDLAE-EASE candidates, not plain EASE.
    ensemble = rrf_fuse([ease_edlae, tiger, sasrec], weights=e_weights,
                        k=N_CAND, rrf_k=e_rrf_k, n_cand=e_n_cand)

    models = {
        "Popularity": popularity_recs(data, users, N_CAND),
        "EASE": ease,
        "EASE-EDLAE": ease_edlae,
        "SASRec": sasrec,
        "TIGER": tiger,
        "Ensemble": ensemble,
    }
    return models, users


def plot_metric_bars(table: pd.DataFrame) -> None:
    """Grouped bars: one cluster per metric, one bar per model."""
    metrics = list(METRICS) + ["Coverage@10"]
    models = list(table.index)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    cmap = plt.get_cmap("viridis", len(models))

    fig, ax = plt.subplots(figsize=(11, 5))
    for j, model in enumerate(models):
        vals = [table.loc[model, m] for m in metrics]
        bars = ax.bar(x + j * width, vals, width, label=model, color=cmap(j))
        ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("score (validation)")
    ax.set_title("Model comparison across ranking metrics (temporal validation)")
    ax.legend(ncol=len(models), fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics_comparison.png", dpi=150)
    plt.close(fig)


def plot_recall_at_k(models: dict, truth: dict, users: np.ndarray) -> None:
    ks = range(1, N_CAND + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis", len(models))
    for j, (name, recs) in enumerate(models.items()):
        curve = recall_curve(recs, truth, users, ks)
        ax.plot(list(ks), curve, label=name, color=cmap(j), lw=2)
    ax.axvline(K, color="gray", ls="--", alpha=0.7, label=f"k={K} (Kaggle)")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (validation)")
    ax.set_title("Recall@k by model (temporal validation)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "recall_at_k.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = InteractionData("data", val_frac=0.1)
    models, users = build_models(data)
    truth = data.val_ground_truth
    print(f"evaluating {len(models)} models on {len(users)} shared val users\n",
          flush=True)

    table = pd.DataFrame(
        {name: evaluate(recs, truth, users, data.n_items)
         for name, recs in models.items()}).T
    table.index.name = "model"
    table = table.sort_values("Recall@10", ascending=False)

    table.to_csv(RESULTS / "metrics_comparison.csv", float_format="%.4f")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"), flush=True)
    best = table.index[0]
    print(f"\nbest by Recall@10 (Kaggle metric): {best} "
          f"({table.loc[best, 'Recall@10']:.4f})", flush=True)

    plot_metric_bars(table)
    plot_recall_at_k(models, truth, users)
    for f in ["metrics_comparison.png", "recall_at_k.png"]:
        print(f"wrote {FIGURES / f}", flush=True)
    print(f"wrote {RESULTS / 'metrics_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
