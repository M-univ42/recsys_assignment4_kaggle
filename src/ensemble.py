
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import InteractionData
from ease import EASE

EASE_L2 = 5000.0   # tuned in src/ease.py
RRF_K = 60         # standard RRF damping constant
N_CAND = 50        # candidates taken from each model


def load_recs(path: str | Path) -> dict[int, list[int]]:
    df = pd.read_csv(path)
    return {int(r.user_idx): [int(x) for x in r.items.split(",")]
            for r in df.itertuples()}


def rrf_fuse(rankings: list[dict[int, list[int]]],
             k: int = 10) -> dict[int, list[int]]:
    users = set.intersection(*(set(r) for r in rankings))
    fused = {}
    for u in users:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking[u][:N_CAND]):
                scores[item] = scores.get(item, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused[u] = sorted(scores, key=lambda i: -scores[i])[:k]
    return fused


def main() -> None:
    data = InteractionData("data", val_frac=0.1)

    tiger_val = load_recs("results/tiger_recs_val.csv")
    users = np.array(sorted(tiger_val))
    X = data.to_csr("train")
    ease_val = EASE(EASE_L2).fit(X).recommend(X, users, k=N_CAND)
    fused_val = rrf_fuse([ease_val, tiger_val])
    for name, recs in [("EASE", ease_val), ("TIGER", tiger_val),
                       ("RRF ensemble", fused_val)]:
        r = data.recall_at_k(recs, k=10, users=users)
        print(f"{name:>14}  val Recall@10 = {r:.4f}", flush=True)

    Path("results").mkdir(exist_ok=True)
    recall = data.recall_at_k(fused_val, k=10, users=users)
    pd.DataFrame([{"model": "EASE+TIGER", "recall_at_10": recall}]).to_csv(
        "results/ensemble_val.csv", index=False)

    tiger_target = load_recs("results/tiger_recs_target.csv")
    X_full = data.to_csr("full")
    ease_target = EASE(EASE_L2).fit(X_full).recommend(
        X_full, data.target_user_idx, k=N_CAND)
    fused_target = rrf_fuse([ease_target, tiger_target])
    path = data.write_submission(fused_target, "submission_ensemble.csv")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
