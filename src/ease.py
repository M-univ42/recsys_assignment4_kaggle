from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp

from data import InteractionData


class EASE:
    def __init__(self, l2: float = 100.0, pop_reg: float = 0.0):
        self.l2 = l2
        # EDLAE-style popularity-weighted ridge (Steck 2020): the diagonal
        # penalty becomes l2 + pop_reg * G_jj, so frequent items are regularized
        # more -- the closed-form effect of emphasized dropout-denoising, which
        # curbs EASE's overfitting toward the identity. pop_reg=0 is plain EASE.
        self.pop_reg = pop_reg
        self.B: np.ndarray | None = None  # item x item weight matrix

    def fit(self, X: sp.csr_matrix) -> "EASE":
        return self.fit_gram((X.T @ X).toarray().astype(np.float64))

    def fit_gram(self, G: np.ndarray) -> "EASE":
        Gl = G.copy()
        Gl[np.diag_indices_from(Gl)] += self.l2 + self.pop_reg * np.diag(G)
        P = scipy.linalg.inv(Gl, overwrite_a=True, check_finite=False)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def recommend(self, X: sp.csr_matrix, users: np.ndarray, k: int = 10,
                  mask_seen: bool = True,
                  batch_size: int = 4096) -> dict[int, list[int]]:
        assert self.B is not None, "call fit() first"
        recs: dict[int, list[int]] = {}
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            Xb = X[batch]
            scores = Xb @ self.B
            if mask_seen:
                scores[Xb.toarray() > 0] = -np.inf
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            recs.update(zip(batch.tolist(), top.tolist()))
        return recs


def main() -> None:
    import time

    data = InteractionData("data", val_frac=0.1)

    # Joint sweep: recency half-life (None = unweighted) x L2. Recall is nearly
    # flat in lambda (the Gram is well-conditioned here), so the grid refines the
    # recency half-life -- the parameter that actually moves the metric -- around
    # its peak and keeps a small lambda pair as flatness evidence. Each (hl, l2)
    # is one ~35s matrix inversion; the Gram is reused across lambda.
    best, best_recall = None, -1.0
    sweep = []
    for hl in [None, 365.0, 730.0, 1000.0, 1200.0, 1400.0, 1600.0, 2000.0]:
        X = data.to_csr("train", half_life_days=hl)
        G = (X.T @ X).toarray().astype(np.float64)
        for l2 in [2000.0, 5000.0]:
            t0 = time.time()
            model = EASE(l2).fit_gram(G)
            recall = data.recall_at_k(model.recommend(X, data.val_users), k=10)
            sweep.append((hl, l2, recall))
            marker = ""
            if recall > best_recall:
                best, best_recall, marker = (hl, l2), recall, "  <- best"
            print(f"half_life={hl or 'inf':>6}  lambda={l2:>8.1f}"
                  f"  val Recall@10={recall:.4f}"
                  f"  ({time.time() - t0:.0f}s){marker}", flush=True)
        del G

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(sweep, columns=["half_life", "l2", "recall_at_10"]).to_csv(
        "results/ease_sweep.csv", index=False)

    hl, l2 = best
    print(f"\nrefitting on full+test with half_life={hl}, lambda={l2}",
          flush=True)
    X_full = data.to_csr("full+test", half_life_days=hl)
    model = EASE(l2).fit(X_full)
    recs = model.recommend(X_full, data.target_user_idx)
    path = data.write_submission(recs, "submission_ease.csv")
    print(f"wrote {path} (val Recall@10 was {best_recall:.4f})")


if __name__ == "__main__":
    main()
