"""EDLAE-style EASE variant: does popularity-weighted regularization help?

Plain EASE adds a uniform ridge lambda to the Gram diagonal. EDLAE (Steck,
"Autoencoders that don't overfit towards the Identity", NeurIPS 2020) shows
that emphasized dropout-denoising is, in closed form, equivalent to an L2
penalty *proportional to each item's frequency* -- it regularizes popular items
more and thereby stops the autoencoder collapsing toward the identity. Here that
is the `pop_reg` knob on EASE: diag_reg_j = lambda + pop_reg * G_jj.

This script sweeps pop_reg at the best half-life/lambda from the EASE sweep and
compares val Recall@10 against the pop_reg=0 baseline (which must reproduce the
plain-EASE number). If a setting wins it refits on full+test and writes
submission_ease_edlae.csv; otherwise it reports the (informative) null result.

    python -u src/ease_variant.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import InteractionData
from ease import EASE
from ensemble import best_ease_params

# Joint grid: heavy popularity weighting can shift the best uniform lambda, so
# both are swept. pop_reg=0 at L2_BASE is the plain-EASE reference.
POP_REG_GRID = [0.0, 30.0, 60.0, 100.0, 200.0, 400.0, 800.0]
L2_GRID = [1000.0, 5000.0]
L2_BASE = 5000.0


def main() -> None:
    data = InteractionData("data", val_frac=0.1)
    hl, _ = best_ease_params()
    print(f"base EASE half_life from sweep: {hl}\n", flush=True)

    X = data.to_csr("train", half_life_days=hl)
    G = (X.T @ X).toarray().astype(np.float64)
    d = np.diag(G)
    print(f"Gram diagonal (item 'popularity' mass): "
          f"mean={d.mean():.1f}  median={np.median(d):.1f}  max={d.max():.1f}\n",
          flush=True)

    rows, best = [], (-1.0, None, None)
    for pr in POP_REG_GRID:
        for l2 in L2_GRID:
            model = EASE(l2, pop_reg=pr).fit_gram(G)
            recall = data.recall_at_k(model.recommend(X, data.val_users), k=10)
            rows.append({"pop_reg": pr, "l2": l2, "recall_at_10": recall})
            marker = ""
            if recall > best[0]:
                best, marker = (recall, pr, l2), "  <- best"
            tag = "  (= plain EASE)" if pr == 0 and l2 == L2_BASE else ""
            print(f"pop_reg={pr:>6.1f}  lambda={l2:>7.1f}  "
                  f"val Recall@10={recall:.4f}{tag}{marker}", flush=True)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/ease_variant_sweep.csv", index=False)

    base = next(r["recall_at_10"] for r in rows
                if r["pop_reg"] == 0.0 and r["l2"] == L2_BASE)
    best_recall, best_pr, best_l2 = best
    delta = best_recall - base
    print(f"\nbaseline (plain EASE)   val Recall@10 = {base:.4f}")
    print(f"best EDLAE pop_reg={best_pr}, lambda={best_l2}  "
          f"val Recall@10 = {best_recall:.4f}  (delta {delta:+.4f})", flush=True)

    if best_pr == 0.0 or delta <= 0:
        print("\nEDLAE-style popularity weighting did NOT improve over plain "
              "EASE on this dataset; keeping the existing submission.", flush=True)
        return

    print(f"\nimprovement found -> refitting on full+test with "
          f"pop_reg={best_pr}, lambda={best_l2}", flush=True)
    X_full = data.to_csr("full+test", half_life_days=hl)
    model = EASE(best_l2, pop_reg=best_pr).fit(X_full)
    recs = model.recommend(X_full, data.target_user_idx)
    path = data.write_submission(recs, "submission_ease_edlae.csv")
    print(f"wrote {path} (val Recall@10 was {best_recall:.4f})", flush=True)


if __name__ == "__main__":
    main()
