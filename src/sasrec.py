from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import InteractionData
from tiger import save_recs

SEED = 42
PAD = 0               # item tokens are item_idx + 1
MAX_LEN = 50          # history truncation (median history is 5)
D_MODEL, N_HEAD, N_LAYERS, FFN, DROPOUT = 128, 4, 2, 256, 0.2
EPOCHS, BATCH, LR = 24, 128, 1e-3
VAL_SAMPLE = 2000     # sampled-user score kept comparable with TIGER's
EVAL_EVERY = 2        # probes are cheap here (one forward, no beam search)
EVAL_SAMPLE = 500
USER_GROUP = 256      # users per inference batch


class SASRec(nn.Module):
    def __init__(self, n_items: int):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, FFN, dropout=DROPOUT, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        # small init keeps tied-embedding logits near zero at start
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[PAD].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        h = self.item(x) + self.pos(torch.arange(L, device=x.device))
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.encoder(h, mask=mask, src_key_padding_mask=(x == PAD))
        return self.norm(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item.weight.T  # tied embeddings


def token_seqs(sequences: dict[int, list[int]]) -> dict[int, list[int]]:
    """item_idx -> token (idx+1), truncated to the last MAX_LEN items."""
    return {u: [i + 1 for i in items[-MAX_LEN:]]
            for u, items in sequences.items()}


def train_model(model: SASRec, seqs: list[list[int]], n_epochs: int,
                eval_fn=None) -> int:
    """Next-item CE training; checkpoint on probe score. Returns best epoch."""
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    by_len = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches = [by_len[s:s + BATCH] for s in range(0, len(by_len), BATCH)]
    best_score, best_state, best_epoch = -1.0, None, n_epochs
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(batches))
        total, count, t0 = 0.0, 0, time.time()
        for bi in order:
            batch = [seqs[i] for i in batches[bi]]
            L = max(len(s) for s in batch)
            x = torch.full((len(batch), L), PAD, dtype=torch.long)
            for r, s in enumerate(batch):
                x[r, :len(s)] = torch.tensor(s)
            logits = model.logits(model(x[:, :-1]))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   x[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            count += len(batch)
        sched.step()
        msg = (f"epoch {epoch:>2}/{n_epochs}  loss={total / count:.4f}"
               f"  ({time.time() - t0:.0f}s)")
        if eval_fn is not None and (epoch % EVAL_EVERY == 0 or epoch == n_epochs):
            score = eval_fn(model)
            msg += f"  probe R@10={score:.4f}"
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
                msg += "  <- best"
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


@torch.no_grad()
def recommend(model: SASRec, tokens: dict[int, list[int]], users: np.ndarray,
              seen: dict[int, set[int]], k: int = 10) -> dict[int, list[int]]:
    """Full-catalog scoring from the last position; one forward per batch."""
    model.eval()
    recs: dict[int, list[int]] = {}
    for start in range(0, len(users), USER_GROUP):
        batch = users[start:start + USER_GROUP]
        rows = [tokens[u] for u in batch]
        lens = torch.tensor([len(r) for r in rows])
        x = torch.full((len(rows), int(lens.max())), PAD, dtype=torch.long)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r)
        h = model(x)[torch.arange(len(rows)), lens - 1]
        scores = model.logits(h)[:, 1:].numpy()  # drop PAD column -> item_idx
        for i, u in enumerate(batch):
            s = scores[i]
            s[list(seen.get(u, ()))] = -np.inf
            top = np.argpartition(-s, k)[:k]
            recs[u] = top[np.argsort(-s[top])].tolist()
    return recs


def run_fit(data: InteractionData, split: str, n_epochs: int,
            eval_users: np.ndarray | None = None):
    tokens = token_seqs(data.sequences(split))
    train_seqs = [s for s in tokens.values() if len(s) >= 2]

    eval_fn = None
    if eval_users is not None:
        seen = data.seen_items(split)

        def eval_fn(m: SASRec) -> float:
            probe = recommend(m, tokens, eval_users, seen, k=10)
            return data.recall_at_k(probe, k=10, users=eval_users)

    torch.manual_seed(SEED)
    model = SASRec(data.n_items)
    print(f"[{split}] training SASRec on {len(train_seqs)} sequences, "
          f"epochs={n_epochs}", flush=True)
    best_epoch = train_model(model, train_seqs, n_epochs, eval_fn)
    return model, tokens, best_epoch


def main() -> None:
    np.random.seed(SEED)
    data = InteractionData("data", val_frac=0.1)
    rng = np.random.default_rng(SEED)
    val_users = rng.choice(data.val_users,
                           min(VAL_SAMPLE, len(data.val_users)), replace=False)

    # ---- fit on train split with checkpoint selection
    model, tokens, best_epoch = run_fit(data, "train", EPOCHS,
                                        eval_users=val_users[:EVAL_SAMPLE])
    seen = data.seen_items("train")
    recs = recommend(model, tokens, data.val_users, seen, k=50)
    recall_full = data.recall_at_k(recs, k=10)
    recall_sample = data.recall_at_k(recs, k=10, users=val_users)
    print(f"\nSASRec val Recall@10 = {recall_full:.4f} (all "
          f"{len(data.val_users)} users) / {recall_sample:.4f} "
          f"({len(val_users)} sampled, best epoch {best_epoch})", flush=True)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([{"model": "SASRec", "recall_at_10": recall_sample}]).to_csv(
        "results/sasrec_val.csv", index=False)
    save_recs({u: recs[u] for u in val_users}, "results/sasrec_recs_val.csv")

    # ---- refit on full data + test-period interactions, write submission
    model, tokens, _ = run_fit(data, "full+test", best_epoch)
    recs = recommend(model, tokens, data.target_user_idx,
                     data.seen_items("full+test"), k=50)
    save_recs(recs, "results/sasrec_recs_target.csv")
    top10 = {u: r[:10] for u, r in recs.items()}
    path = data.write_submission(top10, "submission_sasrec.csv")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
