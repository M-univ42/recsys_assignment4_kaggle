
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from data import InteractionData

SEED = 42
N_LEVELS = 3          # residual quantization depth
CODEBOOK = 256        # codes per level
EMB_TEXT = 64         # SVD dims of TF-IDF block
EMB_COLLAB = 64       # SVD dims of interaction block
PAD, BOS = 0, 1
TOK_OFFSET = 2        # first semantic-code token id

D_MODEL, N_HEAD, N_LAYERS, FFN = 128, 4, 2, 256
MAX_ITEMS = 10        # history truncation (median history is 5)
EPOCHS, BATCH, LR = 24, 256, 1e-3
BEAM = 50             # final inference beam width
USER_GROUP = 32       # users batched together during beam search
VAL_SAMPLE = 2000     # validation users scored (full set is too slow on CPU)
EVAL_BEAM = 20        # cheaper beam for checkpoint-selection probes
EVAL_EVERY = 3        # epochs between probes
EVAL_SAMPLE = 500     # probe users (first 500 of the val sample)


def build_item_embeddings(data: InteractionData, X: sp.csr_matrix) -> np.ndarray:
    """(n_items, EMB_TEXT+EMB_COLLAB) from provided data only."""
    meta = data.item_meta
    text = (meta["title"].fillna("") + " " + meta["main_category"].fillna("")
            + " " + meta["store"].fillna("") + " " + meta["categories"].fillna(""))
    tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
    T = tfidf.fit_transform(text)
    T_svd = TruncatedSVD(EMB_TEXT, random_state=SEED).fit_transform(T)
    E_text = np.zeros((data.n_items, EMB_TEXT), dtype=np.float32)
    E_text[meta["item_idx"].to_numpy()] = T_svd

    E_collab = TruncatedSVD(EMB_COLLAB, random_state=SEED).fit_transform(X.T)

    def norm(E):  # unit rows; zero rows stay zero
        n = np.linalg.norm(E, axis=1, keepdims=True)
        return E / np.maximum(n, 1e-8)

    return np.hstack([norm(E_text), norm(E_collab)]).astype(np.float32)



def build_semantic_ids(E: np.ndarray) -> np.ndarray:
    """(n_items, N_LEVELS+1) int codes; last column disambiguates collisions."""
    residual = E.astype(np.float64).copy()
    codes = []
    for level in range(N_LEVELS):
        km = KMeans(CODEBOOK, n_init=2, max_iter=100, random_state=SEED + level)
        c = km.fit_predict(residual)
        residual -= km.cluster_centers_[c]
        codes.append(c)
    codes = np.stack(codes, axis=1)

    dedup = np.zeros(len(E), dtype=np.int64)
    buckets: dict[tuple, int] = {}
    for i, key in enumerate(map(tuple, codes)):
        dedup[i] = buckets.get(key, 0)
        buckets[key] = dedup[i] + 1
    print(f"semantic IDs: {len(buckets)} unique prefixes, "
          f"max collision bucket = {dedup.max() + 1}", flush=True)
    return np.hstack([codes, dedup[:, None]])


def to_tokens(sem_ids: np.ndarray) -> np.ndarray:
    """Map per-level codes to a flat LM vocabulary with level offsets."""
    toks = sem_ids.copy()
    for level in range(N_LEVELS):
        toks[:, level] += TOK_OFFSET + level * CODEBOOK
    toks[:, N_LEVELS] += TOK_OFFSET + N_LEVELS * CODEBOOK
    return toks


class TigerLM(nn.Module):
    def __init__(self, vocab: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, FFN, dropout=0.1, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, vocab)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.encoder(h, mask=mask, src_key_padding_mask=(x == PAD))
        return self.head(h)


def user_token_seqs(sequences: dict[int, list[int]],
                    item_tokens: np.ndarray) -> dict[int, list[int]]:
    """[BOS] + flattened semantic-ID tokens of the user's last MAX_ITEMS."""
    return {
        u: [BOS] + item_tokens[items[-MAX_ITEMS:]].ravel().tolist()
        for u, items in sequences.items()
    }


def train_lm(model: TigerLM, seqs: list[list[int]], n_epochs: int,
             eval_fn=None) -> int:

    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    # batch similar lengths together to minimize padding waste
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
            logits = model(x[:, :-1])
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



def build_trie(item_tokens: np.ndarray):
    allowed: dict[tuple, set[int]] = {}
    item_of: dict[tuple, int] = {}
    for item, toks in enumerate(item_tokens):
        toks = tuple(toks.tolist())
        for d in range(len(toks)):
            allowed.setdefault(toks[:d], set()).add(toks[d])
        item_of[toks] = item
    return {k: sorted(v) for k, v in allowed.items()}, item_of


@torch.no_grad()
def recommend(model: TigerLM, token_seqs: dict[int, list[int]],
              users: np.ndarray, allowed, item_of,
              seen: dict[int, set[int]], fallback: list[int],
              k: int = 10, beam: int = BEAM,
              verbose: bool = True) -> dict[int, list[int]]:
    model.eval()
    recs: dict[int, list[int]] = {}
    for gstart in range(0, len(users), USER_GROUP):
        gusers = users[gstart:gstart + USER_GROUP]
        beams = {u: [((), 0.0)] for u in gusers}
        for _ in range(N_LEVELS + 1):
            rows = [token_seqs[u] + list(p)
                    for u in gusers for p, _ in beams[u]]
            lens = torch.tensor([len(r) for r in rows])
            x = torch.full((len(rows), int(lens.max())), PAD, dtype=torch.long)
            for i, r in enumerate(rows):
                x[i, :len(r)] = torch.tensor(r)
            last = model(x)[torch.arange(len(rows)), lens - 1]
            logp = F.log_softmax(last, dim=-1)
            i = 0
            for u in gusers:
                cand = []
                for prefix, score in beams[u]:
                    lp = logp[i]
                    i += 1
                    for t in allowed.get(prefix, ()):
                        cand.append((prefix + (t,), score + lp[t].item()))
                beams[u] = sorted(cand, key=lambda c: -c[1])[:beam]
        for u in gusers:
            out, used = [], seen.get(u, set())
            for prefix, _ in beams[u]:
                item = item_of.get(prefix)
                if item is not None and item not in used and item not in out:
                    out.append(item)
                    if len(out) == k:
                        break
            for item in fallback:  # pad with popular unseen if beams ran short
                if len(out) == k:
                    break
                if item not in used and item not in out:
                    out.append(item)
            recs[u] = out
        done = gstart + len(gusers)
        if verbose and (done % 992 < USER_GROUP or done == len(users)):
            print(f"  recommended {done}/{len(users)} users", flush=True)
    return recs


def save_recs(recs: dict[int, list[int]], path: str) -> None:
    """Persist ranked candidate lists (internal indices) for ensembling."""
    pd.DataFrame({
        "user_idx": list(recs),
        "items": [",".join(map(str, v)) for v in recs.values()],
    }).to_csv(path, index=False)



def run_fit(data: InteractionData, split: str, n_epochs: int,
            eval_users: np.ndarray | None = None):
    X = data.to_csr(split)
    print(f"[{split}] building item embeddings + semantic IDs", flush=True)
    E = build_item_embeddings(data, X)
    item_tokens = to_tokens(build_semantic_ids(E))
    vocab = TOK_OFFSET + N_LEVELS * CODEBOOK + int(item_tokens[:, -1].max()) + 1

    seqs_by_user = user_token_seqs(data.sequences(split), item_tokens)
    train_seqs = [s for s in seqs_by_user.values() if len(s) > (N_LEVELS + 1)]
    allowed, item_of = build_trie(item_tokens)
    pop = np.asarray(X.sum(axis=0)).ravel()
    fallback = np.argsort(-pop)[:200].tolist()

    eval_fn = None
    if eval_users is not None:
        seen = data.seen_items(split)

        def eval_fn(m: TigerLM) -> float:
            probe = recommend(m, seqs_by_user, eval_users, allowed, item_of,
                              seen, fallback, k=10, beam=EVAL_BEAM,
                              verbose=False)
            return data.recall_at_k(probe, k=10, users=eval_users)

    max_len = 1 + (N_LEVELS + 1) * (MAX_ITEMS + 1)
    torch.manual_seed(SEED)
    model = TigerLM(vocab, max_len)
    print(f"[{split}] training LM on {len(train_seqs)} sequences, "
          f"vocab={vocab}, epochs={n_epochs}", flush=True)
    best_epoch = train_lm(model, train_seqs, n_epochs, eval_fn)
    return model, seqs_by_user, allowed, item_of, fallback, best_epoch


def main() -> None:
    np.random.seed(SEED)
    data = InteractionData("data", val_frac=0.1)
    rng = np.random.default_rng(SEED)
    val_users = rng.choice(data.val_users,
                           min(VAL_SAMPLE, len(data.val_users)), replace=False)

    model, seqs, allowed, item_of, fallback, best_epoch = run_fit(
        data, "train", EPOCHS, eval_users=val_users[:EVAL_SAMPLE])
    recs = recommend(model, seqs, val_users, allowed, item_of,
                     data.seen_items("train"), fallback, k=50)
    recall = data.recall_at_k(recs, k=10, users=val_users)
    print(f"\nTIGER val Recall@10 = {recall:.4f} "
          f"({len(val_users)} sampled users, best epoch {best_epoch})",
          flush=True)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame([{"model": "TIGER", "recall_at_10": recall}]).to_csv(
        "results/tiger_val.csv", index=False)
    save_recs(recs, "results/tiger_recs_val.csv")

    model, seqs, allowed, item_of, fallback, _ = run_fit(
        data, "full+test", best_epoch)
    recs = recommend(model, seqs, data.target_user_idx, allowed, item_of,
                     data.seen_items("full+test"), fallback, k=50)
    save_recs(recs, "results/tiger_recs_target.csv")
    top10 = {u: r[:10] for u, r in recs.items()}
    path = data.write_submission(top10, "submission_tiger.csv")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
