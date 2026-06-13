from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


class InteractionData:
    def __init__(self, data_dir: str | Path = "data", val_frac: float = 0.1,
                 dedupe: bool = True):
        self.data_dir = Path(data_dir)

        df = pd.read_csv(self.data_dir / "train.csv")
        if dedupe:
            df = df.drop_duplicates()
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

        # Contiguous internal indices over the *train* universe. Models only
        # ever score train items; submission users are all in train.
        self.user_ids = np.sort(df["user_id"].unique())
        self.item_ids = np.sort(df["item_id"].unique())
        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)
        self._user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self._item_to_idx = {it: i for i, it in enumerate(self.item_ids)}

        df["user_idx"] = df["user_id"].map(self._user_to_idx)
        df["item_idx"] = df["item_id"].map(self._item_to_idx)
        self.interactions = df  # full training log, time-sorted

        self.val_cutoff: int | None = None
        self._split(val_frac)

        test = pd.read_csv(self.data_dir / "test.csv")
        if dedupe:
            test = test.drop_duplicates()
        test["user_idx"] = test["user_id"].map(self._user_to_idx)
        test["item_idx"] = test["item_id"].map(self._item_to_idx)
        self.test_interactions = (
            test.sort_values("timestamp", kind="stable").reset_index(drop=True)
        )

        sub = pd.read_csv(self.data_dir / "sample_submission.csv")
        self.submission_template = sub
        self.target_users = sub["user_id"].to_numpy()

        self._item_meta: pd.DataFrame | None = None


    def _split(self, val_frac: float) -> None:
        df = self.interactions
        if val_frac <= 0:
            self.train = df
            self.val = df.iloc[0:0]
            self.val_ground_truth: dict[int, set[int]] = {}
            return

        cut_pos = int(len(df) * (1 - val_frac))
        self.val_cutoff = int(df["timestamp"].iloc[cut_pos])
        before = df[df["timestamp"] < self.val_cutoff]
        after = df[df["timestamp"] >= self.val_cutoff]

        # Evaluate only warm users, like the real test set.
        warm = after["user_idx"].isin(before["user_idx"].unique())
        self.train = before
        self.val = after[warm]
        self.val_ground_truth = (
            self.val.groupby("user_idx")["item_idx"].agg(set).to_dict()
        )


    def to_csr(self, split: str = "train",
               half_life_days: float | None = None) -> sp.csr_matrix:
        """Binary user x item matrix; optionally recency-weighted.

        With `half_life_days`, each (user, item) entry is 0.5^(age/half_life)
        of its most recent interaction, measured from the split's last
        timestamp — older interactions count exponentially less.
        """
        df = self._frame(split)
        if half_life_days is None:
            m = sp.csr_matrix(
                (np.ones(len(df), dtype=np.float32),
                 (df["user_idx"], df["item_idx"])),
                shape=(self.n_users, self.n_items),
            )
            m.data[:] = 1.0  # collapse repeat interactions
            return m
        df = df.drop_duplicates(["user_idx", "item_idx"], keep="last")
        age_days = (df["timestamp"].max() - df["timestamp"]) / 86_400_000
        w = np.power(0.5, age_days / half_life_days).astype(np.float32)
        return sp.csr_matrix(
            (w, (df["user_idx"], df["item_idx"])),
            shape=(self.n_users, self.n_items),
        )

    def sequences(self, split: str = "train") -> dict[int, list[int]]:
        """Chronological item_idx list per user_idx (for SASRec-style models)."""
        df = self._frame(split)
        return df.groupby("user_idx")["item_idx"].agg(list).to_dict()

    def seen_items(self, split: str = "train") -> dict[int, set[int]]:
        """Items each user has interacted with (for inference-time masking)."""
        df = self._frame(split)
        return df.groupby("user_idx")["item_idx"].agg(set).to_dict()

    def _frame(self, split: str) -> pd.DataFrame:
        if split == "train":
            return self.train
        if split == "full":
            return self.interactions
        if split == "full+test":
            return pd.concat(
                [self.interactions, self.test_interactions],
                ignore_index=True,
            ).sort_values("timestamp", kind="stable")
        raise ValueError(
            f"unknown split {split!r}, use 'train', 'full' or 'full+test'")

    @property
    def val_users(self) -> np.ndarray:
        """user_idx values that have validation ground truth."""
        return np.fromiter(self.val_ground_truth, dtype=np.int64)

    @property
    def target_user_idx(self) -> np.ndarray:
        """Internal indices of the users required in the Kaggle submission."""
        return np.array([self._user_to_idx[u] for u in self.target_users])

    @property
    def item_meta(self) -> pd.DataFrame:
        if self._item_meta is None:
            meta = pd.read_csv(self.data_dir / "item_meta.csv")
            meta["item_idx"] = meta["item_id"].map(self._item_to_idx)
            self._item_meta = meta[meta["item_idx"].notna()].astype(
                {"item_idx": np.int64}
            )
        return self._item_meta


    def recall_at_k(self, recommendations: dict[int, list[int]] | np.ndarray,
                    k: int = 10, users: np.ndarray | None = None) -> float:
        if not self.val_ground_truth:
            raise RuntimeError("no validation split (val_frac was 0)")
        if isinstance(recommendations, np.ndarray):
            recommendations = dict(zip(self.val_users, recommendations))
        eval_users = self.val_users if users is None else users
        scores = []
        for user in eval_users:
            truth = self.val_ground_truth[user]
            recs = recommendations.get(user, [])[:k]
            hits = len(truth.intersection(recs))
            scores.append(hits / min(len(truth), k))
        return float(np.mean(scores))


    def write_submission(self, recommendations: dict[int, list[int]] | np.ndarray,
                         path: str | Path = "submission.csv") -> Path:
        if isinstance(recommendations, np.ndarray):
            recommendations = dict(zip(self.target_user_idx, recommendations))
        rows = []
        for sub_id, user_id in zip(self.submission_template["ID"],
                                   self.target_users):
            recs = recommendations[self._user_to_idx[user_id]]
            if len(recs) < 10:
                raise ValueError(f"user {user_id}: only {len(recs)} recs")
            items = ",".join(str(self.item_ids[i]) for i in list(recs)[:10])
            rows.append((sub_id, user_id, items))
        out = pd.DataFrame(rows, columns=["ID", "user_id", "item_id"])
        path = Path(path)
        out.to_csv(path, index=False)
        return path
