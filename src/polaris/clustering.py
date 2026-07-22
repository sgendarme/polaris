from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import Pipeline

from .features import build_user_feature_pipeline, clean_numeric_inf, split_feature_columns


@dataclass
class AgglomerativeUserClusterer:
    demo_cols: List[str]
    n_clusters: int
    linkage: str = "ward"
    seed: int = 42
    name: str = "agglomerative"
    _pipe: Optional[Pipeline] = field(init=False, default=None)
    _centroids: Optional[Dict[int, np.ndarray]] = field(init=False, default=None)
    _train_labels: Optional[Dict[str, str]] = field(init=False, default=None)
    _num_cols: List[str] = field(init=False, default_factory=list)

    def fit(self, users: pd.DataFrame, train_user_ids: Set[str]) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")

        u_train = users[users["userId"].isin(train_user_ids)].copy()
        if u_train.empty:
            raise ValueError("No training users provided to clusterer.")
        if self.n_clusters > len(u_train):
            raise ValueError(f"n_clusters ({self.n_clusters}) > nb users train ({len(u_train)})")

        num_cols, cat_cols = split_feature_columns(u_train, self.demo_cols)
        pipe = build_user_feature_pipeline(num_cols, cat_cols)

        x_train_df = clean_numeric_inf(u_train[self.demo_cols], num_cols)
        x_train = pipe.fit_transform(x_train_df)
        if hasattr(x_train, "toarray"):
            x_train = x_train.toarray()

        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        labels = model.fit_predict(x_train)

        centroids: Dict[int, np.ndarray] = {}
        for cid in range(self.n_clusters):
            idx = np.where(labels == cid)[0]
            if len(idx) == 0:
                centroids[cid] = np.zeros((x_train.shape[1],), dtype=float)
            else:
                centroids[cid] = x_train[idx].mean(axis=0)

        self._pipe = pipe
        self._centroids = centroids
        self._train_labels = {
            str(uid): f"C{int(label)}" for uid, label in zip(u_train["userId"].tolist(), labels.tolist())
        }
        self._num_cols = num_cols

    def _distance_to_centroids(self, users_subset: pd.DataFrame):
        if self._pipe is None or self._centroids is None:
            raise RuntimeError("Clusterer not fitted; call fit() first.")
        if users_subset.empty:
            return [], [], np.empty((0, 0), dtype=np.float64)

        x_df = clean_numeric_inf(users_subset[self.demo_cols], self._num_cols)
        x = self._pipe.transform(x_df)
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        cids = sorted(self._centroids.keys())
        c = np.asarray(np.vstack([self._centroids[cid] for cid in cids]), dtype=np.float64)
        if not np.isfinite(c).all():
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x2 = (x * x).sum(axis=1, keepdims=True)
            c2 = (c * c).sum(axis=1, keepdims=True).T
            xc = x @ c.T
            d2 = x2 + c2 - 2 * xc
        if not np.isfinite(d2).all():
            d2 = np.nan_to_num(d2, nan=np.inf, posinf=np.inf, neginf=np.inf)
        d2 = np.maximum(d2, 0.0)
        return [str(uid) for uid in users_subset["userId"].tolist()], cids, d2

    def assign(self, users: pd.DataFrame, user_ids: Set[str]) -> Dict[str, str]:
        if self._pipe is None or self._centroids is None:
            raise RuntimeError("Clusterer not fitted; call fit() first.")
        if not user_ids:
            return {}

        u = users[users["userId"].isin(user_ids)].copy()
        if u.empty:
            return {}

        pre_labeled = {}
        if self._train_labels:
            pre_labeled = {uid: lbl for uid, lbl in self._train_labels.items() if uid in set(map(str, user_ids))}

        remaining = [uid for uid in u["userId"].tolist() if str(uid) not in pre_labeled]
        if not remaining:
            return pre_labeled

        u_rem = u[u["userId"].isin(remaining)]
        uids, cids, d2 = self._distance_to_centroids(u_rem)
        best = d2.argmin(axis=1)

        out = {**pre_labeled}
        for uid, bi in zip(uids, best.tolist()):
            out[str(uid)] = f"C{int(cids[bi])}"
        return out

    def assign_soft(self, users: pd.DataFrame, user_ids: Set[str], top_m: int = 3) -> Dict[str, Dict[str, float]]:
        if self._pipe is None or self._centroids is None:
            raise RuntimeError("Clusterer not fitted; call fit() first.")
        hard = self.assign(users, user_ids)
        if top_m <= 1 or not user_ids:
            return {str(u): {str(c): 1.0} for u, c in hard.items()}

        u = users[users["userId"].isin(user_ids)].copy()
        if u.empty:
            return {}
        train_seen = set(self._train_labels.keys()) if self._train_labels else set()
        out: Dict[str, Dict[str, float]] = {}

        for uid, lbl in hard.items():
            if uid in train_seen:
                out[uid] = {lbl: 1.0}

        remaining = [uid for uid in u["userId"].tolist() if str(uid) not in out]
        if not remaining:
            return out

        u_rem = u[u["userId"].isin(remaining)]
        uids, cids, d2 = self._distance_to_centroids(u_rem)
        m = min(int(top_m), len(cids))
        for row_uid, row_d2 in zip(uids, d2):
            order = np.argsort(row_d2)[:m]
            vals = np.asarray([1.0 / (float(row_d2[j]) + 1e-9) for j in order], dtype=float)
            if not np.isfinite(vals).all() or float(vals.sum()) <= 0.0:
                vals = np.ones((len(order),), dtype=float)
            vals = vals / float(vals.sum())
            out[str(row_uid)] = {f"C{int(cids[j])}": float(w) for j, w in zip(order.tolist(), vals.tolist())}
        return out
