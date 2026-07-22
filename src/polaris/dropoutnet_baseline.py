from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .baselines import popularity_ranking
from .lightfm_baseline import _user_feature_matrix


def _wmf_als(
    rows: np.ndarray,
    cols: np.ndarray,
    n_users: int,
    n_items: int,
    k: int,
    seed: int,
    iters: int = 15,
    reg: float = 0.1,
    alpha_conf: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    U = 0.01 * rng.standard_normal((n_users, k))
    V = 0.01 * rng.standard_normal((n_items, k))
    by_user: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_users)]
    by_item: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_items)]
    order_u = np.argsort(rows, kind="stable")
    ru, cu = rows[order_u], cols[order_u]
    for u, start, count in zip(*np.unique(ru, return_index=True, return_counts=True)):
        by_user[int(u)] = cu[start:start + count]
    order_i = np.argsort(cols, kind="stable")
    ri, ci = rows[order_i], cols[order_i]
    for i, start, count in zip(*np.unique(ci, return_index=True, return_counts=True)):
        by_item[int(i)] = ri[start:start + count]

    eye = np.eye(k)
    for _ in range(int(iters)):
        VtV = V.T @ V
        for u in range(n_users):
            idx = by_user[u]
            if idx.size == 0:
                U[u] = 0.0
                continue
            Vi = V[idx]
            A = VtV + alpha_conf * (Vi.T @ Vi) + reg * eye
            b = (1.0 + alpha_conf) * Vi.sum(axis=0)
            U[u] = np.linalg.solve(A, b)
        UtU = U.T @ U
        for i in range(n_items):
            idx = by_item[i]
            if idx.size == 0:
                V[i] = 0.0
                continue
            Ui = U[idx]
            A = UtU + alpha_conf * (Ui.T @ Ui) + reg * eye
            b = (1.0 + alpha_conf) * Ui.sum(axis=0)
            V[i] = np.linalg.solve(A, b)
    return U, V


def _train_towers(
    U: np.ndarray,
    V: np.ndarray,
    X_warm: np.ndarray,
    pos_rows: np.ndarray,
    pos_cols: np.ndarray,
    *,
    dropout: float,
    epochs: int,
    seed: int,
    hidden: int = 256,
    batch: int = 1024,
    lr: float = 1e-3,
    max_pairs_per_epoch: int = 200_000,
):
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 1)
    n_users, k = U.shape
    n_items = V.shape[0]
    d = X_warm.shape[1]

    tU = torch.tensor(U, dtype=torch.float32)
    tV = torch.tensor(V, dtype=torch.float32)
    tX = torch.tensor(X_warm, dtype=torch.float32)
    base_scores = tU @ tV.T

    class Tower(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, out_dim)
            )

        def forward(self, x):
            return self.net(x)

    f_u = Tower(k + d, k)
    f_v = Tower(k, k)
    opt = torch.optim.Adam(list(f_u.parameters()) + list(f_v.parameters()), lr=lr)

    n_pos = pos_rows.size
    half = min(max_pairs_per_epoch // 2, n_pos)
    for _ in range(int(epochs)):
        pos_sel = rng.choice(n_pos, size=half, replace=False) if half < n_pos else np.arange(n_pos)
        r = np.concatenate([pos_rows[pos_sel], rng.integers(0, n_users, size=half)])
        c = np.concatenate([pos_cols[pos_sel], rng.integers(0, n_items, size=half)])
        perm = rng.permutation(r.size)
        r, c = r[perm], c[perm]
        for start in range(0, r.size, batch):
            bu = torch.tensor(r[start:start + batch], dtype=torch.long)
            bv = torch.tensor(c[start:start + batch], dtype=torch.long)
            pref = tU[bu]
            keep = torch.tensor(
                (rng.random(len(bu)) >= dropout).astype(np.float32)
            ).unsqueeze(1)
            u_in = torch.cat([pref * keep, tX[bu]], dim=1)
            u_lat = f_u(u_in)
            v_lat = f_v(tV[bv])
            pred = (u_lat * v_lat).sum(dim=1)
            target = base_scores[bu, bv]
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    f_u.eval()
    f_v.eval()
    return f_u, f_v


def dropoutnet_recommender(
    cold_users: List[str],
    *,
    users: pd.DataFrame,
    train_ratings: pd.DataFrame,
    catalog: Sequence[str],
    demo_cols: Sequence[str],
    rating_thr: float,
    n_max: int,
    components: int,
    dropout: float,
    epochs: int,
    seed: int,
    global_fallback: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    import torch

    if global_fallback is None:
        global_fallback = popularity_ranking(train_ratings, rating_thr)
    global_fallback = [str(i) for i in global_fallback]
    size = int(n_max) if n_max and n_max > 0 else None
    fallback_top = global_fallback[:size] if size is not None else list(global_fallback)

    item_ids = [str(i) for i in catalog]
    item_idx = {it: j for j, it in enumerate(item_ids)}
    n_items = len(item_ids)
    warm_ids = sorted({str(u) for u in train_ratings["userId"].unique()})
    known = set(users["userId"].astype(str))
    cold_present = [str(u) for u in cold_users if str(u) in known]
    if not warm_ids or n_items == 0:
        return {str(u): list(fallback_top) for u in cold_users}

    all_ids = warm_ids + [u for u in cold_present if u not in set(warm_ids)]
    feats = _user_feature_matrix(users, demo_cols, warm_ids, all_ids).toarray()
    row_of = {u: i for i, u in enumerate(all_ids)}
    X_warm = feats[: len(warm_ids)]

    uidx = {u: i for i, u in enumerate(warm_ids)}
    pos = train_ratings[train_ratings["rating"] >= rating_thr]
    rows, cols = [], []
    for u, it in zip(pos["userId"].astype(str), pos["itemId"].astype(str)):
        ri, ci = uidx.get(u), item_idx.get(it)
        if ri is not None and ci is not None:
            rows.append(ri)
            cols.append(ci)
    if not rows:
        return {str(u): list(fallback_top) for u in cold_users}
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    k = int(components)
    U, V = _wmf_als(rows, cols, len(warm_ids), n_items, k, seed)
    f_u, f_v = _train_towers(
        U, V, X_warm, rows, cols,
        dropout=float(dropout), epochs=int(epochs), seed=int(seed),
    )

    with torch.no_grad():
        v_lat = f_v(torch.tensor(V, dtype=torch.float32)).numpy()
        out: Dict[str, List[str]] = {}
        item_arr = np.asarray(item_ids)
        tie = np.argsort(item_arr, kind="stable")
        for u in cold_users:
            u = str(u)
            ri = row_of.get(u)
            if ri is None or ri < len(warm_ids):
                out[u] = list(fallback_top)
                continue
            x = feats[ri]
            u_in = np.concatenate([np.zeros(k, dtype=np.float32), x.astype(np.float32)])
            u_lat = f_u(torch.tensor(u_in, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            scores = v_lat @ u_lat
            order = tie[np.argsort(-scores[tie], kind="stable")]
            ranked = [item_ids[j] for j in order]
            out[u] = ranked[:size] if size is not None else ranked
    return out
