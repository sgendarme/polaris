from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .baselines import popularity_ranking
from .lightfm_baseline import _user_feature_matrix


def _train_melu(
    examples_uiy: np.ndarray,
    X_warm: np.ndarray,
    n_items: int,
    *,
    components: int,
    local_lr: float,
    epochs: int,
    seed: int,
    hidden: int = 64,
    tasks_per_epoch: int = 2000,
    support_max: int = 10,
    query_max: int = 10,
    task_batch: int = 16,
    meta_lr: float = 5e-3,
):
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 1)
    d_feat = X_warm.shape[1]
    n_users = X_warm.shape[0]

    user_in = nn.Linear(d_feat, components)
    item_emb = nn.Embedding(n_items, components)
    decision = nn.Sequential(
        nn.Linear(2 * components, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    with torch.no_grad():
        base = float(np.clip(examples_uiy[:, 2].mean(), 1e-3, 1 - 1e-3))
        decision[-1].bias.fill_(float(np.log(base / (1 - base))))
    opt = torch.optim.Adam(
        list(user_in.parameters()) + list(item_emb.parameters()) + list(decision.parameters()),
        lr=meta_lr,
    )
    tX = torch.tensor(X_warm, dtype=torch.float32)

    by_user: Dict[int, np.ndarray] = {}
    for u in np.unique(examples_uiy[:, 0]).astype(int):
        rows = examples_uiy[examples_uiy[:, 0] == u]
        if len(rows) >= 2:
            by_user[u] = rows
    task_users = np.array(sorted(by_user.keys()))
    dec_params = list(decision.parameters())

    def forward(u_idx: int, items: "torch.Tensor", params=None):
        ue = user_in(tX[u_idx]).unsqueeze(0).expand(len(items), -1)
        ie = item_emb(items)
        h = torch.cat([ue, ie], dim=1)
        if params is None:
            return decision(h).squeeze(1)
        w0, b0, w1, b1, w2, b2 = params
        h = torch.relu(h @ w0.T + b0)
        h = torch.relu(h @ w1.T + b1)
        return (h @ w2.T + b2).squeeze(1)

    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(int(epochs)):
        chosen = rng.choice(task_users, size=min(tasks_per_epoch, len(task_users)), replace=False)
        for start in range(0, len(chosen), task_batch):
            opt.zero_grad()
            n_done = 0
            for u in chosen[start:start + task_batch]:
                rows = by_user[int(u)]
                perm = rng.permutation(len(rows))
                sup = rows[perm[:support_max]]
                qry = rows[perm[support_max:support_max + query_max]]
                if len(qry) == 0:
                    qry = sup
                s_items = torch.tensor(sup[:, 1], dtype=torch.long)
                s_y = torch.tensor(sup[:, 2], dtype=torch.float32)
                q_items = torch.tensor(qry[:, 1], dtype=torch.long)
                q_y = torch.tensor(qry[:, 2], dtype=torch.float32)
                loss_s = bce(forward(int(u), s_items), s_y)
                grads = torch.autograd.grad(loss_s, dec_params, create_graph=False)
                adapted = [p - local_lr * g for p, g in zip(dec_params, grads)]
                loss_q = bce(forward(int(u), q_items, params=adapted), q_y)
                loss_q.backward()
                n_done += 1
            if n_done:
                for p in opt.param_groups[0]["params"]:
                    if p.grad is not None:
                        p.grad /= n_done
                opt.step()
    user_in.eval(); item_emb.eval(); decision.eval()
    return user_in, item_emb, decision


def melu_recommender(
    cold_users: List[str],
    *,
    users: pd.DataFrame,
    train_ratings: pd.DataFrame,
    catalog: Sequence[str],
    demo_cols: Sequence[str],
    n_max: int,
    components: int,
    local_lr: float,
    epochs: int,
    seed: int,
    rating_thr: float,
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
    warm_ids = sorted({str(u) for u in train_ratings["userId"].unique()})
    known = set(users["userId"].astype(str))
    cold_present = [str(u) for u in cold_users if str(u) in known]
    if not warm_ids or not item_ids:
        return {str(u): list(fallback_top) for u in cold_users}

    uidx = {u: i for i, u in enumerate(warm_ids)}
    all_ids = warm_ids + [u for u in cold_present if u not in set(warm_ids)]
    feats = _user_feature_matrix(users, demo_cols, warm_ids, all_ids).toarray()
    row_of = {u: i for i, u in enumerate(all_ids)}

    rng = np.random.default_rng(int(seed) + 7)
    rated_by_user: Dict[int, set] = {}
    triples = []
    for u, it, r in zip(train_ratings["userId"].astype(str),
                        train_ratings["itemId"].astype(str),
                        train_ratings["rating"].astype(float)):
        ri, ci = uidx.get(u), item_idx.get(it)
        if ri is not None and ci is not None:
            triples.append((ri, ci, 1.0 if r >= rating_thr else 0.0))
            rated_by_user.setdefault(ri, set()).add(ci)
    if not triples:
        return {str(u): list(fallback_top) for u in cold_users}
    extra = []
    pos_count: Dict[int, int] = {}
    for ri, ci, y in triples:
        if y == 1.0:
            pos_count[ri] = pos_count.get(ri, 0) + 1
    for ri, npos in pos_count.items():
        rated = rated_by_user.get(ri, set())
        cand = rng.integers(0, len(item_ids), size=min(3 * npos, 3 * npos + 10))
        neg = [c for c in cand if c not in rated][:npos]
        extra.extend((ri, int(c), 0.0) for c in neg)
    uiy = np.asarray(triples + extra, dtype=np.float64)

    user_in, item_emb, decision = _train_melu(
        uiy, feats[: len(warm_ids)], len(item_ids),
        components=int(components), local_lr=float(local_lr),
        epochs=int(epochs), seed=int(seed),
    )

    tie = np.argsort(np.asarray(item_ids), kind="stable")
    out: Dict[str, List[str]] = {}
    with torch.no_grad():
        all_items = torch.arange(len(item_ids), dtype=torch.long)
        ie = item_emb(all_items)
        for u in cold_users:
            u = str(u)
            ri = row_of.get(u)
            if ri is None or ri < len(warm_ids):
                out[u] = list(fallback_top)
                continue
            ue = user_in(torch.tensor(feats[ri], dtype=torch.float32)).unsqueeze(0)
            h = torch.cat([ue.expand(len(item_ids), -1), ie], dim=1)
            scores = decision(h).squeeze(1).numpy()
            order = tie[np.argsort(-scores[tie], kind="stable")]
            ranked = [item_ids[j] for j in order]
            out[u] = ranked[:size] if size is not None else ranked
    return out
