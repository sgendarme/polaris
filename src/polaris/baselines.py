from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline

from .features import build_user_feature_pipeline, clean_numeric_inf, split_feature_columns


def popularity_ranking(train_ratings: pd.DataFrame, rating_thr: float) -> List[str]:
    pos = train_ratings[train_ratings["rating"] >= rating_thr]
    counts = pos.groupby("itemId").size().sort_values(ascending=False)
    return [str(i) for i in counts.index.tolist()]


def popularity_recommender(
    cold_users: List[str],
    train_ratings: pd.DataFrame,
    rating_thr: float,
    n_max: int,
) -> Dict[str, List[str]]:
    ranked = popularity_ranking(train_ratings, rating_thr)
    top = ranked[: int(n_max)] if n_max and n_max > 0 else ranked
    return {str(u): list(top) for u in cold_users}


def random_recommender(
    cold_users: List[str],
    catalog: List[str],
    n_max: int,
    seed: int,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    cat = [str(i) for i in catalog]
    out: Dict[str, List[str]] = {}
    size = min(int(n_max), len(cat)) if n_max and n_max > 0 else len(cat)
    for u in cold_users:
        idx = rng.choice(len(cat), size=size, replace=False) if cat else []
        out[str(u)] = [cat[i] for i in idx]
    return out


def _complete_ranking(primary: List[str], fallback: List[str]) -> List[str]:
    seen = set(primary)
    return list(primary) + [it for it in fallback if it not in seen]


def cluster_popularity_rankings(
    train_ratings: pd.DataFrame,
    rating_thr: float,
    cluster_col: str = "cluster_key",
) -> Tuple[List[str], Dict[str, List[str]]]:
    global_rank = popularity_ranking(train_ratings, rating_thr)
    per_cluster: Dict[str, List[str]] = {}
    if cluster_col in train_ratings.columns:
        for ck, block in train_ratings.groupby(cluster_col, sort=False):
            ranked = popularity_ranking(block, rating_thr)
            per_cluster[str(ck)] = _complete_ranking(ranked, global_rank)
    return global_rank, per_cluster


def cluster_popularity_recommender(
    cold_users: List[str],
    train_ratings: pd.DataFrame,
    test_assign: Dict[str, str],
    rating_thr: float,
    n_max: int,
    cluster_col: str = "cluster_key",
) -> Dict[str, List[str]]:
    global_rank, per_cluster = cluster_popularity_rankings(train_ratings, rating_thr, cluster_col)
    size = int(n_max) if n_max and n_max > 0 else None
    out: Dict[str, List[str]] = {}
    for u in cold_users:
        ck = test_assign.get(str(u))
        ranked = per_cluster.get(str(ck), global_rank)
        out[str(u)] = list(ranked[:size]) if size is not None else list(ranked)
    return out


def _build_user_feature_matrix(
    users: pd.DataFrame,
    demo_cols: Sequence[str],
    train_user_ids: Iterable[str],
) -> Tuple[Pipeline, pd.DataFrame, np.ndarray, List[str]]:
    cols = [c for c in demo_cols if c in users.columns]
    if not cols:
        raise ValueError("FeatureKNN requires at least one available user feature column.")

    train_ids = set(map(str, train_user_ids))
    u_train = users[users["userId"].astype(str).isin(train_ids)].copy()
    if u_train.empty:
        raise ValueError("No warm users available for FeatureKNN baseline.")

    num_cols, cat_cols = split_feature_columns(u_train, demo_cols)
    pipe = build_user_feature_pipeline(num_cols, cat_cols)
    x_train_df = clean_numeric_inf(u_train[num_cols + cat_cols], num_cols)
    x_train = pipe.fit_transform(x_train_df)
    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
    x_train = np.asarray(x_train, dtype=np.float64)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    train_uid_order = [str(u) for u in u_train["userId"].tolist()]
    return pipe, u_train, x_train, train_uid_order


def feature_knn_recommender(
    cold_users: List[str],
    users: pd.DataFrame,
    train_ratings: pd.DataFrame,
    demo_cols: Sequence[str],
    rating_thr: float,
    n_max: int,
    k_neighbors: int = 50,
    global_fallback: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    train_user_ids = train_ratings["userId"].astype(str).unique().tolist()
    if global_fallback is None:
        global_fallback = popularity_ranking(train_ratings, rating_thr)
    global_fallback = [str(i) for i in global_fallback]
    size = int(n_max) if n_max and n_max > 0 else None

    pipe, _u_train, x_train, train_uid_order = _build_user_feature_matrix(users, demo_cols, train_user_ids)
    if len(train_uid_order) == 0:
        top = global_fallback[:size] if size is not None else list(global_fallback)
        return {str(u): list(top) for u in cold_users}

    cold_df = users[users["userId"].astype(str).isin(set(map(str, cold_users)))].copy()
    cold_df = cold_df.drop_duplicates(subset=["userId"], keep="first")
    if cold_df.empty:
        top = global_fallback[:size] if size is not None else list(global_fallback)
        return {str(u): list(top) for u in cold_users}

    num_cols, cat_cols = split_feature_columns(users, demo_cols)
    x_cold_df = clean_numeric_inf(cold_df[num_cols + cat_cols], num_cols)
    x_cold = pipe.transform(x_cold_df)
    if hasattr(x_cold, "toarray"):
        x_cold = x_cold.toarray()
    x_cold = np.asarray(x_cold, dtype=np.float64)
    x_cold = np.nan_to_num(x_cold, nan=0.0, posinf=0.0, neginf=0.0)

    k = max(1, min(int(k_neighbors), len(train_uid_order)))
    dists = pairwise_distances(x_cold, x_train, metric="euclidean")

    pos_train = train_ratings[train_ratings["rating"] >= rating_thr].copy()
    pos_by_user: Dict[str, List[str]] = {
        str(u): [str(i) for i in g["itemId"].tolist()]
        for u, g in pos_train.groupby("userId", sort=False)
    }

    out: Dict[str, List[str]] = {}
    for row_idx, uid in enumerate([str(u) for u in cold_df["userId"].tolist()]):
        order = np.argsort(dists[row_idx])[:k]
        scores: Dict[str, float] = {}
        for j in order.tolist():
            warm_uid = train_uid_order[j]
            weight = 1.0 / (float(dists[row_idx, j]) + 1e-9)
            for item in pos_by_user.get(warm_uid, []):
                scores[item] = scores.get(item, 0.0) + weight
        ranked = [it for it, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
        ranked = _complete_ranking(ranked, global_fallback)
        out[uid] = ranked[:size] if size is not None else ranked

    fallback_top = global_fallback[:size] if size is not None else list(global_fallback)
    return {str(u): out.get(str(u), list(fallback_top)) for u in cold_users}

def _context_key_from_row(row, context_cols: Sequence[str]) -> Tuple[str, ...]:
    vals = []
    for c in context_cols:
        v = row.get(c, "unknown")
        if pd.isna(v):
            v = "unknown"
        vals.append(str(v))
    return tuple(vals)


def context_popularity_recommender(
    cold_users: List[str],
    users: pd.DataFrame,
    train_ratings: pd.DataFrame,
    context_cols: Sequence[str],
    rating_thr: float,
    n_max: int,
    global_fallback: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    cols = [c for c in context_cols if c in users.columns]
    if not cols:
        fallback = [str(i) for i in (global_fallback or popularity_ranking(train_ratings, rating_thr))]
        size = int(n_max) if n_max and n_max > 0 else None
        top = fallback[:size] if size is not None else fallback
        return {str(u): list(top) for u in cold_users}

    if global_fallback is None:
        global_fallback = popularity_ranking(train_ratings, rating_thr)
    global_fallback = [str(i) for i in global_fallback]
    size = int(n_max) if n_max and n_max > 0 else None

    tr = train_ratings.copy()
    missing = [c for c in cols if c not in tr.columns]
    if missing:
        tr = tr.merge(users[["userId"] + cols].drop_duplicates("userId"), on="userId", how="left")

    pos = tr[tr["rating"] >= rating_thr].copy()
    exact_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    one_col_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for _, row in pos.iterrows():
        item = str(row["itemId"])
        key = _context_key_from_row(row, cols)
        exact_counts.setdefault(key, {})[item] = exact_counts.setdefault(key, {}).get(item, 0) + 1
        for c, val in zip(cols, key):
            k1 = (c, val)
            one_col_counts.setdefault(k1, {})[item] = one_col_counts.setdefault(k1, {}).get(item, 0) + 1

    user_profiles = users.drop_duplicates("userId").set_index(users["userId"].astype(str), drop=False)
    out: Dict[str, List[str]] = {}
    for uid in map(str, cold_users):
        if uid not in user_profiles.index:
            ranked = []
        else:
            row = user_profiles.loc[uid]
            key = _context_key_from_row(row, cols)
            counts = dict(exact_counts.get(key, {}))
            if not counts:
                for c, val in zip(cols, key):
                    for item, cnt in one_col_counts.get((c, val), {}).items():
                        counts[item] = counts.get(item, 0) + int(cnt)
            ranked = [it for it, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        ranked = _complete_ranking(ranked, global_fallback)
        out[uid] = ranked[:size] if size is not None else ranked
    return out
