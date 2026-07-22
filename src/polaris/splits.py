from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import pandas as pd


def pick_cold_users_popular(
    ratings: pd.DataFrame,
    target_share: Tuple[float, float] = (0.20, 0.30),
    topk_items: int = 50,
    seed: int = 42,
    randomize: bool = False,
    jitter: float = 0.15,
    min_cold_users: int | None = None,
    mode: str = "popular",
) -> Set[str]:
    sizes = ratings.groupby("userId").size()

    if mode != "popular":
        ref = pick_cold_users_popular(
            ratings, target_share=target_share, topk_items=topk_items, seed=seed,
            randomize=randomize, jitter=jitter, min_cold_users=min_cold_users,
            mode="popular",
        )
        n_target = len(ref)
        rng = np.random.default_rng(seed)
        if mode == "random":
            order = rng.permutation(len(sizes))
            ranked_users = [sizes.index[i] for i in order]
        elif mode == "low_activity":
            noise = rng.uniform(1.0 - jitter, 1.0 + jitter, size=len(sizes)) if randomize else 1.0
            keys = sizes.to_numpy(dtype=float) * noise
            order = np.argsort(keys, kind="mergesort")
            ranked_users = [sizes.index[i] for i in order]
        else:
            raise ValueError(f"unknown cold-split mode: {mode!r}")
        return set(map(str, ranked_users[:n_target]))

    top_items = (
        ratings.groupby("itemId")["rating"].count().sort_values(ascending=False).head(topk_items).index.tolist()
    )
    counts = (
        ratings[ratings["itemId"].isin(top_items)]
        .groupby("userId")["rating"]
        .count()
    )

    if randomize:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(1.0 - jitter, 1.0 + jitter, size=len(counts))
        keys = counts.to_numpy(dtype=float) * noise
        order = np.argsort(-keys, kind="mergesort")
        ranked_users = [counts.index[i] for i in order]
    else:
        ranked_users = counts.sort_values(ascending=False).index.tolist()

    selected = []
    total = len(ratings)
    taken = 0
    min_users = int(min_cold_users or 0)
    for uid in ranked_users:
        user_sz = int(sizes.get(uid, 0))
        next_share = (taken + user_sz) / total
        enough_users = len(selected) >= min_users
        if next_share <= target_share[1] or not enough_users:
            selected.append(uid)
            taken += user_sz
            if (taken / total) >= target_share[0] and len(selected) >= min_users:
                break
    return set(map(str, selected))
