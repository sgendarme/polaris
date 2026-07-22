from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np


def mean_or_zero(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int):
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _strip_item_sign(item_token: str) -> str:
    if not item_token:
        return item_token
    return item_token[:-1] if item_token[-1] in "+-" else item_token


def _normalize_recs(recs: Iterable[str]) -> Set[str]:
    return {_strip_item_sign(r) for r in recs}


def evaluate_set_mode_fair(
    test_pos_by_user: Dict[str, Set[str]],
    recs_by_user: Dict[str, Set[str]],
):
    precs, recs, f1s, covs = [], [], [], []
    for u, pos in test_pos_by_user.items():
        recs_set = _normalize_recs(recs_by_user.get(u, set()))
        tp = len(pos & recs_set)
        fp = len(recs_set - pos)
        fn = len(pos - recs_set)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
        covs.append(len(recs_set))
    return {
        "precision": mean_or_zero(precs),
        "recall": mean_or_zero(recs),
        "f1": mean_or_zero(f1s),
        "avg_rec_set_size": mean_or_zero(covs),
        "users_eval": len(precs),
    }


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = list(ranked)[:k]
    if not topk:
        return 0.0
    hits = sum(1 for it in topk if it in relevant)
    return hits / float(k)


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(list(ranked)[:k])
    return len(topk & relevant) / float(len(relevant))


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(list(ranked)[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    test_pos_by_user: Dict[str, Set[str]],
    ranked_by_user: Dict[str, List[str]],
    ks: Sequence[int] = (10, 20, 50),
) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for k in ks:
        acc[f"precision@{k}"] = []
        acc[f"recall@{k}"] = []
        acc[f"ndcg@{k}"] = []

    for u, relevant in test_pos_by_user.items():
        ranked = ranked_by_user.get(u, [])
        for k in ks:
            acc[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            acc[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            acc[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))

    return {name: (float(np.mean(vals)) if vals else 0.0) for name, vals in acc.items()}
