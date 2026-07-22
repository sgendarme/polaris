from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import pandas as pd

from polaris.mining import find_largest_frequent_itemset
from polaris.progress import log as _log


def pos_token(item: str) -> str:
    return f"{item}+"


def neg_token(item: str) -> str:
    return f"{item}-"


def is_positive(token: str) -> bool:
    return bool(token) and token[-1] == "+"


def strip_sign(token: str) -> str:
    if not token:
        return token
    return token[:-1] if token[-1] in "+-" else token


def _rel2abs(theta: float, n: int, min_support_abs: int = 1) -> int:
    floor = max(1, int(min_support_abs or 1))
    if n <= 0:
        return floor
    return max(floor, int(theta) if theta >= 1 else int(math.ceil(theta * n)))


def encode_signed_transactions(
    block_df: pd.DataFrame,
    tau_plus: float,
    tau_minus: float,
    use_negative: bool = True,
) -> Dict[str, FrozenSet[str]]:
    out: Dict[str, FrozenSet[str]] = {}
    for u, g in block_df.groupby("userId", sort=False):
        toks: Set[str] = set()
        ratings = g["rating"].to_numpy()
        items = g["itemId"].to_numpy()
        for i in range(len(ratings)):
            rating, it = ratings[i], str(items[i])
            if rating >= tau_plus:
                toks.add(pos_token(it))
            elif use_negative and rating <= tau_minus:
                toks.add(neg_token(it))
        if toks:
            out[str(u)] = frozenset(toks)
    return out


def _support(pattern: FrozenSet[str], pool: List[FrozenSet[str]]) -> int:
    return sum(1 for t in pool if pattern.issubset(t))


def transactions_as_list(transactions: Dict[str, FrozenSet[str]]) -> List[FrozenSet[str]]:
    return [t for t in transactions.values() if t]


def select_lfis_by_peeling(
    transactions: Dict[str, FrozenSet[str]],
    theta: float,
    max_k: int,
    n_lfi: int,
    log_prefix: str = "",
    min_support_abs: int = 1,
    forbid_singleton_support: bool = False,
    auto_peeling: bool = False,
    min_new_positive_items: int = 1,
    max_lfi_overlap: float = 1.0,
) -> List[FrozenSet[str]]:
    if n_lfi <= 0:
        return []

    pool: List[FrozenSet[str]] = [t for t in transactions.values() if t]
    selected: List[FrozenSet[str]] = []

    for rnd in range(1, int(n_lfi) + 1):
        if not pool:
            break
        support_floor = max(int(min_support_abs or 1), 2 if forbid_singleton_support else 1)
        minsup_abs = _rel2abs(theta, len(pool), min_support_abs=support_floor)
        t_round = time.perf_counter()
        p_t = find_largest_frequent_itemset(pool, minsup=minsup_abs, max_k=max_k)
        if p_t is None:
            if log_prefix:
                _log(
                    f"{log_prefix} round {rnd}/{n_lfi}: no frequent itemset "
                    f"(pool={len(pool)}, minsup={minsup_abs}) -> stop"
                )
            break

        if auto_peeling and selected:
            cur_items = {strip_sign(tok) for tok in p_t if is_positive(tok)}
            prev_items = {strip_sign(tok) for pat in selected for tok in pat if is_positive(tok)}
            new_items = len(cur_items - prev_items)
            if new_items < int(min_new_positive_items):
                if log_prefix:
                    _log(
                        f"{log_prefix} round {rnd}/{n_lfi}: stop by auto-peeling "
                        f"(new_positive_items={new_items})"
                    )
                break
            if prev_items and cur_items:
                overlap = len(cur_items & prev_items) / float(len(cur_items))
                if overlap > float(max_lfi_overlap):
                    if log_prefix:
                        _log(
                            f"{log_prefix} round {rnd}/{n_lfi}: stop by auto-peeling "
                            f"(overlap={overlap:.2f})"
                        )
                    break

        selected.append(p_t)
        before = len(pool)
        pool = [t for t in pool if not p_t.issubset(t)]
        if log_prefix:
            _log(
                f"{log_prefix} round {rnd}/{n_lfi}: |LFI|={len(p_t)} "
                f"sup={_support(p_t, transactions_as_list(transactions))} "
                f"covered={before - len(pool)} pool={before}->{len(pool)} "
                f"minsup={minsup_abs} ({time.perf_counter() - t_round:.2f}s)"
            )

    return selected


def generate_recommendation(
    transactions: Dict[str, FrozenSet[str]],
    selected_lfis: List[FrozenSet[str]],
    delta: float,
    n_max: int,
    negative_penalty: float = 0.0,
    global_pos_rate: Dict[str, float] | None = None,
    pop_rho: float = 0.0,
) -> Tuple[Set[str], Dict[str, float]]:
    score: Dict[str, float] = {}
    alpha = max(0.0, float(negative_penalty))
    rho = max(0.0, float(pop_rho))
    base_rate = global_pos_rate or {}

    for p_t in selected_lfis:
        supporters = [t for t in transactions.values() if p_t.issubset(t)]
        n_sup = len(supporters)
        if n_sup == 0:
            continue
        pos_counts: Counter = Counter()
        neg_counts: Counter = Counter()
        for t in supporters:
            for tok in t:
                item = strip_sign(tok)
                if is_positive(tok):
                    pos_counts[item] += 1
                elif tok and tok[-1] == "-":
                    neg_counts[item] += 1
        for item, c_pos in pos_counts.items():
            p_pos = c_pos / n_sup
            p_neg = neg_counts.get(item, 0) / n_sup
            p = p_pos - alpha * p_neg
            if rho > 0:
                p -= rho * base_rate.get(item, 0.0)
            p = max(0.0, p)
            if p > score.get(item, 0.0):
                score[item] = p

    eligible = [(it, s) for it, s in score.items() if s >= delta and s > 0.0]
    eligible.sort(key=lambda kv: (-kv[1], kv[0]))
    if n_max and n_max > 0:
        eligible = eligible[: int(n_max)]
    rec_set = {it for it, _ in eligible}
    return rec_set, score


def _normalised_rrf_scores(ranked_items: Sequence[str], rrf_k: float = 60.0) -> Dict[str, float]:
    ranked = [str(i) for i in ranked_items]
    if not ranked:
        return {}
    denom = float(rrf_k) + 1.0
    return {item: denom / (float(rrf_k) + float(rank)) for rank, item in enumerate(ranked, start=1)}


@dataclass
class ClusterModel:

    cluster_key: str
    n_train_users: int
    selected_lfis: List[FrozenSet[str]]
    rec_set: Set[str]
    item_scores: Dict[str, float]
    ranked_items: List[str] = field(default_factory=list)
    negative_item_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ranked_items = [
            it for it, _ in sorted(self.item_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]


def cluster_negative_scores(transactions: Dict[str, FrozenSet[str]]) -> Dict[str, float]:
    if not transactions:
        return {}
    counts: Counter = Counter()
    for t in transactions.values():
        for tok in t:
            if tok and tok[-1] == "-":
                counts[strip_sign(tok)] += 1
    n = float(len(transactions))
    return {str(item): float(c) / n for item, c in counts.items()}


def _positive_token_counts(transactions: Dict[str, FrozenSet[str]]) -> Counter:
    cnt: Counter = Counter()
    for t in transactions.values():
        for tok in t:
            if tok and tok[-1] == "+":
                cnt[strip_sign(tok)] += 1
    return cnt


def fit_cluster_models(
    train_df: pd.DataFrame,
    cluster_col: str,
    tau_plus: float,
    tau_minus: float,
    theta: float,
    max_k: int,
    n_lfi: int,
    delta: float,
    n_max: int,
    log_prefix: str = "",
    use_negative: bool = True,
    negative_penalty: float = 0.0,
    min_support_abs: int = 1,
    forbid_singleton_support: bool = False,
    auto_peeling: bool = False,
    min_new_positive_items: int = 1,
    max_lfi_overlap: float = 1.0,
    pop_rho: float = 0.0,
    global_pos_rate_override: Dict[str, float] | None = None,
) -> Dict[str, ClusterModel]:
    global_pos_rate: Dict[str, float] = dict(global_pos_rate_override or {})
    if not global_pos_rate and pop_rho > 0:
        global_tx = encode_signed_transactions(
            train_df, tau_plus=tau_plus, tau_minus=tau_minus, use_negative=use_negative
        )
        n_global = len(global_tx)
        if n_global > 0:
            global_pos_rate = {
                it: c / n_global for it, c in _positive_token_counts(global_tx).items()
            }

    models: Dict[str, ClusterModel] = {}
    groups = list(train_df.groupby(cluster_col, sort=False))
    n_groups = len(groups)
    for ci, (cluster_key, block) in enumerate(groups, start=1):
        transactions = encode_signed_transactions(
            block, tau_plus=tau_plus, tau_minus=tau_minus, use_negative=use_negative
        )
        if not transactions:
            if log_prefix:
                _log(f"{log_prefix} cluster {ci}/{n_groups} '{cluster_key}': empty, skipped")
            models[str(cluster_key)] = ClusterModel(
                cluster_key=str(cluster_key),
                n_train_users=0,
                selected_lfis=[],
                rec_set=set(),
                item_scores={},
                negative_item_scores={},
            )
            continue
        if log_prefix:
            _log(
                f"{log_prefix} cluster {ci}/{n_groups} '{cluster_key}': "
                f"mining (users={len(transactions)}, n_lfi={n_lfi}, max_k={max_k})"
            )
        t_cluster = time.perf_counter()
        lfis = select_lfis_by_peeling(
            transactions=transactions,
            theta=theta,
            max_k=max_k,
            n_lfi=n_lfi,
            log_prefix=f"{log_prefix} cluster {ci}/{n_groups} '{cluster_key}'" if log_prefix else "",
            min_support_abs=min_support_abs,
            forbid_singleton_support=forbid_singleton_support,
            auto_peeling=auto_peeling,
            min_new_positive_items=min_new_positive_items,
            max_lfi_overlap=max_lfi_overlap,
        )
        rec_set, scores = generate_recommendation(
            transactions=transactions,
            selected_lfis=lfis,
            delta=delta,
            n_max=n_max,
            negative_penalty=negative_penalty,
            global_pos_rate=global_pos_rate,
            pop_rho=pop_rho,
        )
        if log_prefix:
            _log(
                f"{log_prefix} cluster {ci}/{n_groups} '{cluster_key}': "
                f"{len(lfis)} LFI(s), rec_set={len(rec_set)} items "
                f"({time.perf_counter() - t_cluster:.2f}s)"
            )
        models[str(cluster_key)] = ClusterModel(
            cluster_key=str(cluster_key),
            n_train_users=len(transactions),
            selected_lfis=lfis,
            rec_set=rec_set,
            item_scores=scores,
            negative_item_scores=cluster_negative_scores(transactions),
        )
    return models
