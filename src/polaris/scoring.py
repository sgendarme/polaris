from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from polaris.config import ExperimentConfig
from polaris.base import RecResult
from polaris.method import ClusterModel, _normalised_rrf_scores


@dataclass(frozen=True)
class ScoreWeights:

    pattern: float
    fallback: float
    feature: float
    context: float
    global_penalty: float
    negative_penalty: float

    @staticmethod
    def from_config(cfg: ExperimentConfig) -> "ScoreWeights":
        w_pattern = min(1.0, max(0.0, float(cfg.hybrid_lambda)))
        w_feature = min(1.0, max(0.0, float(cfg.feature_knn_weight)))
        w_context = min(1.0, max(0.0, float(cfg.context_pop_weight)))
        if w_pattern + w_feature + w_context > 1.0:
            scale = 1.0 / (w_pattern + w_feature + w_context)
            w_pattern *= scale
            w_feature *= scale
            w_context *= scale
        w_fallback = max(0.0, 1.0 - w_pattern - w_feature - w_context)
        return ScoreWeights(
            pattern=w_pattern,
            fallback=w_fallback,
            feature=w_feature,
            context=w_context,
            global_penalty=max(0.0, float(cfg.global_pop_penalty)),
            negative_penalty=max(0.0, float(cfg.personalized_negative_weight)),
        )


def blend_scores(
    signals: Sequence[Tuple[Dict[str, float], float]],
    candidates: Set[str],
) -> Dict[str, float]:
    return {
        item: sum(weight * scores.get(item, 0.0) for scores, weight in signals)
        for item in candidates
    }


def _normalize_by_max(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(float(v) for v in scores.values())
    if max_score <= 0:
        return {}
    return {str(it): float(v) / max_score for it, v in scores.items()}


SIGNAL_NAMES = (
    "pattern",
    "cluster_pop",
    "feature_knn",
    "context_pop",
    "global_penalty",
    "negative",
)


@dataclass(frozen=True)
class PreparedSignals:

    norm_pattern_by_cluster: Dict[str, Dict[str, float]]
    norm_negative_by_cluster: Dict[str, Dict[str, float]]
    global_rrf_scores: Dict[str, float]
    cluster_rrf_scores: Dict[str, Dict[str, float]]
    all_global_items: Set[str]


def prepare_signals(
    cluster_models: Dict[str, ClusterModel],
    global_pop_full: List[str],
    per_cluster_pop_full: Dict[str, List[str]],
    rrf_k: float,
) -> PreparedSignals:
    norm_pattern_by_cluster: Dict[str, Dict[str, float]] = {}
    norm_negative_by_cluster: Dict[str, Dict[str, float]] = {}
    for ck_model, model_obj in cluster_models.items():
        norm_pattern_by_cluster[str(ck_model)] = _normalize_by_max(model_obj.item_scores)
        norm_negative_by_cluster[str(ck_model)] = _normalize_by_max(model_obj.negative_item_scores)
    global_rrf_scores = _normalised_rrf_scores(global_pop_full, rrf_k=float(rrf_k))
    cluster_rrf_scores = {
        str(ck): _normalised_rrf_scores(rank, rrf_k=float(rrf_k))
        for ck, rank in per_cluster_pop_full.items()
    }
    return PreparedSignals(
        norm_pattern_by_cluster=norm_pattern_by_cluster,
        norm_negative_by_cluster=norm_negative_by_cluster,
        global_rrf_scores=global_rrf_scores,
        cluster_rrf_scores=cluster_rrf_scores,
        all_global_items=set(global_rrf_scores.keys()),
    )


def user_signal_terms(
    user: str,
    *,
    prepared: PreparedSignals,
    test_assign: Dict[str, str],
    test_soft_assign: Dict[str, Dict[str, float]],
    feature_knn_rank: Dict[str, List[str]],
    context_pop_rank: Dict[str, List[str]],
    weights: ScoreWeights,
    rrf_k: float,
) -> Tuple[List[Tuple[str, Dict[str, float], float]], Dict[str, float], Set[str]]:
    ck = test_assign.get(user)
    soft_map = test_soft_assign.get(str(user), {str(ck): 1.0} if ck is not None else {})
    pattern_scores: Dict[str, float] = {}
    fallback_scores: Dict[str, float] = {}
    negative_scores: Dict[str, float] = {}
    total_pattern_w = 0.0
    total_negative_w = 0.0

    for soft_ck, soft_w in soft_map.items():
        ww = max(0.0, float(soft_w))
        if ww <= 0:
            continue
        ps = prepared.norm_pattern_by_cluster.get(str(soft_ck), {})
        if ps:
            total_pattern_w += ww
            for item, value in ps.items():
                pattern_scores[item] = pattern_scores.get(item, 0.0) + ww * value
        ns = prepared.norm_negative_by_cluster.get(str(soft_ck), {})
        if ns:
            total_negative_w += ww
            for item, value in ns.items():
                negative_scores[item] = negative_scores.get(item, 0.0) + ww * value
        fs = prepared.cluster_rrf_scores.get(str(soft_ck), prepared.global_rrf_scores)
        for item, value in fs.items():
            fallback_scores[item] = fallback_scores.get(item, 0.0) + ww * value

    if total_pattern_w > 0:
        pattern_scores = {it: val / total_pattern_w for it, val in pattern_scores.items()}
    if total_negative_w > 0:
        negative_scores = {it: val / total_negative_w for it, val in negative_scores.items()}
    if not fallback_scores:
        fallback_scores = dict(prepared.global_rrf_scores)

    feature_scores = _normalised_rrf_scores(feature_knn_rank.get(user, []), rrf_k=float(rrf_k))
    context_scores = _normalised_rrf_scores(context_pop_rank.get(user, []), rrf_k=float(rrf_k))

    terms: List[Tuple[str, Dict[str, float], float]] = [
        ("pattern", pattern_scores, weights.pattern),
        ("cluster_pop", fallback_scores, weights.fallback),
        ("feature_knn", feature_scores, weights.feature),
        ("context_pop", context_scores, weights.context),
        ("global_penalty", prepared.global_rrf_scores, -weights.global_penalty),
        ("negative", negative_scores, -weights.negative_penalty),
    ]
    candidates = (
        prepared.all_global_items
        | set(fallback_scores.keys())
        | set(pattern_scores.keys())
        | set(feature_scores.keys())
        | set(context_scores.keys())
    )
    return terms, soft_map, candidates


def adaptive_rankings(
    *,
    eval_users: List[str],
    test_assign: Dict[str, str],
    test_soft_assign: Dict[str, Dict[str, float]],
    cluster_models: Dict[str, ClusterModel],
    global_pop_full: List[str],
    per_cluster_pop_full: Dict[str, List[str]],
    feature_knn_rank: Dict[str, List[str]],
    context_pop_rank: Dict[str, List[str]],
    weights: ScoreWeights,
    rrf_k: float,
    n_max: int,
) -> RecResult:
    prepared = prepare_signals(cluster_models, global_pop_full, per_cluster_pop_full, rrf_k)
    n_out = int(n_max) if n_max and n_max > 0 else None

    ranked_out: Dict[str, List[str]] = {}
    sets_out: Dict[str, Set[str]] = {}

    for u in eval_users:
        terms, _soft_map, candidates = user_signal_terms(
            u,
            prepared=prepared,
            test_assign=test_assign,
            test_soft_assign=test_soft_assign,
            feature_knn_rank=feature_knn_rank,
            context_pop_rank=context_pop_rank,
            weights=weights,
            rrf_k=rrf_k,
        )
        blended = blend_scores([(scores, w) for _, scores, w in terms], candidates)
        ranked = [it for it, _ in sorted(blended.items(), key=lambda kv: (-kv[1], kv[0]))]
        h_rank = ranked[:n_out] if n_out is not None else ranked
        ranked_out[u] = h_rank
        sets_out[u] = set(h_rank)

    return RecResult(ranked=ranked_out, sets=sets_out)
