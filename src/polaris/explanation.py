from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from polaris.base import RecommendContext
from polaris.method import (
    encode_signed_transactions,
    is_positive,
    pos_token,
    strip_sign,
)
from polaris.scoring import (
    ScoreWeights,
    blend_scores,
    prepare_signals,
    user_signal_terms,
)

SIGNAL_LABELS: Dict[str, str] = {
    "pattern": "frequent-pattern (LFI) evidence",
    "cluster_pop": "cohort popularity",
    "feature_knn": "feature-kNN neighbours",
    "context_pop": "context popularity",
    "global_penalty": "global-popularity penalty",
    "negative": "negative-evidence penalty",
}


@dataclass
class SignalContribution:

    name: str
    label: str
    weight: float
    signal: float
    contribution: float


@dataclass
class PatternEvidence:

    cluster_key: str
    cluster_weight: float
    positive_items: List[str]
    negative_items: List[str]
    support: int
    participation: float
    score: float = 0.0


@dataclass
class Cohort:

    cluster_key: str
    weight: float
    size: int
    profile: Dict[str, str]


@dataclass
class ItemExplanation:
    item: str
    title: str
    score: float
    rank: Optional[int]
    recommended: bool
    contributions: List[SignalContribution]
    patterns: List[PatternEvidence] = field(default_factory=list)


@dataclass
class UserExplanation:
    user: str
    profile: Dict[str, str]
    cohorts: List[Cohort]
    recommendations: List[ItemExplanation]


@dataclass
class _UserScoring:
    weights: ScoreWeights
    terms: List[Tuple[str, Dict[str, float], float]]
    soft_map: Dict[str, float]
    blended: Dict[str, float]
    ranked: List[str]
    rank_of: Dict[str, int]
    n_out: Optional[int]


def _score_user(ctx: RecommendContext, user: str) -> _UserScoring:
    weights = ScoreWeights.from_config(ctx.cfg)
    prepared = prepare_signals(
        ctx.cluster_models, ctx.global_pop_full, ctx.per_cluster_pop_full, float(ctx.cfg.rrf_k)
    )
    terms, soft_map, candidates = user_signal_terms(
        user,
        prepared=prepared,
        test_assign=ctx.test_assign,
        test_soft_assign=ctx.test_soft_assign,
        feature_knn_rank=ctx.baseline_rankings.get("FeatureKNN", {}),
        context_pop_rank=ctx.baseline_rankings.get("ContextPopularity", {}),
        weights=weights,
        rrf_k=float(ctx.cfg.rrf_k),
    )
    blended = blend_scores([(scores, w) for _, scores, w in terms], candidates)
    ranked = [it for it, _ in sorted(blended.items(), key=lambda kv: (-kv[1], kv[0]))]
    rank_of = {it: i for i, it in enumerate(ranked, start=1)}
    n_max = ctx.cfg.n_max
    n_out = int(n_max) if n_max and n_max > 0 else None
    return _UserScoring(
        weights=weights,
        terms=terms,
        soft_map=soft_map,
        blended=blended,
        ranked=ranked,
        rank_of=rank_of,
        n_out=n_out,
    )


def _title_map(ctx: RecommendContext) -> Dict[str, str]:
    items = ctx.items
    if items is None or "title" not in getattr(items, "columns", []):
        return {}
    return {
        str(row.itemId): str(row.title)
        for row in items[["itemId", "title"]].itertuples(index=False)
    }


def _user_profile(ctx: RecommendContext, user: str) -> Dict[str, str]:
    rows = ctx.users[ctx.users["userId"].astype(str) == str(user)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    cols = list(ctx.demo_cols) + [c for c in ctx.context_cols if c not in ctx.demo_cols]
    return {c: str(row[c]) for c in cols if c in rows.columns}


def _cohort_profile(ctx: RecommendContext, cluster_key: str) -> Tuple[int, Dict[str, str]]:
    member_ids = (
        ctx.train_ratings.loc[ctx.train_ratings["cluster_key"] == cluster_key, "userId"]
        .astype(str)
        .unique()
    )
    members = ctx.users[ctx.users["userId"].astype(str).isin(set(member_ids))]
    profile: Dict[str, str] = {}
    for col in ctx.demo_cols:
        if col in members.columns and len(members):
            mode = members[col].mode()
            if len(mode):
                profile[col] = str(mode.iloc[0])
    return int(len(member_ids)), profile


def _cluster_transactions(
    ctx: RecommendContext, cluster_key: str, cache: Dict[str, Dict[str, FrozenSet[str]]]
) -> Dict[str, FrozenSet[str]]:
    if cluster_key in cache:
        return cache[cluster_key]
    block = ctx.train_ratings[ctx.train_ratings["cluster_key"] == cluster_key]
    txns = encode_signed_transactions(
        block,
        tau_plus=ctx.cfg.tau_plus,
        tau_minus=ctx.cfg.tau_minus,
        use_negative=ctx.cfg.use_negative,
    )
    cache[cluster_key] = txns
    return txns


def _global_pos_rate(ctx: RecommendContext) -> Dict[str, float]:
    q = getattr(ctx, "_expl_global_pos_rate", None)
    if q is None:
        tx = encode_signed_transactions(
            ctx.train_ratings, tau_plus=ctx.cfg.tau_plus, tau_minus=ctx.cfg.tau_minus,
            use_negative=ctx.cfg.use_negative,
        )
        n = len(tx)
        cnt: Dict[str, int] = {}
        for t in tx.values():
            for tok in t:
                if is_positive(tok):
                    cnt[strip_sign(tok)] = cnt.get(strip_sign(tok), 0) + 1
        q = {it: c / n for it, c in cnt.items()} if n else {}
        ctx._expl_global_pos_rate = q
    return q


def _pattern_evidence_for_item(
    ctx: RecommendContext,
    soft_map: Dict[str, float],
    item: str,
    titles: Dict[str, str],
    txn_cache: Dict[str, Dict[str, FrozenSet[str]]],
    max_patterns: int,
) -> List[PatternEvidence]:
    rho = float(ctx.cfg.lfi_pop_rho)
    q = _global_pos_rate(ctx) if rho > 0 else {}
    scored: List[tuple] = []
    for ck, weight in soft_map.items():
        model = ctx.cluster_models.get(str(ck))
        if not model or not model.selected_lfis:
            continue
        txns = _cluster_transactions(ctx, str(ck), txn_cache)
        if not txns:
            continue
        cmax = max(model.item_scores.values()) if model.item_scores else 0.0
        best = None
        for lfi in model.selected_lfis:
            supporters = [t for t in txns.values() if lfi.issubset(t)]
            n_sup = len(supporters)
            if n_sup == 0:
                continue
            pos = sum(1 for t in supporters if pos_token(item) in t)
            if pos == 0:
                continue
            participation = pos / n_sup
            score = max(0.0, participation - rho * q.get(str(item), 0.0))
            if score <= 0.0:
                continue
            if best is None or score > best[0]:
                best = (score, PatternEvidence(
                    cluster_key=str(ck),
                    cluster_weight=float(weight),
                    positive_items=[
                        titles.get(strip_sign(tok), strip_sign(tok))
                        for tok in sorted(lfi)
                        if is_positive(tok)
                    ],
                    negative_items=[
                        titles.get(strip_sign(tok), strip_sign(tok))
                        for tok in sorted(lfi)
                        if tok and tok[-1] == "-"
                    ],
                    support=n_sup,
                    participation=participation,
                    score=score,
                ))
        if best is not None:
            contribution = float(weight) * (best[0] / cmax if cmax > 0 else 0.0)
            scored.append((contribution, best[1]))
    scored.sort(key=lambda t: -t[0])
    return [ev for _, ev in scored[:max_patterns]]


def _item_explanation(
    ctx: RecommendContext,
    scoring: _UserScoring,
    item: str,
    titles: Dict[str, str],
    txn_cache: Dict[str, Dict[str, FrozenSet[str]]],
    max_patterns: int,
) -> ItemExplanation:
    contributions = [
        SignalContribution(
            name=name,
            label=SIGNAL_LABELS.get(name, name),
            weight=float(w),
            signal=float(scores.get(item, 0.0)),
            contribution=float(w * scores.get(item, 0.0)),
        )
        for name, scores, w in scoring.terms
    ]
    rank = scoring.rank_of.get(item)
    recommended = rank is not None and (scoring.n_out is None or rank <= scoring.n_out)
    patterns = _pattern_evidence_for_item(
        ctx, scoring.soft_map, item, titles, txn_cache, max_patterns
    )
    return ItemExplanation(
        item=str(item),
        title=titles.get(str(item), str(item)),
        score=float(scoring.blended.get(item, 0.0)),
        rank=rank,
        recommended=recommended,
        contributions=contributions,
        patterns=patterns,
    )


def top_item_pattern_share(ctx: RecommendContext, user: str) -> Tuple[Optional[str], float]:
    scoring = _score_user(ctx, user)
    if not scoring.ranked:
        return None, 0.0
    top = scoring.ranked[0]
    contribs = {name: w * scores.get(top, 0.0) for name, scores, w in scoring.terms}
    pos_total = sum(v for v in contribs.values() if v > 0) or 1.0
    return top, contribs.get("pattern", 0.0) / pos_total


def explain_user(
    ctx: RecommendContext,
    user: str,
    *,
    top_k: int = 5,
    max_patterns: int = 2,
    extra_items: Optional[List[str]] = None,
) -> UserExplanation:
    scoring = _score_user(ctx, user)
    titles = _title_map(ctx)
    txn_cache: Dict[str, Dict[str, FrozenSet[str]]] = {}

    cohorts: List[Cohort] = []
    for ck, w in sorted(scoring.soft_map.items(), key=lambda kv: -kv[1]):
        size, profile = _cohort_profile(ctx, str(ck))
        cohorts.append(Cohort(cluster_key=str(ck), weight=float(w), size=size, profile=profile))

    top_items = scoring.ranked[: scoring.n_out] if scoring.n_out else scoring.ranked
    chosen = list(top_items[:top_k]) + [it for it in (extra_items or []) if it not in top_items[:top_k]]
    recommendations = [
        _item_explanation(ctx, scoring, item, titles, txn_cache, max_patterns)
        for item in chosen
    ]
    return UserExplanation(
        user=str(user),
        profile=_user_profile(ctx, user),
        cohorts=cohorts,
        recommendations=recommendations,
    )


def _fmt_profile(profile: Dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in profile.items()) or "(no side information)"


def render_text(expl: UserExplanation, *, dataset: str = "", bar_width: int = 18) -> str:
    lines: List[str] = []
    head = f"Cold user {expl.user}"
    if dataset:
        head += f"  [{dataset}]"
    lines.append(head)
    lines.append(f"  side information: {_fmt_profile(expl.profile)}")
    lines.append("")
    lines.append("  matched cohorts (soft assignment):")
    for c in expl.cohorts:
        lines.append(
            f"    - {c.cluster_key}  weight {c.weight:.2f}  "
            f"({c.size} warm users: {_fmt_profile(c.profile)})"
        )
    lines.append("")
    lines.append("  recommendations:")
    for r in expl.recommendations:
        tag = f"#{r.rank}" if r.recommended else f"(rank {r.rank}, not in top set)" if r.rank else "(unscored)"
        lines.append(f"    {tag}  {r.title}   score={r.score:.4f}")
        nonzero = [c for c in r.contributions if abs(c.contribution) > 1e-12]
        pos_total = sum(c.contribution for c in nonzero if c.contribution > 0) or 1.0
        for c in sorted(nonzero, key=lambda x: -abs(x.contribution)):
            share = c.contribution / pos_total
            filled = max(0, min(bar_width, int(round(abs(share) * bar_width))))
            bar = ("+" if c.contribution >= 0 else "-") * filled
            lines.append(
                f"        {c.label:<32} {c.contribution:+.4f}  "
                f"({c.signal:.2f}x{c.weight:+.2f})  {bar}"
            )
        for p in r.patterns:
            liked = " + ".join(p.positive_items) if p.positive_items else "(none)"
            note = f"      via {p.cluster_key}: cohort users who liked [{liked}]"
            if p.negative_items:
                note += f" (and disliked [{' + '.join(p.negative_items)}])"
            note += f" also liked this -- participation {p.participation:.2f} of {p.support} supporters"
            if abs(p.score - p.participation) > 1e-9:
                note += f" (deployed contribution {p.score:.2f} after the rho-orthogonalization)"
            lines.append(note)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
