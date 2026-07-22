from __future__ import annotations

import dataclasses
import json
import os
import sys
import zlib
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.baselines import (
    cluster_popularity_rankings,
    context_popularity_recommender,
    feature_knn_recommender,
)
from polaris.method import (
    ClusterModel,
    encode_signed_transactions,
    fit_cluster_models,
    generate_recommendation,
    pos_token,
)
from polaris.explanation import _score_user

DATASETS = [("ml-100k", "ML-100K"), ("ml-1m", "ML-1M"), ("comoda", "LDOS")]
BIG_RANK = 10**6


def build_prepared(ds_key: str, run_cfg: dict, shared: dict, seed: int):
    dataset = build_dataset(ds_key)
    cl = AgglomerativeUserClusterer(
        demo_cols=[str(x) for x in run_cfg["demo_cols"]],
        n_clusters=int(run_cfg["n_clusters"]),
        linkage="ward",
        seed=1,
    )
    exp = Experiment(dataset=dataset, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, run_cfg, n_lfi=int(run_cfg["n_lfi"]))
    prep = exp.prepare_context(cfg, seed=seed, randomize_split=True)
    _populate_baselines(prep.ctx, prep.eval_users, prep.ctx.train_ratings)
    return prep


def _populate_baselines(ctx, users: List[str], train_ratings) -> None:
    gpop, cpop = cluster_popularity_rankings(train_ratings, ctx.cfg.tau_plus)
    ctx.global_pop_full = gpop
    ctx.per_cluster_pop_full = cpop
    ctx.baseline_rankings["FeatureKNN"] = feature_knn_recommender(
        users,
        users=ctx.users,
        train_ratings=train_ratings,
        demo_cols=ctx.demo_cols,
        rating_thr=ctx.cfg.tau_plus,
        n_max=ctx.cfg.n_max,
        k_neighbors=int(ctx.cfg.knn_neighbors),
        global_fallback=gpop,
    )
    ctx.baseline_rankings["ContextPopularity"] = context_popularity_recommender(
        users,
        users=ctx.users,
        train_ratings=train_ratings,
        context_cols=ctx.context_cols,
        rating_thr=ctx.cfg.tau_plus,
        n_max=ctx.cfg.n_max,
        global_fallback=gpop,
    )


def rank_and_top(ctx, user: str, topk: int) -> Tuple[Dict[str, int], List[str]]:
    s = _score_user(ctx, user)
    return s.rank_of, s.ranked[:topk]


def cluster_txns(ctx) -> Dict[str, Dict[str, FrozenSet[str]]]:
    out: Dict[str, Dict[str, FrozenSet[str]]] = {}
    for ck in ctx.cluster_models:
        block = ctx.train_ratings[ctx.train_ratings["cluster_key"] == ck]
        out[str(ck)] = encode_signed_transactions(
            block, tau_plus=ctx.cfg.tau_plus, tau_minus=ctx.cfg.tau_minus,
            use_negative=ctx.cfg.use_negative,
        )
    return out


def soft_map(ctx, user: str) -> Dict[str, float]:
    ck = ctx.test_assign.get(user)
    return ctx.test_soft_assign.get(str(user), {str(ck): 1.0} if ck is not None else {})


def global_pos_rate_of(ctx, train_df) -> dict:
    from polaris.method import _positive_token_counts, encode_signed_transactions
    tx = encode_signed_transactions(train_df, tau_plus=ctx.cfg.tau_plus,
                                    tau_minus=ctx.cfg.tau_minus,
                                    use_negative=ctx.cfg.use_negative)
    n = len(tx)
    if n == 0:
        return {}
    return {it: c / n for it, c in _positive_token_counts(tx).items()}

def supporting_lfi(
    ctx, smap: Dict[str, float], item: str, txns: Dict[str, Dict[str, FrozenSet[str]]]
) -> Optional[Tuple[str, int, int, float]]:
    rho = float(ctx.cfg.lfi_pop_rho)
    q = getattr(ctx, "_global_pos_rate_cache", None)
    if rho > 0 and q is None:
        q = global_pos_rate_of(ctx, ctx.train_ratings)
        ctx._global_pos_rate_cache = q
    best = None
    for ck, w in smap.items():
        model = ctx.cluster_models.get(str(ck))
        if not model or not model.selected_lfis:
            continue
        tx = txns.get(str(ck), {})
        cmax = max(model.item_scores.values()) if model.item_scores else 0.0
        for idx, lfi in enumerate(model.selected_lfis):
            supporters = [t for t in tx.values() if lfi.issubset(t)]
            n_sup = len(supporters)
            if n_sup == 0:
                continue
            pos = sum(1 for t in supporters if pos_token(item) in t)
            if pos == 0:
                continue
            part = pos / n_sup
            score = part - rho * q.get(str(item), 0.0) if rho > 0 else part
            if score <= 0:
                continue
            contribution = float(w) * (score / cmax if cmax > 0 else 0.0)
            if contribution <= 0:
                continue
            if best is None or contribution > best[0]:
                best = (contribution, score, str(ck), idx, n_sup)
    if best is None:
        return None
    return best[2], best[3], best[4], best[1]


def ablate_lfi(ctx, ck: str, idx: int, txns_ck: Dict[str, FrozenSet[str]]):
    model = ctx.cluster_models[ck]
    kept = [l for j, l in enumerate(model.selected_lfis) if j != idx]
    rho = float(ctx.cfg.lfi_pop_rho)
    gpr = global_pos_rate_of(ctx, ctx.train_ratings) if rho > 0 else None
    rec_set, scores = generate_recommendation(
        txns_ck, kept, ctx.cfg.delta, ctx.cfg.n_max, ctx.cfg.negative_penalty,
        global_pos_rate=gpr, pop_rho=rho,
    )
    new_model = ClusterModel(
        cluster_key=ck, n_train_users=model.n_train_users, selected_lfis=kept,
        rec_set=rec_set, item_scores=scores, negative_item_scores=model.negative_item_scores,
    )
    return dataclasses.replace(ctx, cluster_models={**ctx.cluster_models, ck: new_model})


def remove_users(ctx, ck: str, drop_ids: List[str], rescore_users: List[str]):
    drop = set(map(str, drop_ids))
    ptrain = ctx.train_ratings[~ctx.train_ratings["userId"].astype(str).isin(drop)].copy()
    block = ptrain[ptrain["cluster_key"] == ck]
    refit = fit_cluster_models(
        train_df=block, cluster_col="cluster_key",
        tau_plus=ctx.cfg.tau_plus, tau_minus=ctx.cfg.tau_minus, theta=ctx.cfg.theta,
        max_k=ctx.cfg.max_k, n_lfi=ctx.cfg.n_lfi, delta=ctx.cfg.delta, n_max=ctx.cfg.n_max,
        use_negative=ctx.cfg.use_negative, negative_penalty=ctx.cfg.negative_penalty,
        min_support_abs=int(ctx.cfg.min_support_abs),
        forbid_singleton_support=bool(ctx.cfg.forbid_singleton_support),
        auto_peeling=bool(ctx.cfg.auto_peeling),
        min_new_positive_items=int(ctx.cfg.min_new_positive_items),
        max_lfi_overlap=float(ctx.cfg.max_lfi_overlap),
        pop_rho=float(ctx.cfg.lfi_pop_rho),
        global_pos_rate_override=(global_pos_rate_of(ctx, ptrain)
                                  if float(ctx.cfg.lfi_pop_rho) > 0 else None),
    )
    new_model = refit.get(ck) or ClusterModel(ck, 0, [], set(), {}, negative_item_scores={})
    pctx = dataclasses.replace(
        ctx,
        train_ratings=ptrain,
        cluster_models={**ctx.cluster_models, ck: new_model},
        baseline_rankings=dict(ctx.baseline_rankings),
    )
    _populate_baselines(pctx, rescore_users, ptrain)
    return pctx


def run_seed(prep, topk: int, seed: int) -> dict:
    ctx = prep.ctx
    eval_users = prep.eval_users
    txns = cluster_txns(ctx)

    orig_rank: Dict[str, Dict[str, int]] = {}
    orig_top: Dict[str, List[str]] = {}
    covered = 0
    surfaced = 0
    explained: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    for u in eval_users:
        ro, top = rank_and_top(ctx, u, topk)
        orig_rank[u], orig_top[u] = ro, top
        sm = soft_map(ctx, u)
        for i in top:
            surfaced += 1
            sup = supporting_lfi(ctx, sm, i, txns)
            if sup is not None:
                covered += 1
                ck, idx, _n, _p = sup
                explained.setdefault((ck, idx), []).append((u, i))

    coverage = covered / surfaced if surfaced else 0.0
    users_with_expl = len({u for pairs in explained.values() for u, _ in pairs})

    acc = {f"{tag}_{m}": [] for tag in ("lolo", "sup", "rnd")
           for m in ("drank", "drop", "fell", "jac")}

    groups = sorted(explained.items())
    for gi, ((ck, idx), pairs) in enumerate(groups, start=1):
        users = sorted({u for u, _ in pairs})
        lfi = ctx.cluster_models[ck].selected_lfis[idx]
        supporter_ids = [uid for uid, t in txns[ck].items() if lfi.issubset(t)]
        n_sup = len(supporter_ids)

        actx = ablate_lfi(ctx, ck, idx, txns[ck])
        _measure(actx, users, pairs, orig_rank, orig_top, topk, acc, "lolo")

        sctx = remove_users(ctx, ck, supporter_ids, users)
        _measure(sctx, users, pairs, orig_rank, orig_top, topk, acc, "sup")

        pool = list(txns[ck].keys())
        rng = np.random.default_rng((seed * 1_000_003 + zlib.crc32(f"{ck}|{idx}".encode())) % (2**32))
        k = min(n_sup, len(pool))
        if os.environ.get("POLARIS_PLACEBO_MATCH", "") in {"1", "true"} and k:
            sizes = {uid: len(t) for uid, t in txns[ck].items()}
            avail = sorted((uid for uid in pool if uid not in set(supporter_ids)),
                           key=lambda uid: (sizes[uid], uid))
            if len(avail) < k:
                avail = sorted(pool, key=lambda uid: (sizes[uid], uid))
            rnd_ids = []
            for s_sz in sorted(sizes[uid] for uid in supporter_ids)[:k]:
                best = min(avail, key=lambda uid: (abs(sizes[uid] - s_sz), uid))
                rnd_ids.append(best)
                avail.remove(best)
        else:
            rnd_ids = list(rng.choice(pool, size=k, replace=False)) if k else []
        rctx = remove_users(ctx, ck, rnd_ids, users)
        _measure(rctx, users, pairs, orig_rank, orig_top, topk, acc, "rnd")

        print(f"    [{seed}] group {gi}/{len(groups)} cluster={ck} lfi#{idx} "
              f"|U(P_t)|={n_sup} explained_pairs={len(pairs)}", flush=True)

    def agg(key):
        v = acc[key]
        if not v:
            return float("nan")
        return float(np.median(v)) if key.endswith("drank") else float(np.mean(v))

    return {
        "coverage": coverage, "surfaced": surfaced, "covered": covered,
        "users_with_expl": users_with_expl, "n_eval": len(eval_users),
        "n_explained_pairs": sum(len(p) for p in explained.values()),
        "n_groups": len(groups),
        **{k: agg(k) for k in acc},
    }


def _measure(pctx, users, pairs, orig_rank, orig_top, topk, acc, tag) -> None:
    new_rank: Dict[str, Dict[str, int]] = {}
    new_top: Dict[str, List[str]] = {}
    for u in users:
        ro, top = rank_and_top(pctx, u, topk)
        new_rank[u], new_top[u] = ro, top
    for u, i in pairs:
        r0 = orig_rank[u].get(i, BIG_RANK)
        r1 = new_rank[u].get(i)
        fell = r1 is None
        acc[f"{tag}_fell"].append(1.0 if fell else 0.0)
        if not fell:
            acc[f"{tag}_drank"].append(r1 - r0)
        acc[f"{tag}_drop"].append(1.0 if (i in orig_top[u] and i not in new_top[u]) else 0.0)
    for u in users:
        a, b = set(orig_top[u]), set(new_top[u])
        jac = len(a & b) / len(a | b) if (a | b) else 1.0
        acc[f"{tag}_jac"].append(jac)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seeds = list(range(1, nseeds + 1))

    cfg = load_params()
    shared = dict(cfg["shared"])
    targets = DATASETS if which == "all" else [(d, l) for d, l in DATASETS if d == which]

    results = []
    for ds_key, label in targets:
        run_cfg = next(r for r in cfg["main_results"] if r["dataset"] == ds_key)
        print(f"\n=== {label}  (alpha={run_cfg['hybrid_lambda']}, n_lfi={run_cfg['n_lfi']}, "
              f"seeds={seeds}, top-{topk}) ===", flush=True)
        per_seed = []
        for s in seeds:
            prep = build_prepared(ds_key, run_cfg, shared, s)
            per_seed.append(run_seed(prep, topk, s))

        results.append(summarize(label, topk, seeds, per_seed, ds_key))

    tested = [r for r in results if r["p2_drop"] == r["p2_drop"]]
    if len(tested) > 1:
        rej = holm([r["p2_drop"] for r in tested])
        print(f"\n=== Holm-Bonferroni over causal-excess dropout (family m={len(tested)}) ===")
        for r, ok in zip(tested, rej):
            print(f"  {r['label']:<10} excess {r['excess_mean']:+.3f}  "
                  f"two-sided p={r['p2_drop']:.2e}  Holm-significant: {'YES' if ok else 'no'}")


def _arr(per_seed, k):
    return np.array([r[k] for r in per_seed if r[k] == r[k]], dtype=float)


def _ms(per_seed, k):
    a = _arr(per_seed, k)
    return (float(np.mean(a)), float(np.std(a, ddof=1) if len(a) > 1 else 0.0)) if len(a) else (float("nan"), 0.0)


def _paired_excess(per_seed, sup_key, rnd_key):
    d = _arr(per_seed, sup_key) - _arr(per_seed, rnd_key)
    n = len(d)
    mean, std = float(np.mean(d)), float(np.std(d, ddof=1) if n > 1 else 0.0)
    p2 = p1 = float("nan")
    if n >= 2 and np.any(d != 0):
        try:
            p2 = float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
            p1 = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            pass
    return mean, std, p2, p1, n


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= 0.05 / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def summarize(label, topk, seeds, per_seed, ds_key) -> dict:
    cov_m, cov_s = _ms(per_seed, "coverage")
    print(f"\n  -- {label} summary over {len(seeds)} seed(s) --")
    print(f"  coverage@{topk}            : {cov_m:.3f} +/- {cov_s:.3f}")
    print(f"  users with >=1 expl     : {_ms(per_seed,'users_with_expl')[0]:.0f} / "
          f"{_ms(per_seed,'n_eval')[0]:.0f}    "
          f"explained pairs/seed: {_ms(per_seed,'n_explained_pairs')[0]:.0f}")
    print(f"  {'probe (dropout@%d)' % topk:<28}{'mean':>8}{'+/-std':>9}")
    for tag, name in (("lolo", "(A) leave-one-LFI-out"),
                      ("sup", "(B) leave-supporters-out"),
                      ("rnd", "(B) placebo random-users")):
        m, s = _ms(per_seed, tag + "_drop")
        print(f"  {name:<28}{m:>8.3f}{s:>9.3f}")

    em, es, ep2, ep1, en = _paired_excess(per_seed, "sup_drop", "rnd_drop")
    sig = "n/a" if ep2 != ep2 else (f"p={ep2:.2e}" + ("  *" if ep2 < 0.05 else ""))
    print(f"  => CAUSAL EXCESS dropout : {em:+.3f} +/- {es:.3f}   "
          f"Wilcoxon two-sided {sig}  (one-sided p={ep1:.2e})")

    fm, fs = _ms(per_seed, "sup_fell")
    rfm, _ = _ms(per_seed, "rnd_fell")
    fem, fes, fep2, fep1, _ = _paired_excess(per_seed, "sup_fell", "rnd_fell")
    fsig = "n/a" if fep2 != fep2 else f"p={fep2:.2e}"
    print(f"  [secondary] fell-out of candidate set (subset of dropout):")
    print(f"     supporters {fm:.3f} +/- {fs:.3f}  vs placebo {rfm:.3f}  "
          f"-> excess {fem:+.3f} +/- {fes:.3f}  Wilcoxon two-sided {fsig}")

    os.makedirs("outputs/explain_ablation", exist_ok=True)
    out = {
        "dataset": label, "seeds": seeds, "topk": topk,
        "coverage": {"mean": cov_m, "std": cov_s},
        "dropout": {t: dict(zip(("mean", "std"), _ms(per_seed, t + "_drop")))
                    for t in ("lolo", "sup", "rnd")},
        "causal_excess_dropout": {"mean": em, "std": es,
                                  "wilcoxon_p_2sided": ep2, "wilcoxon_p_1sided": ep1, "n": en},
        "fellout": {"sup": {"mean": fm, "std": fs}, "rnd": {"mean": rfm},
                    "excess": {"mean": fem, "std": fes,
                               "wilcoxon_p_2sided": fep2, "wilcoxon_p_1sided": fep1}},
        "per_seed": per_seed,
    }
    path = f"outputs/explain_ablation/{ds_key}_{len(seeds)}seeds_top{topk}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  written: {path}")
    return {"label": label, "excess_mean": em, "p2_drop": ep2, "p2_fell": fep2}


if __name__ == "__main__":
    main()
