from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.metrics import evaluate_ranking, evaluate_set_mode_fair
from polaris.experiment import Experiment
from polaris.validation import _WarmOnlyDataset, _warm_ratings
from polaris.scoring import ScoreWeights, prepare_signals, user_signal_terms


def extract(ctx, users, test_pos, cap=0):
    prepared = prepare_signals(
        ctx.cluster_models, ctx.global_pop_full, ctx.per_cluster_pop_full, float(ctx.cfg.rrf_k)
    )
    weights = ScoreWeights.from_config(ctx.cfg)
    rrfk = float(ctx.cfg.rrf_k)
    out = []
    for u in users:
        terms, _, cands = user_signal_terms(
            u, prepared=prepared, test_assign=ctx.test_assign,
            test_soft_assign=ctx.test_soft_assign,
            feature_knn_rank=ctx.baseline_rankings.get("FeatureKNN", {}),
            context_pop_rank=ctx.baseline_rankings.get("ContextPopularity", {}),
            weights=weights, rrf_k=rrfk,
        )
        items = sorted(cands)
        if not items:
            continue
        if cap and len(items) > cap:
            blend = {it: sum(w * sc.get(it, 0.0) for (_, sc, w) in terms) for it in items}
            items = sorted(items, key=lambda it: (-blend[it], it))[:cap]
        X = np.array([[sc.get(it, 0.0) for (_, sc, _) in terms] for it in items], dtype=float)
        y = np.array([1.0 if it in test_pos.get(u, set()) else 0.0 for it in items], dtype=float)
        out.append((u, items, X, y))
    return out


def metrics(tp, sets_by_u, ranked_by_u):
    sm = evaluate_set_mode_fair(tp, sets_by_u)
    rm = evaluate_ranking(tp, ranked_by_u, ks=(10, 20, 50))
    return sm["f1"], sm["recall"], rm["ndcg@10"], rm["precision@10"], sm["precision"]


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    model = sys.argv[3] if len(sys.argv) > 3 else "lambdamart"
    seeds = list(range(1, nseeds + 1))
    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    run_cfg = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds))
    dataset = build_dataset(ds)
    nmax = int(shared["n_max"])
    print(f"[reranker] dataset={ds} seeds={nseeds} model={model} "
          f"(trained on inner fold)", flush=True)

    base_acc, rr_acc, dc_acc = [], [], []
    for s in seeds:
        cfg = ExperimentConfig.from_dicts(shared, run_cfg, n_lfi=int(run_cfg["n_lfi"]))
        _full, warm = _warm_ratings(dataset, shared, run_cfg, s)
        inner_ds = _WarmOnlyDataset(dataset.name, warm)
        inner_cl = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in run_cfg["demo_cols"]],
            n_clusters=int(run_cfg["n_clusters"]), linkage="ward", seed=s)
        exp_inner = Experiment(dataset=inner_ds, clusterer=inner_cl)
        prep_in = exp_inner.prepare_context(cfg, seed=s, randomize_split=True)
        exp_inner._run_recommenders(prep_in.ctx, prep_in.eval_users)
        train = extract(prep_in.ctx, prep_in.eval_users, prep_in.test_pos_by_user, cap=600)
        Xtr = np.vstack([t[2] for t in train]); ytr = np.concatenate([t[3] for t in train])
        groups = [len(t[1]) for t in train]

        if model == "lambdamart":
            import lightgbm as lgb
            ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, num_leaves=15,
                                    learning_rate=0.05, min_child_samples=20, label_gain=[0, 1],
                                    random_state=s, verbose=-1)
            ranker.fit(Xtr, (ytr > 0.5).astype(int), group=groups)
            predict = ranker.predict
        elif model == "linear":
            ranker = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            ranker.fit(Xtr, (ytr > 0.5).astype(int))
            predict = lambda X: ranker.predict_proba(X)[:, 1]
        else:
            ranker = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.1,
                                                    l2_regularization=1.0, random_state=s)
            ranker.fit(Xtr, ytr)
            predict = ranker.predict

        cl = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in run_cfg["demo_cols"]],
            n_clusters=int(run_cfg["n_clusters"]), linkage="ward", seed=s)
        exp = Experiment(dataset=dataset, clusterer=cl)
        prep = exp.prepare_context(cfg, seed=s, randomize_split=True)
        results = exp._run_recommenders(prep.ctx, prep.eval_users)
        tp = prep.test_pos_by_user
        base_acc.append(metrics(tp, results["POLARIS"].sets, results["POLARIS"].ranked))

        test = extract(prep.ctx, prep.eval_users, tp, cap=600)
        polaris_ranked = results["POLARIS"].ranked
        rr_ranked, rr_sets = {}, {}
        dc_ranked, dc_sets = {}, {}
        for u, items, X, _y in test:
            pred = predict(X)
            order = [items[i] for i in np.argsort(-pred)]
            rr_ranked[u] = order[:nmax]
            rr_sets[u] = set(order[:nmax])
            sset = rr_sets[u]
            base_ord = [it for it in polaris_ranked.get(u, []) if it in sset]
            dc_ranked[u] = base_ord + [it for it in order[:nmax] if it not in set(base_ord)]
            dc_sets[u] = sset
        rr_acc.append(metrics(tp, rr_sets, rr_ranked))
        dc_acc.append(metrics(tp, dc_sets, dc_ranked))

    ba = np.array(base_acc); base = ba.mean(0); rr = np.mean(rr_acc, axis=0); dc = np.array(dc_acc)
    print(f"\n{'method':<16}{'F1':>9}{'recall':>9}{'NDCG@10':>10}{'P@10':>9}")
    print(f"{'POLARIS blend':<16}{base[0]:>9.4f}{base[1]:>9.4f}{base[2]:>10.4f}{base[3]:>9.4f}")
    d = rr - base
    print(f"{'reranker':<16}{rr[0]:>9.4f}{rr[1]:>9.4f}{rr[2]:>10.4f}{rr[3]:>9.4f}"
          f"  (dF1{d[0]:+.4f} dNDCG{d[2]:+.4f})")
    dcm = dc.mean(0); dd = dcm - base
    df1 = dc[:, 0] - ba[:, 0]; dnd = dc[:, 2] - ba[:, 2]; dp10 = dc[:, 3] - ba[:, 3]
    print(f"{'rr-decoupled':<16}{dcm[0]:>9.4f}{dcm[1]:>9.4f}{dcm[2]:>10.4f}{dcm[3]:>9.4f}"
          f"  (dF1{dd[0]:+.4f} dR{dd[1]:+.4f} dNDCG{dd[2]:+.4f} dP10{dd[3]:+.4f})")
    print(f"   paired F1: W/L {int((df1>1e-9).sum())}/{int((df1<-1e-9).sum())}  "
          f"NDCG: W/L {int((dnd>1e-9).sum())}/{int((dnd<-1e-9).sum())}  (n={len(dc)})")

    def _wp(d):
        from scipy.stats import wilcoxon
        try:
            return float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")
        except ValueError:
            return float("nan")

    if len(seeds) > 1:
        print(f"   Wilcoxon two-sided (decoupled - POLARIS): F1 p={_wp(df1):.2e}  "
              f"NDCG@10 p={_wp(dnd):.2e}  P@10 p={_wp(dp10):.2e}", flush=True)
    os.makedirs("outputs/reranker", exist_ok=True)
    out = {
        "dataset": ds, "seeds": seeds, "model": model,
        "metric_names": ["f1", "recall", "ndcg@10", "precision@10", "precision"],
        "polaris_blend": {"mean": base.tolist(), "per_seed": ba.tolist()},
        "reranker_coupled": {"mean": np.asarray(rr).tolist()},
        "reranker_decoupled": {"mean": dcm.tolist(), "per_seed": dc.tolist(),
                               "delta_vs_blend": dd.tolist(),
                               "wilcoxon_p_2sided": {"f1": _wp(df1), "ndcg@10": _wp(dnd),
                                                     "precision@10": _wp(dp10)}},
    }
    path = f"outputs/reranker/reranker_{ds}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {path}", flush=True)


if __name__ == "__main__":
    main()
