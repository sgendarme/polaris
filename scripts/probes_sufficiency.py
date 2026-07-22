from __future__ import annotations

import dataclasses
import json
import os
import sys
import zlib
from typing import Dict, List

import numpy as np

import probes_lolo_lsuo as E
from polaris.config import load_params
from polaris.method import ClusterModel, fit_cluster_models

SUB = "SUB"


def keep_substrate(ctx, keep_ids: List[str], cold_users: List[str]):
    keep = set(map(str, keep_ids))
    ptrain = ctx.train_ratings[ctx.train_ratings["userId"].astype(str).isin(keep)].copy()
    ptrain["cluster_key"] = SUB
    refit = fit_cluster_models(
        train_df=ptrain, cluster_col="cluster_key",
        tau_plus=ctx.cfg.tau_plus, tau_minus=ctx.cfg.tau_minus, theta=ctx.cfg.theta,
        max_k=ctx.cfg.max_k, n_lfi=ctx.cfg.n_lfi, delta=ctx.cfg.delta, n_max=ctx.cfg.n_max,
        use_negative=ctx.cfg.use_negative, negative_penalty=ctx.cfg.negative_penalty,
        min_support_abs=int(ctx.cfg.min_support_abs),
        forbid_singleton_support=bool(ctx.cfg.forbid_singleton_support),
        auto_peeling=bool(ctx.cfg.auto_peeling),
        min_new_positive_items=int(ctx.cfg.min_new_positive_items),
        max_lfi_overlap=float(ctx.cfg.max_lfi_overlap),
        pop_rho=float(ctx.cfg.lfi_pop_rho),
        global_pos_rate_override=(E.global_pos_rate_of(ctx, ptrain)
                                  if float(ctx.cfg.lfi_pop_rho) > 0 else None),
    )
    sub_model = refit.get(SUB) or ClusterModel(SUB, 0, [], set(), {}, negative_item_scores={})
    test_assign = dict(ctx.test_assign)
    test_soft = dict(ctx.test_soft_assign)
    for u in cold_users:
        test_assign[u] = SUB
        test_soft[str(u)] = {SUB: 1.0}
    pctx = dataclasses.replace(
        ctx,
        train_ratings=ptrain,
        cluster_models={SUB: sub_model},
        test_assign=test_assign,
        test_soft_assign=test_soft,
        baseline_rankings=dict(ctx.baseline_rankings),
    )
    E._populate_baselines(pctx, list(cold_users), ptrain)
    return pctx, sub_model


def _recovery(pctx, users, pairs, topk, acc, tag) -> None:
    new_top: Dict[str, set] = {}
    for u in users:
        _ro, top = E.rank_and_top(pctx, u, topk)
        new_top[u] = set(top)
    for u, i in pairs:
        acc[f"{tag}_rec"].append(1.0 if i in new_top[u] else 0.0)


def run_seed(prep, topk: int, seed: int) -> dict:
    ctx = prep.ctx
    eval_users = prep.eval_users
    txns = E.cluster_txns(ctx)

    surfaced = covered = 0
    explained: Dict[tuple, List[tuple]] = {}
    for u in eval_users:
        _ro, top = E.rank_and_top(ctx, u, topk)
        sm = E.soft_map(ctx, u)
        for i in top:
            surfaced += 1
            sup = E.supporting_lfi(ctx, sm, i, txns)
            if sup is not None:
                covered += 1
                ck, idx, _n, _p = sup
                explained.setdefault((ck, idx), []).append((u, i))

    coverage = covered / surfaced if surfaced else 0.0
    users_with_expl = len({u for pairs in explained.values() for u, _ in pairs})
    acc = {"named_rec": [], "rnd_rec": []}

    groups = sorted(explained.items())
    for gi, ((ck, idx), pairs) in enumerate(groups, start=1):
        users = sorted({u for u, _ in pairs})
        lfi = ctx.cluster_models[ck].selected_lfis[idx]
        supporter_ids = [uid for uid, t in txns[ck].items() if lfi.issubset(t)]
        n_sup = len(supporter_ids)

        nctx, _ = keep_substrate(ctx, supporter_ids, users)
        _recovery(nctx, users, pairs, topk, acc, "named")

        pool = list(txns[ck].keys())
        rng = np.random.default_rng((seed * 1_000_003 + zlib.crc32(f"{ck}|{idx}".encode())) % (2**32))
        k = min(n_sup, len(pool))
        rnd_ids = list(rng.choice(pool, size=k, replace=False)) if k else []
        rctx, _ = keep_substrate(ctx, rnd_ids, users)
        _recovery(rctx, users, pairs, topk, acc, "rnd")

        print(f"    [{seed}] group {gi}/{len(groups)} cluster={ck} lfi#{idx} "
              f"|U(P_t)|={n_sup} explained_pairs={len(pairs)}", flush=True)

    return {
        "coverage": coverage, "surfaced": surfaced, "covered": covered,
        "users_with_expl": users_with_expl, "n_eval": len(eval_users),
        "n_explained_pairs": sum(len(p) for p in explained.values()),
        "n_groups": len(groups),
        "named_rec": float(np.mean(acc["named_rec"])) if acc["named_rec"] else float("nan"),
        "rnd_rec": float(np.mean(acc["rnd_rec"])) if acc["rnd_rec"] else float("nan"),
    }


def summarize(label, topk, seeds, per_seed, ds_key) -> dict:
    cov_m, cov_s = E._ms(per_seed, "coverage")
    nm, ns = E._ms(per_seed, "named_rec")
    rm, rs = E._ms(per_seed, "rnd_rec")
    em, es, p2, p1, n = E._paired_excess(per_seed, "named_rec", "rnd_rec")

    print(f"\n  -- {label} sufficiency over {len(seeds)} seed(s) --")
    print(f"  coverage@{topk}             : {cov_m:.3f} +/- {cov_s:.3f}")
    print(f"  explained pairs/seed     : {E._ms(per_seed,'n_explained_pairs')[0]:.0f}   "
          f"distinct LFIs/seed: {E._ms(per_seed,'n_groups')[0]:.0f}")
    print(f"  recovery@{topk} named substrate : {nm:.3f} +/- {ns:.3f}   (raw is low/noisy by design: OOD)")
    print(f"  recovery@{topk} random placebo  : {rm:.3f} +/- {rs:.3f}")
    sig = "n/a" if p2 != p2 else (f"p={p2:.2e}" + ("  *" if p2 < 0.05 else ""))
    print(f"  => SUFFICIENCY EXCESS    : {em:+.3f} +/- {es:.3f}   "
          f"Wilcoxon two-sided {sig}  (one-sided p={p1:.2e})")

    os.makedirs("outputs/explain_ablation", exist_ok=True)
    out = {
        "dataset": label, "seeds": seeds, "topk": topk, "probe": "sufficiency",
        "coverage": {"mean": cov_m, "std": cov_s},
        "recovery": {"named": {"mean": nm, "std": ns}, "rnd": {"mean": rm, "std": rs}},
        "sufficiency_excess": {"mean": em, "std": es,
                               "wilcoxon_p_2sided": p2, "wilcoxon_p_1sided": p1, "n": n},
        "per_seed": per_seed,
    }
    path = f"outputs/explain_ablation/sufficiency_{ds_key}_{len(seeds)}seeds_top{topk}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  written: {path}")
    return {"label": label, "excess_mean": em, "p2": p2}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seeds = list(range(1, nseeds + 1))

    cfg = load_params()
    shared = dict(cfg["shared"])
    targets = E.DATASETS if which == "all" else [(d, l) for d, l in E.DATASETS if d == which]

    results = []
    for ds_key, label in targets:
        run_cfg = next(r for r in cfg["main_results"] if r["dataset"] == ds_key)
        print(f"\n=== {label} sufficiency (alpha={run_cfg['hybrid_lambda']}, "
              f"n_lfi={run_cfg['n_lfi']}, seeds={seeds}, top-{topk}) ===", flush=True)
        per_seed = [run_seed(E.build_prepared(ds_key, run_cfg, shared, s), topk, s) for s in seeds]
        results.append(summarize(label, topk, seeds, per_seed, ds_key))

    tested = [r for r in results if r["p2"] == r["p2"]]
    if len(tested) > 1:
        rej = E.holm([r["p2"] for r in tested])
        print(f"\n=== Holm-Bonferroni over sufficiency excess (family m={len(tested)}) ===")
        for r, ok in zip(tested, rej):
            print(f"  {r['label']:<10} excess {r['excess_mean']:+.3f}  "
                  f"two-sided p={r['p2']:.2e}  Holm-significant: {'YES' if ok else 'no'}")


if __name__ == "__main__":
    main()
