from __future__ import annotations

import json
import os
import sys
import zlib
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

import probes_lolo_lsuo as E
from polaris.config import load_params


def participation(txns_ck: Dict[str, frozenset], user_ids: List[str],
                  q: Dict[str, float] | None = None, rho: float = 0.0) -> Dict[str, float]:
    n = 0
    cnt: Counter = Counter()
    for u in user_ids:
        t = txns_ck.get(u)
        if t is None:
            continue
        n += 1
        for tok in t:
            if tok and tok[-1] == "+":
                cnt[tok[:-1]] += 1
    if n == 0:
        return {}
    out = {}
    for it, c in cnt.items():
        v = c / n - (float(rho) * q.get(it, 0.0) if rho > 0 and q else 0.0)
        if v > 0:
            out[it] = v
    return out


def ranked_by_part(part: Dict[str, float]) -> List[str]:
    return [it for it, _ in sorted(part.items(), key=lambda kv: (-kv[1], kv[0]))]


def run_seed(prep, topk: int, seed: int) -> dict:
    ctx = prep.ctx
    pos_by = prep.test_pos_by_user
    eval_users = prep.eval_users
    txns = E.cluster_txns(ctx)

    explained: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    surfaced = covered = 0
    base_hits = base_tot = 0
    for u in eval_users:
        _ro, top = E.rank_and_top(ctx, u, topk)
        pos = pos_by.get(u, set())
        sm = E.soft_map(ctx, u)
        for i in top:
            surfaced += 1
            base_tot += 1
            if i in pos:
                base_hits += 1
            sup = E.supporting_lfi(ctx, sm, i, txns)
            if sup is not None:
                covered += 1
                ck, idx, _n, _p = sup
                explained.setdefault((ck, idx), []).append((u, i))

    coverage = covered / surfaced if surfaced else float("nan")
    base_rate = base_hits / base_tot if base_tot else float("nan")

    named_hits = named_tot = 0
    rand_hits = rand_tot = 0
    ak_named: List[float] = []
    ak_rnd: List[float] = []

    for (ck, idx), pairs in sorted(explained.items()):
        lfi = ctx.cluster_models[ck].selected_lfis[idx]
        supporter_ids = [uid for uid, t in txns[ck].items() if lfi.issubset(t)]
        n_sup = len(supporter_ids)

        rho = float(os.environ.get("POLARIS_ANCHOR_RHO", ctx.cfg.lfi_pop_rho))
        q = E.global_pos_rate_of(ctx, ctx.train_ratings) if rho > 0 else None
        named_rank = ranked_by_part(participation(txns[ck], supporter_ids, q, rho))

        pool = list(txns[ck].keys())
        rng = np.random.default_rng((seed * 1_000_003 + zlib.crc32(f"{ck}|{idx}".encode())) % (2**32))
        k = min(n_sup, len(pool))
        rnd_ids = list(rng.choice(pool, size=k, replace=False)) if k else []
        rand_rank = ranked_by_part(participation(txns[ck], rnd_ids, q, rho))

        by_user: Dict[str, List[str]] = {}
        for u, i in pairs:
            by_user.setdefault(u, []).append(i)

        for u, items in by_user.items():
            pos = pos_by.get(u, set())
            m = len(items)
            named_hits += sum(1 for i in items if i in pos)
            named_tot += m
            rand_hits += sum(1 for i in rand_rank[:m] if i in pos)
            rand_tot += m
            ak_named.append(len(set(named_rank[:topk]) & pos) / topk)
            ak_rnd.append(len(set(rand_rank[:topk]) & pos) / topk)

    return {
        "coverage": coverage, "base_rate": base_rate,
        "surfaced": surfaced, "covered": covered,
        "n_explained_pairs": named_tot, "n_groups": len(explained),
        "n_eval": len(eval_users),
        "prec_named": named_hits / named_tot if named_tot else float("nan"),
        "prec_rnd": rand_hits / rand_tot if rand_tot else float("nan"),
        "ak_named": float(np.mean(ak_named)) if ak_named else float("nan"),
        "ak_rnd": float(np.mean(ak_rnd)) if ak_rnd else float("nan"),
    }


def summarize(label, topk, seeds, per_seed, ds_key) -> dict:
    cov_m, cov_s = E._ms(per_seed, "coverage")
    br_m, br_s = E._ms(per_seed, "base_rate")
    pn_m, pn_s = E._ms(per_seed, "prec_named")
    pr_m, pr_s = E._ms(per_seed, "prec_rnd")
    em, es, p2, p1, n = E._paired_excess(per_seed, "prec_named", "prec_rnd")
    an_m, an_s = E._ms(per_seed, "ak_named")
    ar_m, ar_s = E._ms(per_seed, "ak_rnd")
    aem, aes, ap2, ap1, _ = E._paired_excess(per_seed, "ak_named", "ak_rnd")

    print(f"\n  -- {label} anchoring over {len(seeds)} seed(s) --")
    print(f"  coverage@{topk}              : {cov_m:.3f} +/- {cov_s:.3f}    "
          f"base rate (all recs) : {br_m:.3f}")
    print(f"  explained pairs/seed      : {E._ms(per_seed,'n_explained_pairs')[0]:.0f}")
    print(f"  [PRIMARY] anchoring precision (pattern-justified recs that are held-out positives):")
    print(f"     named  : {pn_m:.3f} +/- {pn_s:.3f}   random : {pr_m:.3f} +/- {pr_s:.3f}")
    sig = "n/a" if p2 != p2 else (f"p={p2:.2e}" + ("  *" if p2 < 0.05 else ""))
    print(f"     => ANCHORING EXCESS : {em:+.3f} +/- {es:.3f}   "
          f"Wilcoxon two-sided {sig}  (one-sided p={p1:.2e})")
    print(f"  [SECONDARY] anchoring@{topk} (raw top-K by participation, symmetric):")
    asig = "n/a" if ap2 != ap2 else (f"p={ap2:.2e}" + ("  *" if ap2 < 0.05 else ""))
    print(f"     named  : {an_m:.3f} +/- {an_s:.3f}   random : {ar_m:.3f} +/- {ar_s:.3f}"
          f"   excess {aem:+.3f} +/- {aes:.3f}  two-sided {asig}")

    os.makedirs("outputs/explain_ablation", exist_ok=True)
    out = {
        "dataset": label, "seeds": seeds, "topk": topk, "probe": "anchoring",
        "coverage": {"mean": cov_m, "std": cov_s},
        "base_rate": {"mean": br_m, "std": br_s},
        "anchoring_precision": {"named": {"mean": pn_m, "std": pn_s},
                                "rnd": {"mean": pr_m, "std": pr_s}},
        "anchoring_excess": {"mean": em, "std": es,
                             "wilcoxon_p_2sided": p2, "wilcoxon_p_1sided": p1, "n": n},
        "anchoring_at_k": {"named": {"mean": an_m, "std": an_s},
                           "rnd": {"mean": ar_m, "std": ar_s},
                           "excess": {"mean": aem, "std": aes,
                                      "wilcoxon_p_2sided": ap2, "wilcoxon_p_1sided": ap1}},
        "per_seed": per_seed,
    }
    path = f"outputs/explain_ablation/anchoring_{ds_key}_{len(seeds)}seeds_top{topk}.json"
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
        print(f"\n=== {label} anchoring (alpha={run_cfg['hybrid_lambda']}, "
              f"n_lfi={run_cfg['n_lfi']}, seeds={seeds}, top-{topk}) ===", flush=True)
        per_seed = [run_seed(E.build_prepared(ds_key, run_cfg, shared, s), topk, s) for s in seeds]
        results.append(summarize(label, topk, seeds, per_seed, ds_key))

    tested = [r for r in results if r["p2"] == r["p2"]]
    if len(tested) > 1:
        rej = E.holm([r["p2"] for r in tested])
        print(f"\n=== Holm-Bonferroni over anchoring-precision excess (family m={len(tested)}) ===")
        for r, ok in zip(tested, rej):
            print(f"  {r['label']:<10} excess {r['excess_mean']:+.3f}  "
                  f"two-sided p={r['p2']:.2e}  Holm-significant: {'YES' if ok else 'no'}")


if __name__ == "__main__":
    main()
