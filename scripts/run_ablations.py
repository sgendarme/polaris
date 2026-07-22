from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.validation import nested_validation_scores

DATASETS = [("comoda", "LDOS"), ("ml-100k", "ML-100K"), ("ml-1m", "ML-1M")]
ETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
M_GRID = [1, 2, 3, 4]


def run20(ds_key, run_cfg, shared, seeds):
    dataset = build_dataset(ds_key)
    cl = AgglomerativeUserClusterer(
        demo_cols=[str(x) for x in run_cfg["demo_cols"]],
        n_clusters=int(run_cfg["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=dataset, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, run_cfg, n_lfi=int(run_cfg["n_lfi"]))
    res = exp.run(cfg, seeds=seeds, randomize_split=True)
    out = {"f1": [], "ndcg@10": [], "precision@10": []}
    for s in res["per_seed"]:
        out["f1"].append(float(s["set_metrics"]["POLARIS"]["f1"]))
        out["ndcg@10"].append(float(s["ranking_metrics"]["POLARIS"]["ndcg@10"]))
        out["precision@10"].append(float(s["ranking_metrics"]["POLARIS"]["precision@10"]))
    return out


def paired(ref, var):
    res = {}
    for m in ref:
        a, b = np.asarray(ref[m]), np.asarray(var[m])
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
        except ValueError:
            p = float("nan")
        res[m] = {"delta_mean": float(d.mean()), "win": int((d > 0).sum()),
                  "loss": int((d < 0).sum()), "wilcoxon_p_2sided": p}
    return res


def main() -> None:
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seeds = list(range(1, nseeds + 1))
    val_seeds = list(range(1, 6))

    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    os.makedirs("outputs/ablations", exist_ok=True)

    for ds_key, label in DATASETS:
        base = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
        out = {"dataset": label, "seeds": seeds}

        print(f"\n=== {label} phase 1: deployed / alpha0 / eta0 (20 seeds) ===", flush=True)
        ref = run20(ds_key, dict(base), shared, seeds)
        a0 = dict(base); a0["hybrid_lambda"] = 0.0
        va0 = run20(ds_key, a0, shared, seeds)
        e0 = dict(base); e0["feature_knn_weight"] = 0.0
        ve0 = run20(ds_key, e0, shared, seeds)
        out["deployed"] = {m: float(np.mean(v)) for m, v in ref.items()}
        out["alpha0"] = {"means": {m: float(np.mean(v)) for m, v in va0.items()},
                         "paired_vs_deployed": paired(ref, va0)}
        out["eta0"] = {"means": {m: float(np.mean(v)) for m, v in ve0.items()},
                       "paired_vs_deployed": paired(ref, ve0)}
        for tag in ("alpha0", "eta0"):
            d = out[tag]["paired_vs_deployed"]
            print(f"[{label}] {tag}: " + "  ".join(
                f"{m} {d[m]['delta_mean']:+.4f} ({d[m]['win']}/{d[m]['loss']}, p={d[m]['wilcoxon_p_2sided']:.1e})"
                for m in d), flush=True)

        print(f"=== {label} phase 2: tuned no-pattern ensemble (eta x M grid, val) ===", flush=True)
        dataset = build_dataset(ds_key)
        best = None
        for eta in ETA_GRID:
            for m_soft in M_GRID:
                rc = dict(base)
                rc["hybrid_lambda"] = 0.0
                rc["feature_knn_weight"] = eta
                rc["soft_clusters"] = m_soft
                sc = nested_validation_scores(dataset, shared, rc, val_seeds, target_method="POLARIS")
                if best is None or sc["ndcg@10"] > best[0]:
                    best = (sc["ndcg@10"], eta, m_soft)
        print(f"[{label}] ensemble selected: eta={best[1]} M={best[2]} (val NDCG {best[0]:.4f})",
              flush=True)
        ens = dict(base)
        ens["hybrid_lambda"] = 0.0
        ens["feature_knn_weight"] = best[1]
        ens["soft_clusters"] = best[2]
        vens = run20(ds_key, ens, shared, seeds)
        out["ensemble_nopattern"] = {
            "selected": {"eta": best[1], "M": best[2], "val_ndcg10": best[0]},
            "means": {m: float(np.mean(v)) for m, v in vens.items()},
            "paired_vs_deployed": paired(ref, vens)}
        d = out["ensemble_nopattern"]["paired_vs_deployed"]
        print(f"[{label}] ensemble: " + "  ".join(
            f"{m} {d[m]['delta_mean']:+.4f} ({d[m]['win']}/{d[m]['loss']}, p={d[m]['wilcoxon_p_2sided']:.1e})"
            for m in d), flush=True)

        print(f"=== {label} phase 3: split statistics ===", flush=True)
        cl = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in base["demo_cols"]],
            n_clusters=int(base["n_clusters"]), linkage="ward", seed=1)
        exp = Experiment(dataset=dataset, clusterer=cl)
        cfg = ExperimentConfig.from_dicts(shared, base, n_lfi=int(base["n_lfi"]))
        ncold, pos_per_user, top50_share, top5pct_share = [], [], [], []
        for s in seeds:
            prep = exp.prepare_context(cfg, seed=s, randomize_split=True)
            tp = prep.test_pos_by_user
            ncold.append(len(prep.eval_users))
            counts = [len(v) for v in tp.values()]
            pos_per_user.extend(counts)
            pop = prep.ctx.global_pop_full
            top50 = set(pop[:50])
            top5 = set(pop[:max(1, int(0.05 * len(pop)))])
            allpos = [i for v in tp.values() for i in v]
            if allpos:
                top50_share.append(sum(1 for i in allpos if i in top50) / len(allpos))
                top5pct_share.append(sum(1 for i in allpos if i in top5) / len(allpos))
        out["split_stats"] = {
            "cold_users_per_seed_mean": float(np.mean(ncold)),
            "positives_per_cold_user_mean": float(np.mean(pos_per_user)),
            "positives_per_cold_user_median": float(np.median(pos_per_user)),
            "share_positives_in_top50_items": float(np.mean(top50_share)),
            "share_positives_in_top5pct_items": float(np.mean(top5pct_share)),
        }
        print(f"[{label}] stats: {json.dumps(out['split_stats'])}", flush=True)

        path = f"outputs/ablations/{ds_key}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{label}] written {path}", flush=True)


if __name__ == "__main__":
    main()
