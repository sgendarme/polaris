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

DATASETS = [("comoda", "LDOS"), ("ml-100k", "ML-100K"), ("ml-1m", "ML-1M")]
METHODS = ["POLARIS", "FeatureKNN", "DropoutNet", "LightFM", "ClusterPopularity",
           "ContextPopularity", "Popularity"]


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = list(range(1, nseeds + 1))
    modes = ["random", "low_activity"] if which == "all" else [which]

    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    shared["include_lightfm"] = True
    shared["include_dropoutnet"] = True
    os.makedirs("outputs/robust_splits", exist_ok=True)

    for mode in modes:
        for ds_key, label in DATASETS:
            rc = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
            rc["cold_split_mode"] = mode
            rc["dropoutnet_components"] = 64
            rc["dropoutnet_dropout"] = 0.5

            dataset = build_dataset(ds_key)
            cl = AgglomerativeUserClusterer(
                demo_cols=[str(x) for x in rc["demo_cols"]],
                n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
            exp = Experiment(dataset=dataset, clusterer=cl)
            cfg = ExperimentConfig.from_dicts(shared, rc, n_lfi=int(rc["n_lfi"]))
            print(f"\n=== {label} / {mode} split ({nseeds} seeds) ===", flush=True)
            res = exp.run(cfg, seeds=seeds, randomize_split=True)

            per_seed: dict = {}
            ncold, shares = [], []
            for s in res["per_seed"]:
                for m in s["set_metrics"]:
                    row = per_seed.setdefault(m, {"f1": [], "ndcg@10": [], "precision@10": []})
                    row["f1"].append(float(s["set_metrics"][m]["f1"]))
                    row["ndcg@10"].append(float(s["ranking_metrics"][m]["ndcg@10"]))
                    row["precision@10"].append(float(s["ranking_metrics"][m]["precision@10"]))
            ncold.append(float(res["config"].get("num_cold_users", float("nan"))))
            shares.append(float(res["config"].get("cold_interaction_share", float("nan"))))

            means = {m: {k: float(np.mean(v)) for k, v in row.items()}
                     for m, row in per_seed.items()}
            print(f"{'method':<18} {'F1':>8} {'NDCG@10':>9} {'P@10':>8}", flush=True)
            for m in METHODS:
                if m in means:
                    mm = means[m]
                    print(f"{m:<18} {mm['f1']:>8.4f} {mm['ndcg@10']:>9.4f} "
                          f"{mm['precision@10']:>8.4f}", flush=True)

            ref = per_seed["POLARIS"]
            tests = {}
            for m in METHODS:
                if m == "POLARIS" or m not in per_seed:
                    continue
                tests[m] = {}
                for k in ["f1", "ndcg@10", "precision@10"]:
                    a, b = np.asarray(ref[k]), np.asarray(per_seed[m][k])
                    d = a - b
                    try:
                        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
                    except ValueError:
                        p = float("nan")
                    tests[m][k] = {"delta_mean": float(d.mean()), "win": int((d > 0).sum()),
                                   "loss": int((d < 0).sum()), "wilcoxon_p_2sided": p}

            out = {"dataset": label, "mode": mode, "seeds": seeds,
                   "cold_users_mean": float(np.nanmean(ncold)),
                   "removed_share_mean": float(np.nanmean(shares)),
                   "means": means, "per_seed": per_seed, "paired_vs_polaris": tests}
            path = f"outputs/robust_splits/{mode}_{ds_key}.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[{label}/{mode}] written {path}", flush=True)


if __name__ == "__main__":
    main()
