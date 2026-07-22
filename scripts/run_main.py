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
SET_METRICS = ["precision", "recall", "f1"]
RANK_METRICS = [f"{m}@{k}" for k in (10, 20, 50)
                for m in ("ndcg", "precision", "recall")]
RERANKER_JSON = "outputs/reranker/reranker_ml-1m.json"


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    fast = len(sys.argv) > 3 and sys.argv[3] == "fast"
    seeds = list(range(1, nseeds + 1))

    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    shared["include_lightfm"] = not fast
    shared["include_dropoutnet"] = not fast
    shared["include_melu"] = not fast
    targets = DATASETS if which == "all" else [(d, l) for d, l in DATASETS if d == which]
    os.makedirs("outputs/final_tables", exist_ok=True)

    for ds_key, label in targets:
        run_cfg = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
        dataset = build_dataset(ds_key)
        cl = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in run_cfg["demo_cols"]],
            n_clusters=int(run_cfg["n_clusters"]), linkage="ward", seed=1)
        exp = Experiment(dataset=dataset, clusterer=cl)
        cfg = ExperimentConfig.from_dicts(shared, run_cfg, n_lfi=int(run_cfg["n_lfi"]))
        print(f"=== {label}: {nseeds}-seed test run (all methods) ===", flush=True)
        res = exp.run(cfg, seeds=seeds, randomize_split=True)

        per_seed: dict = {}
        for s in res["per_seed"]:
            for method in s["set_metrics"]:
                row = per_seed.setdefault(method, {m: [] for m in SET_METRICS + RANK_METRICS})
                for m in SET_METRICS:
                    row[m].append(float(s["set_metrics"][method][m]))
                for m in RANK_METRICS:
                    row[m].append(float(s["ranking_metrics"][method][m]))

        if ds_key == "ml-1m" and os.path.exists(RERANKER_JSON):
            rr = json.load(open(RERANKER_JSON))
            names = rr["metric_names"]
            arr = np.asarray(rr["reranker_decoupled"]["per_seed"])[: len(seeds)]
            per_seed["POLARIS+RR"] = {
                m: (arr[:, names.index(m)].tolist() if m in names
                    else [float("nan")] * len(seeds))
                for m in SET_METRICS + RANK_METRICS
            }

        ref_name = "POLARIS+RR" if "POLARIS+RR" in per_seed else "POLARIS"
        ref = per_seed[ref_name]
        tests: dict = {}
        for method, row in per_seed.items():
            if method == ref_name:
                continue
            tests[method] = {}
            for m in ["f1", "ndcg@10", "precision@10"]:
                a = np.asarray(ref[m]); b = np.asarray(row[m])
                if np.isnan(a).any() or np.isnan(b).any() or len(a) != len(b):
                    continue
                d = a - b
                try:
                    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
                except ValueError:
                    p = float("nan")
                tests[method][m] = {"delta_mean": float(d.mean()),
                                    "win": int((d > 0).sum()), "loss": int((d < 0).sum()),
                                    "wilcoxon_p_2sided": p}

        means = {meth: {m: float(np.nanmean(v)) for m, v in row.items()}
                 for meth, row in per_seed.items()}
        print(f"{'method':<20} {'F1':>8} {'NDCG@10':>9} {'P@10':>8}", flush=True)
        for meth in sorted(means, key=lambda x: -means[x]["ndcg@10"]):
            mm = means[meth]
            print(f"{meth:<20} {mm['f1']:>8.4f} {mm['ndcg@10']:>9.4f} {mm['precision@10']:>8.4f}",
                  flush=True)

        out = {"dataset": label, "seeds": seeds, "deployed_ref": ref_name,
               "means": means, "per_seed": per_seed, "paired_vs_ref": tests}
        path = (f"outputs/final_tables/test20_{ds_key}_fast.json" if fast
                else f"outputs/final_tables/test20_{ds_key}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{label}] written {path}\n", flush=True)


if __name__ == "__main__":
    main()
