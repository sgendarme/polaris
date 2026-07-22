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

ENSEMBLE_SEL = {"ml-100k": {"eta": 0.5, "M": 4}, "ml-1m": {"eta": 0.1, "M": 2}}
METRICS = ["precision", "recall", "f1", "ndcg@10", "precision@10"]


def run_polaris_row(ds_key, rc, shared, seeds):
    ds = build_dataset(ds_key)
    cl = AgglomerativeUserClusterer(demo_cols=[str(x) for x in rc["demo_cols"]],
                                    n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=ds, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, rc, n_lfi=int(rc["n_lfi"]))
    res = exp.run(cfg, seeds=seeds, randomize_split=True)
    row = {k: [] for k in METRICS}
    for s in res["per_seed"]:
        for k in ["precision", "recall", "f1"]:
            row[k].append(float(s["set_metrics"]["POLARIS"][k]))
        for k in ["ndcg@10", "precision@10"]:
            row[k].append(float(s["ranking_metrics"]["POLARIS"][k]))
    return row


def paired(ref, other, keys):
    out = {}
    for k in keys:
        a, b = np.asarray(ref[k], dtype=float), np.asarray(other[k], dtype=float)
        if np.isnan(a).any():
            continue
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
        except ValueError:
            p = float("nan")
        out[k] = {"delta_mean": float(d.mean()),
                  "ci95": float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                  "win": int((d > 0).sum()), "loss": int((d < 0).sum()), "p": p}
    return out


def main() -> None:
    ds_key = sys.argv[1]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = list(range(1, nseeds + 1))
    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    base = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))

    ref_all = json.load(open(f"outputs/final_tables/test20_{ds_key}.json"))
    ref = ref_all["per_seed"][ref_all["deployed_ref"]]

    out = {"dataset": ds_key, "seeds": seeds}
    for name, patch in [
        ("softclusterpop", {"hybrid_lambda": 0.0, "feature_knn_weight": 0.0,
                            "context_pop_weight": 0.0, "personalized_negative_weight": 0.0}),
        ("ensemble", {"hybrid_lambda": 0.0,
                      "feature_knn_weight": ENSEMBLE_SEL[ds_key]["eta"],
                      "soft_clusters": ENSEMBLE_SEL[ds_key]["M"]}),
    ]:
        print(f"=== {ds_key} {name} ===", flush=True)
        rc = dict(base); rc.update(patch)
        row = run_polaris_row(ds_key, rc, shared, seeds)
        out[f"{name}_per_seed"] = row
        out[f"{name}_means"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))]
                                for k, v in row.items()}
        keys = [k for k in ["f1", "ndcg@10", "precision@10"]]
        out[f"paired_{name}"] = paired(ref, row, keys)

    os.makedirs("outputs/controls", exist_ok=True)
    path = f"outputs/controls/softpop_ensemble_{ds_key}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
