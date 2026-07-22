from __future__ import annotations

import json
import os
import sys

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.validation import coordinate_descent

DEMO_ONLY = ["age", "gender", "city", "country", "age_bin"]


def main() -> None:
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ml-100k"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = list(range(1, nseeds + 1))
    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    rc = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
    if ds_key == "comoda":
        rc["demo_cols"] = DEMO_ONLY
        rc["context_cols"] = []

    dataset = build_dataset(ds_key)
    sel, sc, trials = coordinate_descent(dataset, shared, rc, list(range(1, 6)),
                                         target_method="POLARIS-PatternOnly", passes=2)
    print(f"selected: theta={sel['theta']} max_k={sel['max_k']} n_lfi={sel['n_lfi']} "
          f"delta={sel['delta']}  val NDCG@10={sc['ndcg@10']:.4f}", flush=True)

    cl = AgglomerativeUserClusterer(demo_cols=[str(x) for x in rc["demo_cols"]],
                                    n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=dataset, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, sel, n_lfi=int(sel["n_lfi"]))
    res = exp.run(cfg, seeds=seeds, randomize_split=True)
    row = {k: [] for k in ["precision", "recall", "f1", "ndcg@10", "precision@10"]}
    for s in res["per_seed"]:
        for k in ["precision", "recall", "f1"]:
            row[k].append(float(s["set_metrics"]["POLARIS-PatternOnly"][k]))
        for k in ["ndcg@10", "precision@10"]:
            row[k].append(float(s["ranking_metrics"]["POLARIS-PatternOnly"][k]))
    means = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in row.items()}
    print("test means:", {k: round(v[0], 4) for k, v in means.items()}, flush=True)

    os.makedirs("outputs/controls", exist_ok=True)
    path = f"outputs/controls/patternonly_tuned_{ds_key}.json"
    json.dump({"dataset": ds_key, "selected": {k: sel[k] for k in
                                               ["theta", "max_k", "n_lfi", "delta"]},
               "val_ndcg10": sc["ndcg@10"], "n_trials": len(trials),
               "per_seed": row, "means": means}, open(path, "w"), indent=1)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
