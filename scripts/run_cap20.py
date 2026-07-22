from __future__ import annotations

import json
import os
import sys

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment

DATASETS = ["ml-100k", "ml-1m", "comoda"]


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = list(range(1, nseeds + 1))
    targets = DATASETS if which == "all" else [which]

    cfg_all = load_params()
    os.makedirs("outputs/final_tables", exist_ok=True)

    for ds_key in targets:
        shared = dict(cfg_all["shared"])
        shared["n_max"] = 20
        shared["include_lightfm"] = True
        rc = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
        ds = build_dataset(ds_key)
        cl = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in rc["demo_cols"]],
            n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
        exp = Experiment(dataset=ds, clusterer=cl)
        cfg = ExperimentConfig.from_dicts(shared, rc, n_lfi=int(rc["n_lfi"]))
        print(f"=== {ds_key}: N=20 cap ({nseeds} seeds) ===", flush=True)
        res = exp.run(cfg, seeds=seeds, randomize_split=True)

        out: dict = {}
        for s in res["per_seed"]:
            for m in s["set_metrics"]:
                row = out.setdefault(m, {"f1": [], "precision": [], "ndcg@10": [], "precision@10": []})
                row["f1"].append(s["set_metrics"][m]["f1"])
                row["precision"].append(s["set_metrics"][m]["precision"])
                row["ndcg@10"].append(s["ranking_metrics"][m]["ndcg@10"])
                row["precision@10"].append(s["ranking_metrics"][m]["precision@10"])
        agg = {m: {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in row.items()}
               for m, row in out.items()}
        path = f"outputs/final_tables/cap20_{ds_key}.json"
        json.dump(agg, open(path, "w"), indent=2)
        print("written", path, {m: round(agg[m]["f1"][0], 4) for m in agg}, flush=True)


if __name__ == "__main__":
    main()
