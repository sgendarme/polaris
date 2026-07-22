from __future__ import annotations

import json
import os
import sys

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.explanation import top_item_pattern_share


def main() -> None:
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ml-100k"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    cfg_all = load_params()
    run_cfg = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
    dataset = build_dataset(ds_key)
    cl = AgglomerativeUserClusterer(
        demo_cols=[str(x) for x in run_cfg["demo_cols"]],
        n_clusters=int(run_cfg["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=dataset, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(cfg_all["shared"], run_cfg,
                                      n_lfi=int(run_cfg["n_lfi"]))

    shares = []
    for seed in range(1, nseeds + 1):
        prep = exp.prepare_context(cfg, seed=seed, randomize_split=True)
        ctx = prep.ctx
        users = [str(u) for u in prep.eval_users]
        exp._run_recommenders(ctx, users)
        for u in users:
            top, share = top_item_pattern_share(ctx, u)
            if top is not None:
                shares.append(float(share))
        print(f"seed {seed}: {len(shares)} users cumulated", flush=True)

    arr = np.asarray(shares)
    out = {"dataset": ds_key, "seeds": list(range(1, nseeds + 1)),
           "n_users": int(arr.size),
           "median": float(np.median(arr)), "mean": float(arr.mean()),
           "q25": float(np.quantile(arr, 0.25)), "q75": float(np.quantile(arr, 0.75))}
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/pattern_share_{ds_key}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
