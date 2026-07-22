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

METRICS = ["precision", "recall", "f1", "ndcg@10", "precision@10"]


def main() -> None:
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seeds = list(range(1, nseeds + 1))
    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    shared["include_lightfm"] = True
    shared["include_dropoutnet"] = True
    shared["include_melu"] = True
    rc = dict(cfg_all["context_diagnostic"])

    ds = build_dataset("comoda")
    cl = AgglomerativeUserClusterer(demo_cols=[str(x) for x in rc["demo_cols"]],
                                    n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=ds, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, rc, n_lfi=int(rc["n_lfi"]))
    print(f"=== LDOS context diagnostic ({nseeds} seeds) ===", flush=True)
    res = exp.run(cfg, seeds=seeds, randomize_split=True)

    per_seed: dict = {}
    for s in res["per_seed"]:
        for m in s["set_metrics"]:
            row = per_seed.setdefault(m, {k: [] for k in METRICS})
            for k in ["precision", "recall", "f1"]:
                row[k].append(float(s["set_metrics"][m][k]))
            for k in ["ndcg@10", "precision@10"]:
                row[k].append(float(s["ranking_metrics"][m][k]))

    means = {m: {k: float(np.mean(v)) for k, v in row.items()} for m, row in per_seed.items()}
    print(f"{'method':<20} {'F1':>8} {'NDCG@10':>9} {'P@10':>8}", flush=True)
    for m in sorted(means, key=lambda x: -means[x]["ndcg@10"]):
        mm = means[m]
        print(f"{m:<20} {mm['f1']:>8.4f} {mm['ndcg@10']:>9.4f} {mm['precision@10']:>8.4f}",
              flush=True)

    ref = per_seed["POLARIS"]
    tests = {}
    for m, row in per_seed.items():
        if m == "POLARIS":
            continue
        tests[m] = {}
        for k in ["f1", "ndcg@10", "precision@10"]:
            a, b = np.asarray(ref[k]), np.asarray(row[k])
            d = a - b
            try:
                p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
            except ValueError:
                p = float("nan")
            tests[m][k] = {"delta_mean": float(d.mean()), "win": int((d > 0).sum()),
                           "loss": int((d < 0).sum()), "wilcoxon_p_2sided": p}

    os.makedirs("outputs/ldos_diagnostic", exist_ok=True)
    path = "outputs/ldos_diagnostic/context_diagnostic.json"
    json.dump({"seeds": seeds, "means": means, "per_seed": per_seed,
               "paired_vs_polaris": tests}, open(path, "w"), indent=1)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
