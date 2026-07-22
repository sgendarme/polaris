from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["ml-100k", "ml-1m", "comoda"]
METRICS = ["f1", "ndcg@10", "precision@10"]
SKIP = {"Random", "POLARIS-PatternOnly"}


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def main() -> None:
    rows = []
    for ds in DATASETS:
        path = f"outputs/final_tables/test20_{ds}.json"
        if not os.path.exists(path):
            print(f"missing {path} (run run_main.py first)")
            continue
        t = json.load(open(path))
        ref = t["per_seed"][t["deployed_ref"]]
        for base, row in t["per_seed"].items():
            if base == t["deployed_ref"] or base in SKIP or base == "POLARIS":
                continue
            for m in METRICS:
                a = np.asarray(ref[m], dtype=float)
                b = np.asarray(row[m], dtype=float)
                if np.isnan(a).any() or np.isnan(b).any():
                    continue
                d = a - b
                n = len(d)
                mean = float(d.mean())
                ci = 1.96 * float(d.std(ddof=1)) / math.sqrt(n)
                try:
                    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
                except ValueError:
                    p = float("nan")
                rows.append({
                    "dataset": ds, "baseline": base, "metric": m,
                    "delta_mean": mean, "ci95": ci,
                    "win": int((d > 0).sum()), "loss": int((d < 0).sum()), "p": p,
                })

    valid = [r for r in rows if not math.isnan(r["p"])]
    rej = holm([r["p"] for r in valid])
    for r, ok in zip(valid, rej):
        r["holm_significant"] = bool(ok)

    print(f"{'dataset':<9} {'baseline':<18} {'metric':<13} {'delta':>9} {'+-CI':>8} "
          f"{'W/L':>6} {'p':>10} {'Holm':>5}")
    for r in rows:
        star = "*" if r.get("holm_significant") else ""
        print(f"{r['dataset']:<9} {r['baseline']:<18} {r['metric']:<13} "
              f"{r['delta_mean']:>+9.4f} {r['ci95']:>8.4f} "
              f"{r['win']:>3}/{r['loss']:<2} {r['p']:>10.2e} {star:>5}")

    os.makedirs("outputs/final_tables", exist_ok=True)
    with open("outputs/final_tables/significance.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwritten outputs/final_tables/significance.json")


if __name__ == "__main__":
    main()
