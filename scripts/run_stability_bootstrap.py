from __future__ import annotations

import json
import os
import sys

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.method import (
    encode_signed_transactions,
    is_positive,
    select_lfis_by_peeling,
    strip_sign,
)

DEMO_ONLY = ["age", "gender", "city", "country", "age_bin"]


def pos_items(lfis):
    return {strip_sign(t) for p in lfis for t in p if is_positive(t)}


def main() -> None:
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ml-100k"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    rc = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))
    if ds_key == "comoda":
        rc["demo_cols"] = DEMO_ONLY
        rc["context_cols"] = []

    ds = build_dataset(ds_key)
    cl = AgglomerativeUserClusterer(demo_cols=[str(x) for x in rc["demo_cols"]],
                                    n_clusters=int(rc["n_clusters"]), linkage="ward", seed=1)
    exp = Experiment(dataset=ds, clusterer=cl)
    cfg = ExperimentConfig.from_dicts(shared, rc, n_lfi=int(rc["n_lfi"]))
    prep = exp.prepare_context(cfg, seed=1, randomize_split=True)
    train = prep.ctx.train_ratings

    jaccs = []
    for ck, block in train.groupby("cluster_key", sort=False):
        tx = encode_signed_transactions(block, cfg.tau_plus, cfg.tau_minus, cfg.use_negative)
        if not tx:
            continue
        orig = pos_items(select_lfis_by_peeling(
            tx, cfg.theta, cfg.max_k, cfg.n_lfi,
            min_support_abs=cfg.min_support_abs,
            forbid_singleton_support=cfg.forbid_singleton_support,
            auto_peeling=cfg.auto_peeling,
            min_new_positive_items=cfg.min_new_positive_items,
            max_lfi_overlap=cfg.max_lfi_overlap))
        if not orig:
            continue
        uids = list(tx.keys())
        rng = np.random.default_rng(1234)
        for _ in range(B):
            sample = rng.choice(uids, size=len(uids), replace=True)
            btx = {f"b{i}": tx[u] for i, u in enumerate(sample)}
            boot = pos_items(select_lfis_by_peeling(
                btx, cfg.theta, cfg.max_k, cfg.n_lfi,
                min_support_abs=cfg.min_support_abs,
                forbid_singleton_support=cfg.forbid_singleton_support,
                auto_peeling=cfg.auto_peeling,
                min_new_positive_items=cfg.min_new_positive_items,
                max_lfi_overlap=cfg.max_lfi_overlap))
            inter = len(orig & boot)
            union = len(orig | boot) or 1
            jaccs.append(inter / union)
    print(f"{ds_key}: bootstrap LFI stability (positive-token Jaccard) "
          f"= {np.mean(jaccs):.3f} +/- {np.std(jaccs):.3f}  (n={len(jaccs)})", flush=True)
    os.makedirs("outputs", exist_ok=True)
    out = {"dataset": ds_key, "bootstrap_resamples": B,
           "positive_token_jaccard_mean": float(np.mean(jaccs)),
           "positive_token_jaccard_std": float(np.std(jaccs)),
           "n": len(jaccs)}
    with open(f"outputs/stability_bootstrap_{ds_key}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
