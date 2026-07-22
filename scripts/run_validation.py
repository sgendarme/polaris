from __future__ import annotations

import json
import os
import sys

from polaris.config import load_params
from polaris.datasets import build_dataset
from polaris.validation import coordinate_descent

def main() -> None:
    ds_key = sys.argv[1] if len(sys.argv) > 1 else "ml-100k"
    method = sys.argv[2] if len(sys.argv) > 2 else "POLARIS"
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nseeds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    seeds = list(range(1, nseeds + 1))

    cfg_all = load_params()
    shared = dict(cfg_all["shared"])
    if method in ("LightFM",):
        shared["include_lightfm"] = True
    if method in ("DropoutNet",):
        shared["include_dropoutnet"] = True
    if method in ("MeLU",):
        shared["include_melu"] = True
    run_cfg = dict(next(r for r in cfg_all["main_results"] if r["dataset"] == ds_key))

    dataset = build_dataset(ds_key)
    selected, scores, trials = coordinate_descent(
        dataset, shared, run_cfg, seeds, target_method=method, passes=passes)

    print(f"\nselected ({ds_key}/{method}): val NDCG@10={scores['ndcg@10']:.4f} "
          f"F1={scores['f1']:.4f}")
    changed = {k: v for k, v in selected.items()
               if k in run_cfg and run_cfg.get(k) != v or k not in run_cfg}
    print("changed axes:", {k: v for k, v in changed.items() if not isinstance(v, (list, dict))})

    os.makedirs("outputs/validation", exist_ok=True)
    path = f"outputs/validation/{ds_key}_{method}.json"
    json.dump({"dataset": ds_key, "method": method, "seeds": seeds,
               "scores": {"ndcg@10": scores["ndcg@10"], "f1": scores["f1"]},
               "selected": {k: v for k, v in selected.items()
                            if not isinstance(v, (list, dict))},
               "n_trials": len(trials)},
              open(path, "w"), indent=2)
    print("written", path)


if __name__ == "__main__":
    main()
