from __future__ import annotations

import os
import sys

os.environ["POLARIS_PLACEBO_MATCH"] = "1"

import probes_lolo_lsuo as E
from polaris.config import load_params

DEMO_ONLY = ["age", "gender", "city", "country", "age_bin"]


def main() -> None:
    ds = sys.argv[1] if len(sys.argv) > 1 else "ml-100k"
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    demo_only = (sys.argv[4] if len(sys.argv) > 4 else "0") in {"1", "true"}
    seeds = list(range(1, nseeds + 1))

    cfg = load_params()
    shared = dict(cfg["shared"])
    rc = dict(next(r for r in cfg["main_results"] if r["dataset"] == ds))
    tag = f"{ds}_matched"
    if demo_only:
        rc["demo_cols"] = DEMO_ONLY
        rc["context_cols"] = []
        tag = f"{ds}_demo_matched"

    print(f"=== {ds} LSUO with activity-matched placebo (seeds={seeds}) ===", flush=True)
    per_seed = [E.run_seed(E.build_prepared(ds, rc, shared, s), topk, s) for s in seeds]
    E.summarize(ds, topk, seeds, per_seed, tag)


if __name__ == "__main__":
    main()
