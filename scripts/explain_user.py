from __future__ import annotations

import argparse
import dataclasses
import json
from typing import List, Optional

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig, load_params
from polaris.datasets import build_dataset
from polaris.experiment import Experiment
from polaris.explanation import explain_user, render_text, top_item_pattern_share


def _resolve_item(ctx, query: Optional[str]) -> Optional[str]:
    if not query:
        return None
    q = str(query)
    items = ctx.items
    if q in set(str(i) for i in items["itemId"].tolist()):
        return q
    if "title" in items.columns:
        hit = items[items["title"].astype(str).str.lower().str.contains(q.lower(), regex=False)]
        if len(hit):
            return str(hit.iloc[0]["itemId"])
    return None


def _auto_pick_user(ctx, candidates: List[str], limit: int = 150) -> Optional[str]:
    best_u: Optional[str] = None
    best_share = -1.0
    for u in candidates[:limit]:
        top, share = top_item_pattern_share(ctx, u)
        if top is not None and share > best_share:
            best_u, best_share = u, share
    return best_u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-100k", choices=["ml-100k", "ml-1m", "comoda"])
    ap.add_argument("--user", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--max-patterns", type=int, default=2)
    ap.add_argument("--why-not", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg_all = load_params()
    seed = int(args.seed)
    run_cfg = next(r for r in cfg_all["main_results"] if r["dataset"] == args.dataset)
    dataset = build_dataset(str(run_cfg["dataset"]))
    clusterer = AgglomerativeUserClusterer(
        demo_cols=[str(x) for x in run_cfg["demo_cols"]],
        n_clusters=int(run_cfg["n_clusters"]),
        linkage="ward",
        seed=seed,
    )
    experiment = Experiment(dataset=dataset, clusterer=clusterer)
    cfg = ExperimentConfig.from_dicts(cfg_all["shared"], run_cfg, n_lfi=int(run_cfg["n_lfi"]))
    prep = experiment.prepare_context(cfg, seed=seed, randomize_split=False)
    ctx = prep.ctx

    requested: List[str] = []
    if args.user:
        if str(args.user) not in ctx.test_assign:
            raise SystemExit(
                f"User {args.user} is not a cold/test user for {args.dataset} at seed {seed}."
            )
        requested.append(str(args.user))

    run_users = list(dict.fromkeys([str(u) for u in prep.eval_users] + requested))
    experiment._run_recommenders(ctx, run_users)

    target = requested[0] if requested else _auto_pick_user(ctx, [str(u) for u in prep.eval_users])
    if target is None:
        raise SystemExit("No cold user available to explain.")

    why_not = _resolve_item(ctx, args.why_not)
    extra = [why_not] if why_not else None
    expl = explain_user(
        ctx, target, top_k=int(args.top), max_patterns=int(args.max_patterns), extra_items=extra
    )

    if args.json:
        print(json.dumps(dataclasses.asdict(expl), indent=2))
    else:
        print(render_text(expl, dataset=str(run_cfg.get("dataset_label", args.dataset))))


if __name__ == "__main__":
    main()
