from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig
from polaris.datasets.base import DatasetLoader, DatasetSplit
from polaris.splits import pick_cold_users_popular

from polaris.experiment import ADAPTIVE_METHOD, Experiment


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _WarmOnlyDataset:

    def __init__(self, name: str, split: DatasetSplit):
        self.name = name
        self._split = split

    def load(self) -> DatasetSplit:
        return self._split


def _warm_ratings(dataset: DatasetLoader, shared: Dict[str, Any], run_cfg: Dict[str, Any],
                  seed: int) -> Tuple[DatasetSplit, DatasetSplit]:
    full = dataset.load()
    cold = pick_cold_users_popular(
        full.ratings,
        target_share=(float(shared["cold_share_min"]), float(shared["cold_share_max"])),
        topk_items=int(shared["cold_popular_items"]),
        seed=int(seed),
        randomize=True,
        min_cold_users=run_cfg.get("min_cold_users", shared.get("min_cold_users")),
    )
    warm = full.ratings[~full.ratings["userId"].isin(cold)].copy()
    return full, DatasetSplit(ratings=warm, users=full.users, items=full.items)


def nested_validation_scores(
    dataset: DatasetLoader,
    shared: Dict[str, Any],
    run_cfg: Dict[str, Any],
    seeds: Sequence[int],
    *,
    ks: Tuple[int, ...] = (10, 20, 50),
    target_method: str = ADAPTIVE_METHOD,
) -> Dict[str, Any]:
    run_cfg = {k: v for k, v in run_cfg.items() if k != "cluster_pop_n_clusters"}
    per_method_nd: Dict[str, List[float]] = {}
    per_method_f1: Dict[str, List[float]] = {}
    n_val: List[int] = []
    for s in seeds:
        _full, warm_split = _warm_ratings(dataset, shared, run_cfg, s)
        wrapper = _WarmOnlyDataset(dataset.name, warm_split)
        clusterer = AgglomerativeUserClusterer(
            demo_cols=[str(x) for x in run_cfg["demo_cols"]],
            n_clusters=int(run_cfg["n_clusters"]),
            linkage="ward",
            seed=int(s),
        )
        experiment = Experiment(dataset=wrapper, clusterer=clusterer)
        cfg = ExperimentConfig.from_dicts(shared, run_cfg, n_lfi=int(run_cfg["n_lfi"]), ks=ks)
        res = experiment.run(cfg, seeds=[int(s)], randomize_split=True)
        for m, rank in res["ranking_metrics_agg"].items():
            per_method_nd.setdefault(m, []).append(float(rank["ndcg@10"]["mean"]))
            per_method_f1.setdefault(m, []).append(float(res["set_metrics_agg"][m]["f1"]["mean"]))
        n_val.append(int(res["config"]["num_eval_users"]))

    methods = {m: {"ndcg@10": float(np.mean(per_method_nd[m])),
                   "f1": float(np.mean(per_method_f1[m]))}
               for m in per_method_nd}
    tgt = methods[target_method]
    return {
        "ndcg@10": tgt["ndcg@10"],
        "f1": tgt["f1"],
        "methods": methods,
        "mean_val_users": float(np.mean(n_val)),
    }


_CLUSTER_GRID = {
    "ml-100k": [6, 8, 9, 10, 12, 15],
    "ml-1m": [10, 12, 15, 18, 20],
    "comoda": [3, 4, 5, 6, 8],
}
_AXIS_GRIDS: Dict[str, List[Any]] = {
    "theta": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    "max_k": [2, 3, 4, 5],
    "n_lfi": [1, 2, 3, 4, 5],
    "lfi_pop_rho": [0.0, 0.25, 0.5, 0.75, 1.0],
    "delta": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "soft_clusters": [1, 2, 3, 4],
    "hybrid_lambda": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
    "feature_knn_weight": [0.0, 0.1, 0.2, 0.3],
    "context_pop_weight": [0.0, 0.1, 0.2, 0.25],
    "knn_neighbors": [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500],
    "lightfm_components": [16, 32, 64, 128],
    "lightfm_lr": [0.01, 0.05, 0.1],
    "lightfm_epochs": [15, 30, 60],
    "dropoutnet_components": [16, 32, 64, 128],
    "dropoutnet_dropout": [0.3, 0.5, 0.7],
    "melu_components": [16, 32, 64],
    "melu_local_lr": [0.01, 0.05, 0.1],
}
METHOD_AXES: Dict[str, List[str]] = {
    "POLARIS": ["n_clusters", "theta", "max_k", "n_lfi", "lfi_pop_rho", "delta",
                "soft_clusters", "hybrid_lambda", "feature_knn_weight",
                "context_pop_weight"],
    "POLARIS-PatternOnly": ["theta", "max_k", "n_lfi", "delta"],
    "FeatureKNN": ["knn_neighbors"],
    "ClusterPopularity": ["n_clusters"],
    "LightFM": ["lightfm_components", "lightfm_lr", "lightfm_epochs"],
    "DropoutNet": ["dropoutnet_components", "dropoutnet_dropout"],
    "MeLU": ["melu_components", "melu_local_lr"],
}


def axes_for(target_method: str, dataset_name: str, run_cfg: Dict[str, Any],
             shared: Dict[str, Any]) -> List[Tuple[str, List[Any]]]:
    has_context = bool(run_cfg.get("context_cols") or shared.get("context_cols"))
    axes: List[Tuple[str, List[Any]]] = []
    for name in METHOD_AXES[target_method]:
        if name == "n_clusters":
            grid = list(_CLUSTER_GRID.get(dataset_name, [run_cfg["n_clusters"]]))
        else:
            grid = list(_AXIS_GRIDS[name])
        if name == "context_pop_weight" and not has_context:
            continue
        cur = run_cfg.get(name, shared.get(name))
        vals = list(grid)
        if cur is not None and cur not in vals:
            vals.append(cur)
        seen = set()
        vals = [v for v in vals if not (v in seen or seen.add(v))]
        try:
            vals = sorted(vals)
        except TypeError:
            pass
        axes.append((name, vals))
    return axes


@dataclass
class Trial:
    params: Dict[str, Any]
    ndcg10: float
    f1: float


def _better(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    if a[0] > b[0] + 1e-9:
        return True
    if abs(a[0] - b[0]) <= 1e-9 and a[1] > b[1] + 1e-9:
        return True
    return False


def coordinate_descent(
    dataset: DatasetLoader,
    shared: Dict[str, Any],
    base_run_cfg: Dict[str, Any],
    seeds: Sequence[int],
    *,
    target_method: str = ADAPTIVE_METHOD,
    passes: int = 2,
    log: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Trial]]:
    axes = axes_for(target_method, dataset.name, base_run_cfg, shared)
    cur = dict(base_run_cfg)
    cache: Dict[Tuple, Dict[str, Any]] = {}
    trials: List[Trial] = []
    axis_names = [a for a, _ in axes]

    def key(cfg: Dict[str, Any]) -> Tuple:
        return tuple(cfg.get(n, shared.get(n)) for n in axis_names)

    def evaluate(cfg: Dict[str, Any]) -> Dict[str, Any]:
        k = key(cfg)
        if k in cache:
            return cache[k]
        sc = nested_validation_scores(dataset, shared, cfg, seeds, target_method=target_method)
        cache[k] = sc
        trials.append(Trial(params={n: cfg.get(n, shared.get(n)) for n in axis_names},
                            ndcg10=sc["ndcg@10"], f1=sc["f1"]))
        return sc

    best = evaluate(cur)
    if log:
        _progress(f"[validate:{dataset.name}/{target_method}] start NDCG@10={best['ndcg@10']:.4f} "
                  f"F1={best['f1']:.4f} (val users~{best['mean_val_users']:.0f})")

    for p in range(1, passes + 1):
        improved = False
        for name, grid in axes:
            local_best_val = (best["ndcg@10"], best["f1"])
            local_best_choice = cur.get(name, shared.get(name))
            for v in grid:
                if v == cur.get(name, shared.get(name)):
                    continue
                trial_cfg = dict(cur)
                trial_cfg[name] = v
                sc = evaluate(trial_cfg)
                if _better((sc["ndcg@10"], sc["f1"]), local_best_val):
                    local_best_val = (sc["ndcg@10"], sc["f1"])
                    local_best_choice = v
                    best = sc
            if local_best_choice != cur.get(name, shared.get(name)):
                if log:
                    _progress(f"[validate:{dataset.name}] pass {p}: {name} "
                              f"{cur.get(name, shared.get(name))} -> {local_best_choice} "
                              f"(val NDCG@10={local_best_val[0]:.4f})")
                cur[name] = local_best_choice
                improved = True
        if not improved:
            if log:
                _progress(f"[validate:{dataset.name}] pass {p}: converged, no axis improved")
            break

    return cur, best, trials
