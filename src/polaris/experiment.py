from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from polaris.clustering import AgglomerativeUserClusterer
from polaris.config import ExperimentConfig
from polaris.datasets import DatasetLoader
from polaris.metrics import evaluate_ranking, evaluate_set_mode_fair
from polaris.progress import log as _log
from polaris.base import RecommendContext, RecResult
from polaris.baselines import cluster_popularity_rankings
from polaris.method import fit_cluster_models
from polaris.recommenders import (
    AdaptivePOLARISRecommender,
    ClusterPopularityRecommender,
    ContextPopularityRecommender,
    DropoutNetRecommender,
    FeatureKNNRecommender,
    MeLURecommender,
    LightFMRecommender,
    PatternOnlyRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from polaris.splits import pick_cold_users_popular

ADAPTIVE_METHOD = "POLARIS"


def _build_recommenders(include_lightfm: bool = False, include_dropoutnet: bool = False,
                        include_melu: bool = False) -> List[Any]:
    recs: List[Any] = [
        AdaptivePOLARISRecommender(),
        PatternOnlyRecommender(),
        FeatureKNNRecommender(),
        ContextPopularityRecommender(),
        ClusterPopularityRecommender(),
        PopularityRecommender(),
        RandomRecommender(),
    ]
    if include_lightfm:
        recs.append(LightFMRecommender())
    if include_dropoutnet:
        recs.append(DropoutNetRecommender())
    if include_melu:
        recs.append(MeLURecommender())
    return recs


@dataclass
class RunResult:
    config: Dict[str, Any]
    set_metrics: Dict[str, Dict[str, float]]
    ranking_metrics: Dict[str, Dict[str, float]]
    per_cluster: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedContext:

    ctx: RecommendContext
    eval_users: List[str]
    test_pos_by_user: Dict[str, set]
    ctx_cols: List[str]
    num_cold_users: int
    num_ratings: int
    num_test_interactions: int


@dataclass
class Experiment:
    dataset: DatasetLoader
    clusterer: AgglomerativeUserClusterer

    def _make_split(self, ratings, cfg: ExperimentConfig, seed: int, randomize_split: bool, tag: str):
        t_stage = time.perf_counter()
        cold_users = pick_cold_users_popular(
            ratings,
            target_share=(float(cfg.cold_share[0]), float(cfg.cold_share[1])),
            topk_items=int(cfg.cold_popular_items),
            seed=int(seed),
            randomize=bool(randomize_split),
            min_cold_users=cfg.min_cold_users,
            mode=str(cfg.cold_split_mode),
        )
        train_r = ratings[~ratings["userId"].isin(cold_users)].copy()
        test_r = ratings[ratings["userId"].isin(cold_users)].copy()
        train_ids = set(train_r["userId"].unique())
        test_ids = set(test_r["userId"].unique())
        _log(
            f"{tag} cold-start split: {len(train_ids)} warm / {len(test_ids)} cold users "
            f"({time.perf_counter() - t_stage:.2f}s)"
        )
        return cold_users, train_r, test_r, train_ids, test_ids

    def _assign_clusters(self, users, train_ids, test_ids, cfg: ExperimentConfig, tag: str):
        t_stage = time.perf_counter()
        self.clusterer.fit(users, train_user_ids=train_ids)
        train_assign = self.clusterer.assign(users, train_ids)
        test_assign = self.clusterer.assign(users, test_ids)
        if int(cfg.soft_clusters) > 1 and hasattr(self.clusterer, "assign_soft"):
            test_soft_assign = self.clusterer.assign_soft(users, test_ids, top_m=int(cfg.soft_clusters))
        else:
            test_soft_assign = {str(u): {str(c): 1.0} for u, c in test_assign.items()}
        _log(
            f"{tag} clustering done: {self.clusterer.n_clusters} clusters "
            f"({time.perf_counter() - t_stage:.2f}s)"
        )
        return train_assign, test_assign, test_soft_assign

    def _run_recommenders(self, ctx: RecommendContext, eval_users: List[str]) -> Dict[str, RecResult]:
        recommenders = _build_recommenders(
            include_lightfm=bool(ctx.cfg.include_lightfm),
            include_dropoutnet=bool(ctx.cfg.include_dropoutnet),
            include_melu=bool(ctx.cfg.include_melu),
        )
        results: Dict[str, RecResult] = {}
        for rec in recommenders:
            if rec.name == ADAPTIVE_METHOD:
                continue
            res = rec.recommend(ctx, eval_users)
            results[rec.name] = res
            ctx.baseline_rankings[rec.name] = res.ranked
        adaptive = next(r for r in recommenders if r.name == ADAPTIVE_METHOD)
        results[ADAPTIVE_METHOD] = adaptive.recommend(ctx, eval_users)
        return {rec.name: results[rec.name] for rec in recommenders}

    def prepare_context(
        self,
        cfg: ExperimentConfig,
        *,
        seed: int,
        randomize_split: bool,
    ) -> PreparedContext:
        np.random.seed(seed)
        tag = f"[{self.dataset.name}|seed={seed}|n_lfi={cfg.n_lfi}]"
        _log(f"{tag} loading dataset ...")
        t_stage = time.perf_counter()
        data = self.dataset.load()
        ratings, users, items = data.ratings, data.users, data.items
        _log(
            f"{tag} loaded: {len(ratings)} ratings, {users['userId'].nunique()} users, "
            f"{items['itemId'].nunique()} items ({time.perf_counter() - t_stage:.2f}s)"
        )

        cold_users, train_r, test_r, train_ids, test_ids = self._make_split(
            ratings, cfg, seed, randomize_split, tag
        )
        train_assign, test_assign, test_soft_assign = self._assign_clusters(
            users, train_ids, test_ids, cfg, tag
        )

        train_r = train_r.copy()
        train_r["cluster_key"] = train_r["userId"].map(train_assign)
        train_r = train_r.dropna(subset=["cluster_key"])

        cp_test_assign = None
        cp_cluster_col = "cluster_key"
        cp_c = cfg.cluster_pop_n_clusters
        if cp_c is not None and int(cp_c) != int(self.clusterer.n_clusters):
            cp_clusterer = AgglomerativeUserClusterer(
                demo_cols=list(self.clusterer.demo_cols),
                n_clusters=int(cp_c),
                linkage=self.clusterer.linkage,
                seed=self.clusterer.seed,
            )
            cp_clusterer.fit(users, train_user_ids=train_ids)
            cp_train_assign = cp_clusterer.assign(users, train_ids)
            cp_test_assign = cp_clusterer.assign(users, test_ids)
            train_r["cluster_key_cp"] = train_r["userId"].map(cp_train_assign)
            cp_cluster_col = "cluster_key_cp"
            _log(f"{tag} ClusterPopularity uses its own clustering ({cp_clusterer.n_clusters} clusters)")

        _log(f"{tag} fitting per-cluster pattern models ...")
        t_stage = time.perf_counter()
        models = fit_cluster_models(
            train_df=train_r,
            cluster_col="cluster_key",
            tau_plus=cfg.tau_plus,
            tau_minus=cfg.tau_minus,
            theta=cfg.theta,
            max_k=cfg.max_k,
            n_lfi=cfg.n_lfi,
            delta=cfg.delta,
            n_max=cfg.n_max,
            log_prefix=tag,
            use_negative=cfg.use_negative,
            negative_penalty=cfg.negative_penalty,
            min_support_abs=int(cfg.min_support_abs),
            forbid_singleton_support=bool(cfg.forbid_singleton_support),
            auto_peeling=bool(cfg.auto_peeling),
            min_new_positive_items=int(cfg.min_new_positive_items),
            max_lfi_overlap=float(cfg.max_lfi_overlap),
            pop_rho=float(cfg.lfi_pop_rho),
        )
        _log(f"{tag} all clusters fitted ({time.perf_counter() - t_stage:.2f}s)")

        pos_test = test_r[test_r["rating"] >= cfg.tau_plus]
        test_pos_by_user: Dict[str, set] = {
            str(u): set(map(str, g["itemId"].unique())) for u, g in pos_test.groupby("userId")
        }
        eval_users = list(test_pos_by_user.keys())

        catalog = [str(i) for i in items["itemId"].unique().tolist()]
        ctx_cols = [str(c) for c in cfg.context_cols if str(c) in users.columns]
        global_pop_full, per_cluster_pop_full = cluster_popularity_rankings(train_r, cfg.tau_plus)
        ctx = RecommendContext(
            cfg=cfg,
            seed=seed,
            users=users,
            items=items,
            train_ratings=train_r,
            catalog=catalog,
            demo_cols=list(self.clusterer.demo_cols),
            context_cols=ctx_cols,
            test_assign=test_assign,
            test_soft_assign=test_soft_assign,
            cluster_models=models,
            global_pop_full=global_pop_full,
            per_cluster_pop_full=per_cluster_pop_full,
            cluster_pop_test_assign=cp_test_assign,
            cluster_pop_cluster_col=cp_cluster_col,
        )

        return PreparedContext(
            ctx=ctx,
            eval_users=eval_users,
            test_pos_by_user=test_pos_by_user,
            ctx_cols=ctx_cols,
            num_cold_users=len(cold_users),
            num_ratings=len(ratings),
            num_test_interactions=len(test_r),
        )

    def _run_single_seed(
        self,
        cfg: ExperimentConfig,
        *,
        seed: int,
        randomize_split: bool,
    ) -> RunResult:
        prep = self.prepare_context(cfg, seed=seed, randomize_split=randomize_split)
        ctx = prep.ctx
        eval_users = prep.eval_users
        tag = f"[{self.dataset.name}|seed={seed}|n_lfi={cfg.n_lfi}]"

        _log(f"{tag} evaluating {len(eval_users)} cold users ...")
        t_stage = time.perf_counter()
        results = self._run_recommenders(ctx, eval_users)
        set_metrics = {
            name: evaluate_set_mode_fair(prep.test_pos_by_user, res.sets) for name, res in results.items()
        }
        ranking_metrics = {
            name: evaluate_ranking(prep.test_pos_by_user, res.ranked, ks=cfg.ks)
            for name, res in results.items()
        }
        _log(
            f"{tag} evaluation done: POLARIS F1={set_metrics['POLARIS'].get('f1', 0):.4f} "
            f"NDCG@10={ranking_metrics['POLARIS'].get('ndcg@10', 0):.4f} "
            f"({time.perf_counter() - t_stage:.2f}s)"
        )

        per_cluster = [
            {
                "cluster": m.cluster_key,
                "train_users": m.n_train_users,
                "n_lfi_selected": len(m.selected_lfis),
                "rec_set_size": len(m.rec_set),
                "n_candidate_items": len(m.item_scores),
            }
            for m in sorted(ctx.cluster_models.values(), key=lambda x: x.cluster_key)
        ]
        config = {
            "dataset": self.dataset.name,
            "seed": seed,
            "n_clusters": self.clusterer.n_clusters,
            "num_cold_users": prep.num_cold_users,
            "num_eval_users": len(eval_users),
            "cold_interaction_share": float(prep.num_test_interactions / prep.num_ratings) if prep.num_ratings else 0.0,
        }
        return RunResult(
            config=config,
            set_metrics=set_metrics,
            ranking_metrics=ranking_metrics,
            per_cluster=per_cluster,
        )

    def run(
        self,
        cfg: ExperimentConfig,
        *,
        seeds: Sequence[int],
        randomize_split: bool = False,
    ) -> Dict[str, Any]:
        seeds = [int(s) for s in seeds]
        runs = [
            self._run_single_seed(cfg, seed=s, randomize_split=randomize_split)
            for s in seeds
        ]

        methods = list(runs[0].set_metrics.keys())

        def agg(get) -> Dict[str, Dict[str, Dict[str, float]]]:
            out: Dict[str, Dict[str, Dict[str, float]]] = {}
            for m in methods:
                out[m] = {}
                for key in get(runs[0])[m].keys():
                    vals = [get(r)[m][key] for r in runs]
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    out[m][key] = {"mean": mean, "std": std}
            return out

        return {
            "config": runs[0].config | {"seeds": seeds},
            "set_metrics_agg": agg(lambda r: r.set_metrics),
            "ranking_metrics_agg": agg(lambda r: r.ranking_metrics),
            "per_seed": [
                {
                    "seed": r.config["seed"],
                    "num_eval_users": r.config["num_eval_users"],
                    "set_metrics": r.set_metrics,
                    "ranking_metrics": r.ranking_metrics,
                }
                for r in runs
            ],
            "per_cluster_first_seed": runs[0].per_cluster,
        }
