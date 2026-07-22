from __future__ import annotations

from dataclasses import dataclass
from typing import List

from polaris.base import RecommendContext, RecResult
from polaris.baselines import (
    cluster_popularity_recommender,
    context_popularity_recommender,
    feature_knn_recommender,
    popularity_recommender,
    random_recommender,
)
from polaris.dropoutnet_baseline import dropoutnet_recommender
from polaris.melu_baseline import melu_recommender
from polaris.lightfm_baseline import lightfm_recommender
from polaris.scoring import ScoreWeights, adaptive_rankings


@dataclass
class RandomRecommender:
    name: str = "Random"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = random_recommender(cold_users, ctx.catalog, ctx.cfg.n_max, ctx.seed)
        return RecResult.from_ranked(ranked)


@dataclass
class PopularityRecommender:
    name: str = "Popularity"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = popularity_recommender(cold_users, ctx.train_ratings, ctx.cfg.tau_plus, ctx.cfg.n_max)
        return RecResult.from_ranked(ranked)


@dataclass
class ClusterPopularityRecommender:
    name: str = "ClusterPopularity"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        test_assign = ctx.cluster_pop_test_assign if ctx.cluster_pop_test_assign is not None else ctx.test_assign
        ranked = cluster_popularity_recommender(
            cold_users, ctx.train_ratings, test_assign, ctx.cfg.tau_plus, ctx.cfg.n_max,
            cluster_col=ctx.cluster_pop_cluster_col,
        )
        return RecResult.from_ranked(ranked)


@dataclass
class FeatureKNNRecommender:
    name: str = "FeatureKNN"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = feature_knn_recommender(
            cold_users,
            users=ctx.users,
            train_ratings=ctx.train_ratings,
            demo_cols=ctx.demo_cols,
            rating_thr=ctx.cfg.tau_plus,
            n_max=ctx.cfg.n_max,
            k_neighbors=int(ctx.cfg.knn_neighbors),
            global_fallback=ctx.global_pop_full,
        )
        return RecResult.from_ranked(ranked)


@dataclass
class ContextPopularityRecommender:
    name: str = "ContextPopularity"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = context_popularity_recommender(
            cold_users,
            users=ctx.users,
            train_ratings=ctx.train_ratings,
            context_cols=ctx.context_cols,
            rating_thr=ctx.cfg.tau_plus,
            n_max=ctx.cfg.n_max,
            global_fallback=ctx.global_pop_full,
        )
        return RecResult.from_ranked(ranked)


@dataclass
class LightFMRecommender:

    name: str = "LightFM"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = lightfm_recommender(
            cold_users,
            users=ctx.users,
            train_ratings=ctx.train_ratings,
            catalog=ctx.catalog,
            demo_cols=ctx.demo_cols,
            rating_thr=ctx.cfg.tau_plus,
            n_max=ctx.cfg.n_max,
            no_components=int(ctx.cfg.lightfm_components),
            epochs=int(ctx.cfg.lightfm_epochs),
            learning_rate=float(ctx.cfg.lightfm_lr),
            seed=int(ctx.seed),
            global_fallback=ctx.global_pop_full,
        )
        return RecResult.from_ranked(ranked)


@dataclass
class DropoutNetRecommender:

    name: str = "DropoutNet"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = dropoutnet_recommender(
            cold_users,
            users=ctx.users,
            train_ratings=ctx.train_ratings,
            catalog=ctx.catalog,
            demo_cols=ctx.demo_cols,
            rating_thr=ctx.cfg.tau_plus,
            n_max=ctx.cfg.n_max,
            components=int(ctx.cfg.dropoutnet_components),
            dropout=float(ctx.cfg.dropoutnet_dropout),
            epochs=int(ctx.cfg.dropoutnet_epochs),
            seed=int(ctx.seed),
            global_fallback=ctx.global_pop_full,
        )
        return RecResult.from_ranked(ranked)


class MeLURecommender:

    name: str = "MeLU"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = melu_recommender(
            cold_users,
            users=ctx.users,
            train_ratings=ctx.train_ratings,
            catalog=ctx.catalog,
            demo_cols=ctx.demo_cols,
            n_max=ctx.cfg.n_max,
            components=int(ctx.cfg.melu_components),
            local_lr=float(ctx.cfg.melu_local_lr),
            epochs=int(ctx.cfg.melu_epochs),
            seed=int(ctx.seed),
            rating_thr=ctx.cfg.tau_plus,
            global_fallback=ctx.global_pop_full,
        )
        return RecResult.from_ranked(ranked)


@dataclass
class PatternOnlyRecommender:

    name: str = "POLARIS-PatternOnly"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ranked = {}
        sets = {}
        for u in cold_users:
            ck = ctx.test_assign.get(u)
            model = ctx.cluster_models.get(str(ck)) if ck is not None else None
            ranked[u] = list(model.ranked_items) if model else []
            sets[u] = set(model.rec_set) if model else set()
        return RecResult(ranked=ranked, sets=sets)


@dataclass
class AdaptivePOLARISRecommender:

    name: str = "POLARIS"

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        return adaptive_rankings(
            eval_users=cold_users,
            test_assign=ctx.test_assign,
            test_soft_assign=ctx.test_soft_assign,
            cluster_models=ctx.cluster_models,
            global_pop_full=ctx.global_pop_full,
            per_cluster_pop_full=ctx.per_cluster_pop_full,
            feature_knn_rank=ctx.baseline_rankings.get("FeatureKNN", {}),
            context_pop_rank=ctx.baseline_rankings.get("ContextPopularity", {}),
            weights=ScoreWeights.from_config(ctx.cfg),
            rrf_k=float(ctx.cfg.rrf_k),
            n_max=ctx.cfg.n_max,
        )
