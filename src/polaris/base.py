from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Set

import pandas as pd

from polaris.config import ExperimentConfig
from polaris.method import ClusterModel


@dataclass
class RecommendContext:

    cfg: ExperimentConfig
    seed: int
    users: pd.DataFrame
    items: pd.DataFrame
    train_ratings: pd.DataFrame
    catalog: List[str]
    demo_cols: List[str]
    context_cols: List[str]
    test_assign: Dict[str, str]
    test_soft_assign: Dict[str, Dict[str, float]]
    cluster_models: Dict[str, ClusterModel]
    global_pop_full: List[str]
    per_cluster_pop_full: Dict[str, List[str]]
    cluster_pop_test_assign: Optional[Dict[str, str]] = None
    cluster_pop_cluster_col: str = "cluster_key"
    baseline_rankings: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


@dataclass
class RecResult:

    ranked: Dict[str, List[str]]
    sets: Dict[str, Set[str]]

    @staticmethod
    def from_ranked(ranked: Dict[str, List[str]]) -> "RecResult":
        return RecResult(ranked=ranked, sets={u: set(v) for u, v in ranked.items()})


class Recommender(Protocol):
    name: str

    def recommend(self, ctx: RecommendContext, cold_users: List[str]) -> RecResult:
        ...
