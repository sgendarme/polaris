from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "conf" / "params.json"


@dataclass(frozen=True)
class ExperimentConfig:

    tau_plus: float
    tau_minus: float
    theta: float
    delta: float
    max_k: int
    n_lfi: int
    n_max: int
    cold_share: Tuple[float, float] = (0.20, 0.30)
    cold_popular_items: int = 50
    ks: Tuple[int, ...] = (10, 20, 50)
    min_cold_users: Optional[int] = None
    cluster_pop_n_clusters: Optional[int] = None

    use_negative: bool = True
    negative_penalty: float = 0.0
    min_support_abs: int = 1
    forbid_singleton_support: bool = False
    auto_peeling: bool = False
    min_new_positive_items: int = 1
    max_lfi_overlap: float = 1.0
    lfi_pop_rho: float = 0.0
    cold_split_mode: str = "popular"

    hybrid_lambda: float = 0.35
    soft_clusters: int = 1
    global_pop_penalty: float = 0.0
    rrf_k: float = 60.0
    knn_neighbors: int = 50
    feature_knn_weight: float = 0.0
    context_pop_weight: float = 0.0
    context_cols: Tuple[str, ...] = ()
    personalized_negative_weight: float = 0.0

    include_lightfm: bool = False
    lightfm_components: int = 64
    lightfm_epochs: int = 30
    lightfm_lr: float = 0.05

    include_dropoutnet: bool = False
    dropoutnet_components: int = 64
    dropoutnet_dropout: float = 0.5
    dropoutnet_epochs: int = 30

    include_melu: bool = False
    melu_components: int = 32
    melu_local_lr: float = 0.1
    melu_epochs: int = 40

    @staticmethod
    def from_dicts(
        shared: Dict[str, Any],
        run_cfg: Dict[str, Any],
        *,
        n_lfi: int,
        ks: Sequence[int] = (10, 20, 50),
    ) -> "ExperimentConfig":

        def pick(key: str, default: Any) -> Any:
            return run_cfg.get(key, shared.get(key, default))

        return ExperimentConfig(
            tau_plus=float(shared["tau_plus"]),
            tau_minus=float(shared["tau_minus"]),
            theta=float(run_cfg["theta"]),
            delta=float(run_cfg["delta"]),
            max_k=int(run_cfg["max_k"]),
            n_lfi=int(n_lfi),
            n_max=int(shared["n_max"]),
            cold_share=(float(shared["cold_share_min"]), float(shared["cold_share_max"])),
            cold_popular_items=int(shared["cold_popular_items"]),
            ks=tuple(int(k) for k in ks),
            min_cold_users=pick("min_cold_users", None),
            cluster_pop_n_clusters=pick("cluster_pop_n_clusters", None),
            use_negative=bool(pick("use_negative", True)),
            negative_penalty=float(pick("negative_penalty", 0.0)),
            min_support_abs=int(pick("min_support_abs", 1)),
            forbid_singleton_support=bool(pick("forbid_singleton_support", False)),
            auto_peeling=bool(pick("auto_peeling", False)),
            min_new_positive_items=int(pick("min_new_positive_items", 1)),
            max_lfi_overlap=float(pick("max_lfi_overlap", 1.0)),
            lfi_pop_rho=float(pick("lfi_pop_rho", 0.0)),
            cold_split_mode=str(pick("cold_split_mode", "popular")),
            hybrid_lambda=float(pick("hybrid_lambda", 0.35)),
            soft_clusters=int(pick("soft_clusters", 1)),
            global_pop_penalty=float(pick("global_pop_penalty", 0.0)),
            rrf_k=float(pick("rrf_k", 60.0)),
            knn_neighbors=int(pick("knn_neighbors", 50)),
            feature_knn_weight=float(pick("feature_knn_weight", 0.0)),
            context_pop_weight=float(pick("context_pop_weight", 0.0)),
            context_cols=tuple(pick("context_cols", []) or []),
            personalized_negative_weight=float(pick("personalized_negative_weight", 0.0)),
            include_lightfm=bool(pick("include_lightfm", False)),
            lightfm_components=int(pick("lightfm_components", 64)),
            lightfm_epochs=int(pick("lightfm_epochs", 30)),
            lightfm_lr=float(pick("lightfm_lr", 0.05)),
            include_dropoutnet=bool(pick("include_dropoutnet", False)),
            dropoutnet_components=int(pick("dropoutnet_components", 64)),
            dropoutnet_dropout=float(pick("dropoutnet_dropout", 0.5)),
            dropoutnet_epochs=int(pick("dropoutnet_epochs", 30)),
            include_melu=bool(pick("include_melu", False)),
            melu_components=int(pick("melu_components", 32)),
            melu_local_lr=float(pick("melu_local_lr", 0.1)),
            melu_epochs=int(pick("melu_epochs", 40)),
        )


def load_params() -> Dict[str, Any]:
    return json.loads(CONF_PATH.read_text())
