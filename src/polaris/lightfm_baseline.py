from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .config import ROOT
from .features import build_user_feature_pipeline, clean_numeric_inf, split_feature_columns
from .baselines import popularity_ranking


def _lightfm_python() -> Path:
    override = os.environ.get("POLARIS_LIGHTFM_PYTHON")
    if override:
        return Path(override)
    return ROOT / ".venv-lightfm" / "bin" / "python"


def _bridge_script() -> Path:
    return ROOT / "scripts" / "lightfm_bridge.py"


def _user_feature_matrix(
    users: pd.DataFrame,
    demo_cols: Sequence[str],
    warm_ids: Sequence[str],
    all_ids: Sequence[str],
) -> coo_matrix:
    cols = [c for c in demo_cols if c in users.columns]
    if not cols:
        raise ValueError("LightFM requires at least one available user feature column.")
    num_cols, cat_cols = split_feature_columns(users, demo_cols)
    pipe = build_user_feature_pipeline(num_cols, cat_cols)

    by_id = users.drop_duplicates("userId").set_index(users.drop_duplicates("userId")["userId"].astype(str))
    warm_df = by_id.loc[[u for u in warm_ids if u in by_id.index]]
    pipe.fit(clean_numeric_inf(warm_df[num_cols + cat_cols], num_cols))

    all_df = by_id.loc[[u for u in all_ids if u in by_id.index]]
    x = pipe.transform(clean_numeric_inf(all_df[num_cols + cat_cols], num_cols))
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return coo_matrix(x)


def lightfm_recommender(
    cold_users: List[str],
    *,
    users: pd.DataFrame,
    train_ratings: pd.DataFrame,
    catalog: Sequence[str],
    demo_cols: Sequence[str],
    rating_thr: float,
    n_max: int,
    no_components: int,
    epochs: int,
    seed: int,
    learning_rate: float = 0.05,
    global_fallback: Sequence[str] | None = None,
) -> Dict[str, List[str]]:
    if global_fallback is None:
        global_fallback = popularity_ranking(train_ratings, rating_thr)
    global_fallback = [str(i) for i in global_fallback]
    size = int(n_max) if n_max and n_max > 0 else None
    fallback_top = global_fallback[:size] if size is not None else list(global_fallback)

    item_ids = [str(i) for i in catalog]
    item_idx = {it: j for j, it in enumerate(item_ids)}
    n_items = len(item_ids)

    warm_ids = sorted({str(u) for u in train_ratings["userId"].unique()})
    cold_present = [str(u) for u in cold_users if str(u) in set(users["userId"].astype(str))]
    all_ids = sorted(set(warm_ids) | set(cold_present))
    if not warm_ids or n_items == 0:
        return {str(u): list(fallback_top) for u in cold_users}

    uidx = {u: i for i, u in enumerate(all_ids)}
    user_features = _user_feature_matrix(users, demo_cols, warm_ids, all_ids)

    pos = train_ratings[train_ratings["rating"] >= rating_thr]
    rows, cols = [], []
    for u, it in zip(pos["userId"].astype(str), pos["itemId"].astype(str)):
        ri, ci = uidx.get(u), item_idx.get(it)
        if ri is not None and ci is not None:
            rows.append(ri)
            cols.append(ci)
    inter = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(all_ids), n_items),
    )

    with tempfile.TemporaryDirectory(prefix="polaris_lightfm_") as tmp:
        work = Path(tmp)
        np.savez(
            work / "arrays.npz",
            uf_row=user_features.row.astype(np.int64),
            uf_col=user_features.col.astype(np.int64),
            uf_data=user_features.data.astype(np.float32),
            uf_shape=np.asarray(user_features.shape, dtype=np.int64),
            in_row=inter.row.astype(np.int64),
            in_col=inter.col.astype(np.int64),
            in_data=inter.data.astype(np.float32),
            in_shape=np.asarray(inter.shape, dtype=np.int64),
        )
        (work / "meta.json").write_text(json.dumps({
            "all_user_ids": all_ids,
            "item_ids": item_ids,
            "cold_user_ids": cold_present,
            "params": {
                "no_components": int(no_components),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
                "n_max": int(n_max),
            },
        }))
        proc = subprocess.run(
            [str(_lightfm_python()), str(_bridge_script()), str(work)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LightFM subprocess failed (code {proc.returncode}).\n"
                f"stderr:\n{proc.stderr[-2000:]}"
            )
        ranked = json.loads((work / "rankings.json").read_text())

    out: Dict[str, List[str]] = {}
    for u in cold_users:
        u = str(u)
        r = [str(i) for i in ranked.get(u, [])]
        if not r:
            r = list(fallback_top)
        out[u] = r[:size] if size is not None else r
    return out
