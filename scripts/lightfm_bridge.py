#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, identity

warnings.filterwarnings("ignore")

from lightfm import LightFM


def _coo(z, prefix: str):
    shape = tuple(int(x) for x in z[f"{prefix}_shape"])
    return coo_matrix(
        (z[f"{prefix}_data"], (z[f"{prefix}_row"], z[f"{prefix}_col"])), shape=shape
    ).tocsr()


def main() -> None:
    work = Path(sys.argv[1])
    meta = json.loads((work / "meta.json").read_text())
    params = meta["params"]
    z = np.load(work / "arrays.npz")

    user_features = _coo(z, "uf")
    interactions = _coo(z, "in")
    n_users, n_items = interactions.shape
    item_features = identity(n_items, format="csr", dtype=np.float32)

    model = LightFM(
        loss="warp",
        no_components=int(params["no_components"]),
        learning_rate=float(params.get("learning_rate", 0.05)),
        random_state=int(params["seed"]),
    )
    model.fit(
        interactions,
        user_features=user_features,
        item_features=item_features,
        epochs=int(params["epochs"]),
        num_threads=1,
    )

    all_user_ids = [str(u) for u in meta["all_user_ids"]]
    item_ids = [str(i) for i in meta["item_ids"]]
    uidx = {u: i for i, u in enumerate(all_user_ids)}
    n_max = int(params["n_max"])
    item_arr = np.arange(n_items)

    rankings = {}
    for u in meta["cold_user_ids"]:
        u = str(u)
        row = uidx[u]
        scores = model.predict(
            row, item_arr, user_features=user_features, item_features=item_features
        )
        top = np.argsort(-scores)[:n_max]
        rankings[u] = [item_ids[j] for j in top.tolist()]

    (work / "rankings.json").write_text(json.dumps(rankings))


if __name__ == "__main__":
    main()
