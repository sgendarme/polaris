from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_columns(users: pd.DataFrame, demo_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    cols = [c for c in demo_cols if c in users.columns]
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(users[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    return num_cols, cat_cols


def build_user_feature_pipeline(num_cols: Sequence[str], cat_cols: Sequence[str]) -> Pipeline:
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(steps=[("pre", pre)])


def clean_numeric_inf(df: pd.DataFrame, num_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if num_cols:
        out[list(num_cols)] = out[list(num_cols)].replace([np.inf, -np.inf], np.nan)
    return out
