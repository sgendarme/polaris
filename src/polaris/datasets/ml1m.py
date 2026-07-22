from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .base import DatasetLoader, DatasetSplit


AGE_CODE_TO_APPROX_AGE: Dict[int, int] = {
    1: 15,
    18: 21,
    25: 30,
    35: 40,
    45: 47,
    50: 53,
    56: 60,
}


@dataclass
class MovieLens1MDataset(DatasetLoader):
    data_dir: str
    name: str = "ml-1m"

    def load(self) -> DatasetSplit:
        ratings_path = os.path.join(self.data_dir, "ratings.dat")
        users_path = os.path.join(self.data_dir, "users.dat")
        movies_path = os.path.join(self.data_dir, "movies.dat")

        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Missing file: {ratings_path}")
        if not os.path.exists(users_path):
            raise FileNotFoundError(f"Missing file: {users_path}")
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Missing file: {movies_path}")

        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "itemId", "rating", "ts"],
        )
        users = pd.read_csv(
            users_path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "gender", "age_code", "occupation", "zip"],
        )
        items = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            header=None,
            names=["itemId", "title", "genres"],
            encoding="latin-1",
        )

        users["age_code"] = pd.to_numeric(users["age_code"], errors="coerce").astype("Int64")
        users["age"] = users["age_code"].map(AGE_CODE_TO_APPROX_AGE).astype("Float64")
        users["occupation"] = users["occupation"].astype(str)
        users["zip"] = users["zip"].astype(str)

        bins = [0, 18, 25, 35, 45, 55, 65, 200]
        labels = ["<=18", "19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        users["age_bin"] = pd.cut(
            users["age"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

        items["genres"] = items["genres"].fillna("unknown").astype(str)
        items["genre"] = items["genres"].str.split("|").str[0].fillna("unknown").replace("", "unknown")

        ratings["userId"] = ratings["userId"].astype(str)
        ratings["itemId"] = ratings["itemId"].astype(str)
        users["userId"] = users["userId"].astype(str)
        items["itemId"] = items["itemId"].astype(str)

        return DatasetSplit(ratings=ratings, users=users, items=items)
