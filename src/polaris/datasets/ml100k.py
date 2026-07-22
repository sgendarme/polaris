from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from .base import DatasetLoader, DatasetSplit


GENRE_COLS: List[str] = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class MovieLens100KDataset(DatasetLoader):
    data_dir: str
    name: str = "ml-100k"

    def load(self) -> DatasetSplit:
        ratings = pd.read_csv(
            os.path.join(self.data_dir, "u.data"),
            sep="\t",
            header=None,
            names=["userId", "itemId", "rating", "ts"],
        )
        users = pd.read_csv(
            os.path.join(self.data_dir, "u.user"),
            sep="|",
            header=None,
            names=["userId", "age", "gender", "occupation", "zip"],
        )
        item_cols = ["itemId", "title", "release_date", "video_release", "imdb_url"] + GENRE_COLS
        items = pd.read_csv(
            os.path.join(self.data_dir, "u.item"),
            sep="|",
            header=None,
            names=item_cols,
            encoding="latin-1",
        )

        items["genre"] = items[GENRE_COLS].idxmax(axis=1)

        bins = [0, 18, 25, 35, 45, 55, 65, 200]
        labels = ["<=18", "19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        users["age_bin"] = pd.cut(
            users["age"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

        ratings["userId"] = ratings["userId"].astype(str)
        ratings["itemId"] = ratings["itemId"].astype(str)
        users["userId"] = users["userId"].astype(str)
        items["itemId"] = items["itemId"].astype(str)

        return DatasetSplit(ratings=ratings, users=users, items=items)
