from __future__ import annotations

from pathlib import Path

from ..config import ROOT
from .base import DatasetLoader, DatasetSplit
from .comoda import CoMoDaDataset
from .ml1m import MovieLens1MDataset
from .ml100k import MovieLens100KDataset

_DIRS = {
    "ml-100k": ROOT / "resources" / "ml-100k",
    "ml-1m": ROOT / "resources" / "ml-1m",
    "comoda": ROOT / "resources" / "CoMoDa",
}


def dataset_dir(dataset: str) -> Path:
    return _DIRS[dataset]


def build_dataset(name: str) -> DatasetLoader:
    dataset = name.lower()
    data_dir = str(dataset_dir(dataset))
    if dataset == "ml-100k":
        return MovieLens100KDataset(data_dir=data_dir)
    if dataset == "ml-1m":
        return MovieLens1MDataset(data_dir=data_dir)
    if dataset == "comoda":
        return CoMoDaDataset(data_dir=data_dir)
    raise ValueError(f"Unknown dataset: {name}")


__all__ = ["DatasetLoader", "DatasetSplit", "build_dataset", "dataset_dir"]
