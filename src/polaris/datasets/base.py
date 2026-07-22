from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass
class DatasetSplit:
    ratings: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame


class DatasetLoader(Protocol):
    name: str

    def load(self) -> DatasetSplit:
        ...
