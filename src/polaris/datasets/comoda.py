from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

from .base import DatasetLoader, DatasetSplit


def _norm_col(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _first_matching_col(columns: Iterable[Any], candidates_norm: Iterable[str]) -> Optional[Any]:
    norm2orig: Dict[str, Any] = {}
    for col in columns:
        norm = _norm_col(col)
        if norm not in norm2orig:
            norm2orig[norm] = col
    for cand in candidates_norm:
        if cand in norm2orig:
            return norm2orig[cand]
    return None


def _coerce_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def _mode(series: pd.Series) -> Any:
    s = series.dropna()
    if s.empty:
        return pd.NA
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else s.iloc[0]


def _map_gender(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"m", "male", "1"}:
            return "M"
        if v in {"f", "female", "2"}:
            return "F"
        if v in {"u", "unknown"}:
            return "U"
        return value.strip()
    try:
        v_int = int(value)
    except Exception:
        return "U"
    if v_int == 1:
        return "M"
    if v_int == 2:
        return "F"
    return "U"


def _read_interactions(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, "LDOS-CoMoDa.csv")
    tsv_path = os.path.join(data_dir, "LDOS-CoMoDa.tsv")
    xls_path = os.path.join(data_dir, "LDOS-CoMoDa.xls")

    na_vals = ["-1", -1, "NULL", "null"]
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, sep=None, engine="python", na_values=na_vals)
    if os.path.exists(tsv_path):
        return pd.read_csv(tsv_path, sep="\t", na_values=na_vals)
    if os.path.exists(xls_path):
        try:
            return pd.read_excel(xls_path, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                "Reading CoMoDa from 'LDOS-CoMoDa.xls' requires optional 'xlrd'. "
                "Install it (pip install xlrd>=2.0.1) or convert to CSV."
            ) from exc

    raise FileNotFoundError(
        "CoMoDa interactions file not found. Expected LDOS-CoMoDa.csv/tsv/xls in "
        f"{data_dir!r}."
    )


def _read_items_metadata(data_dir: str) -> pd.DataFrame:
    candidates = [
        os.path.join(data_dir, "itemmetadata2_v2.csv"),
        os.path.join(data_dir, "itemMetadataDecoded.csv"),
        os.path.join(data_dir, "itemsTitles.csv"),
    ]
    na_vals = ["-1", -1, "NULL", "null"]
    if os.path.exists(candidates[0]):
        return pd.read_csv(candidates[0], sep=";", engine="python", na_values=na_vals)
    if os.path.exists(candidates[1]):
        return pd.read_csv(candidates[1], sep=";", engine="python", na_values=na_vals)
    if os.path.exists(candidates[2]):
        return pd.read_csv(candidates[2], sep=";", header=None, names=["itemId", "title"], na_values=na_vals)
    return pd.DataFrame(columns=["itemId", "title"])


def _pick_first_nonnull(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for c in cols:
        if c not in df.columns:
            continue
        mask = out.isna() & df[c].notna()
        out.loc[mask] = df.loc[mask, c]
    return out


@dataclass
class CoMoDaDataset(DatasetLoader):
    data_dir: str
    name: str = "comoda"

    def load(self) -> DatasetSplit:
        raw = _read_interactions(self.data_dir).replace([-1, "-1"], pd.NA)

        user_col = _first_matching_col(raw.columns, ["userid"])
        item_col = _first_matching_col(raw.columns, ["itemid", "movieid"])
        rating_col = _first_matching_col(raw.columns, ["rating"])
        if user_col is None or item_col is None or rating_col is None:
            if len(raw.columns) < 3:
                raise KeyError("CoMoDa requires userId/itemId/rating columns.")
            user_col, item_col, rating_col = raw.columns[:3]

        age_col = _first_matching_col(raw.columns, ["age", "userage", "usersage"])
        sex_col = _first_matching_col(raw.columns, ["sex", "gender", "usersex", "usergender"])
        city_col = _first_matching_col(raw.columns, ["city", "usercity"])
        country_col = _first_matching_col(raw.columns, ["country", "usercountry"])
        context_names = [
            "time", "daytype", "season", "location", "weather", "social",
            "endemo", "dominantemo", "mood", "physical", "decision", "interaction",
        ]
        context_map = {name: _first_matching_col(raw.columns, [name]) for name in context_names}
        context_map = {name: col for name, col in context_map.items() if col is not None}

        rating_cols = [user_col, item_col, rating_col] + list(context_map.values())
        ratings = raw[rating_cols].rename(
            columns={user_col: "userId", item_col: "itemId", rating_col: "rating", **{v: k for k, v in context_map.items()}}
        )
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        ratings = ratings.dropna(subset=["userId", "itemId", "rating"]).copy()
        ratings["userId"] = _coerce_id(ratings["userId"])
        ratings["itemId"] = _coerce_id(ratings["itemId"])
        context_cols = list(context_map.keys())
        agg_r = {"rating": "mean"}
        agg_r.update({c: _mode for c in context_cols if c in ratings.columns})
        ratings = (
            ratings.groupby(["userId", "itemId"], as_index=False)
            .agg(agg_r)
            .sort_values(["userId", "itemId"], kind="stable")
            .reset_index(drop=True)
        )
        for c in context_cols:
            if c in ratings.columns:
                ratings[c] = ratings[c].astype("object").where(ratings[c].notna(), "unknown").astype(str)

        users_raw = pd.DataFrame({"userId": raw[user_col]}).dropna(subset=["userId"]).copy()
        users_raw["userId"] = _coerce_id(users_raw["userId"])
        users_raw["age"] = pd.to_numeric(raw[age_col], errors="coerce") if age_col is not None else pd.NA
        users_raw["gender"] = raw[sex_col].map(_map_gender) if sex_col is not None else pd.NA
        users_raw["city"] = (
            raw[city_col].astype(str).str.strip().replace({"nan": pd.NA}) if city_col is not None else pd.NA
        )
        users_raw["country"] = (
            raw[country_col].astype(str).str.strip().replace({"nan": pd.NA}) if country_col is not None else pd.NA
        )
        for ctx_name, ctx_col in context_map.items():
            users_raw[ctx_name] = raw[ctx_col].astype("object").where(raw[ctx_col].notna(), "unknown").astype(str)

        agg: Dict[str, Callable] = {"age": "median", "gender": _mode, "city": _mode, "country": _mode}
        agg.update({ctx_name: _mode for ctx_name in context_map.keys()})
        users = users_raw.groupby("userId", as_index=False).agg(agg)
        users["gender"] = users["gender"].fillna("U")
        users["city"] = users["city"].fillna("unknown")
        users["country"] = users["country"].fillna("unknown")
        users["occupation"] = "unknown"
        users["zip"] = "00000"
        for ctx_name in context_map.keys():
            users[ctx_name] = users[ctx_name].fillna("unknown").astype(str)

        bins = [0, 18, 25, 35, 45, 55, 65, 200]
        labels = ["<=18", "19-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        users["age_bin"] = pd.cut(users["age"], bins=bins, labels=labels, right=True, include_lowest=True)
        users["userId"] = users["userId"].astype(str)

        items_raw = _read_items_metadata(self.data_dir)
        cols_norm = {_norm_col(c): c for c in items_raw.columns}
        item_id_col = cols_norm.get("itemid")
        if item_id_col is None and "itemId" in items_raw.columns:
            item_id_col = "itemId"
        if item_id_col is None:
            item_id_col = _first_matching_col(items_raw.columns, ["itemid"])

        if item_id_col is None:
            items = pd.DataFrame({"itemId": ratings["itemId"].unique().tolist(), "genre": "unknown"})
        else:
            rename_map = {
                item_id_col: "itemId",
                cols_norm.get("title", "Title"): "title",
                cols_norm.get("director", "Director"): "director",
                cols_norm.get("country", "Country"): "movie_country",
                cols_norm.get("language", "Language"): "language",
                cols_norm.get("year", "Year"): "year",
                cols_norm.get("genre1", "Genre1"): "genre1",
                cols_norm.get("genre2", "Genre2"): "genre2",
                cols_norm.get("genre3", "Genre3"): "genre3",
                cols_norm.get("actor1", "Actor1"): "actor1",
                cols_norm.get("actor2", "Actor2"): "actor2",
                cols_norm.get("actor3", "Actor3"): "actor3",
                cols_norm.get("budget", "Budget"): "budget",
                cols_norm.get("imdblink", "IMDB_Link"): "imdb_url",
                cols_norm.get("numofratings", "Num_of_Ratings"): "num_ratings",
            }
            items = items_raw.rename(columns={k: v for k, v in rename_map.items() if k in items_raw.columns}).copy()
            items = items.dropna(subset=["itemId"]).copy()
            items["itemId"] = _coerce_id(items["itemId"])
            if "year" in items.columns:
                items["year"] = pd.to_numeric(items["year"], errors="coerce")
            if "budget" in items.columns:
                items["budget"] = pd.to_numeric(items["budget"], errors="coerce")
            if "num_ratings" in items.columns:
                items["num_ratings"] = pd.to_numeric(items["num_ratings"], errors="coerce")
            if "genre" not in items.columns:
                items["genre"] = _pick_first_nonnull(items, ["genre1", "genre2", "genre3"])
            items["genre"] = items["genre"].fillna("unknown")
            items = items.drop_duplicates(subset=["itemId"], keep="first").reset_index(drop=True)

        items["itemId"] = items["itemId"].astype(str)

        missing_users = sorted(set(ratings["userId"]) - set(users["userId"]))
        if missing_users:
            extra_u = pd.DataFrame(
                {
                    "userId": missing_users,
                    "age": pd.NA,
                    "gender": "U",
                    "city": "unknown",
                    "country": "unknown",
                    "occupation": "unknown",
                    "zip": "00000",
                    "age_bin": pd.NA,
                    **{ctx_name: "unknown" for ctx_name in context_map.keys()},
                }
            )
            users = pd.concat([users, extra_u], ignore_index=True)

        missing_items = sorted(set(ratings["itemId"]) - set(items["itemId"]))
        if missing_items:
            extra_i = pd.DataFrame({"itemId": missing_items, "genre": "unknown"})
            items = pd.concat([items, extra_i], ignore_index=True)

        return DatasetSplit(ratings=ratings, users=users, items=items)
