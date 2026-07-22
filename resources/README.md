# Datasets

The datasets are not distributed with this repository (`.gitignore` excludes
them). Download them from their official sources and place the files here:

```
resources/
  ml-100k/            MovieLens 100K -- https://grouplens.org/datasets/movielens/100k/
    u.data  u.user  u.item
  ml-1m/              MovieLens 1M -- https://grouplens.org/datasets/movielens/1m/
    ratings.dat  users.dat  movies.dat
  CoMoDa/             LDOS-CoMoDa -- https://www.lucami.org/en/research/ldos-comoda-dataset/
    LDOS-CoMoDa.csv         interactions (CSV export; TSV/XLS also accepted)
    itemmetadata2_v2.csv    item metadata (itemsTitles.csv works as a fallback)
```

The MovieLens archives unpack directly to the expected file names.
LDOS-CoMoDa is available on request from its authors; its item metadata file
defines the 3,120-item catalog used in the paper.
