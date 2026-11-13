"""Aggregation of Home Credit (Credit Risk Model Stability) tables to one row per case_id.

Naming conventions of the source data:
- depth 0 tables: one row per case_id
- depth 1 tables: several rows per case_id, indexed by num_group1
- depth 2 tables: several rows per (case_id, num_group1), indexed by num_group2
- column suffix: P = days past due, M = masked category, A = amount,
  D = date, T / L = other numeric or categorical
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY = "case_id"
GROUP_COLS = {"num_group1", "num_group2"}
NUM_STATS = ["max", "mean", "min"]
SUFFIX_RE = re.compile(r"[A-Z]$")


def table_name(path: str | Path) -> str:
    """'train_credit_bureau_a_1_3.parquet' -> 'credit_bureau_a_1'."""
    stem = Path(path).stem
    stem = re.sub(r"^(train|test)_", "", stem)
    return re.sub(r"_\d+$", "", stem) if re.search(r"_\d+_\d+$", stem) else stem


def table_depth(name: str) -> int:
    match = re.search(r"_(\d)$", name)
    return int(match.group(1)) if match else 0


def group_files(paths: Iterable[str | Path]) -> dict[str, list[Path]]:
    """Group split parquet chunks by logical table name."""
    groups: dict[str, list[Path]] = {}
    for p in sorted(Path(x) for x in paths):
        groups.setdefault(table_name(p), []).append(p)
    return groups


def load_chunk(path: str | Path) -> pd.DataFrame:
    """Reads a parquet chunk with a small memory footprint.

    Strings become categoricals and floats become float32, which divides memory
    by 5 to 10 on the credit bureau tables.
    """
    df = pq.read_table(path).to_pandas(strings_to_categorical=True, self_destruct=True)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    return df


def dates_to_days(df: pd.DataFrame, decision: pd.Series) -> pd.DataFrame:
    """Replaces *D date columns by the number of days before date_decision.

    `decision` is date_decision indexed by case_id. Dates are parsed once per
    distinct value (on the categories), not once per row.
    """
    ref = df[KEY].map(decision).to_numpy(dtype="datetime64[ns]")
    for c in [c for c in df.columns if c.endswith("D")]:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = pd.to_datetime(s.cat.categories, errors="coerce").to_numpy(dtype="datetime64[ns]")
            codes = s.cat.codes.to_numpy()
            values = np.where(codes >= 0, cats[np.clip(codes, 0, None)], np.datetime64("NaT"))
        else:
            values = pd.to_datetime(s, errors="coerce").to_numpy(dtype="datetime64[ns]")
        df[c] = ((values - ref) / np.timedelta64(1, "D")).astype("float32")
    return df


def aggregate_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reduces a depth 0, 1 or 2 table to one row per case_id.

    Numeric columns: max, mean, min. Categorical columns: number of distinct
    values. Depth 2 rows are aggregated directly per case_id.
    """
    prefix = f"{name}__"
    if table_depth(name) == 0:
        return df.drop_duplicates(KEY).rename(columns={c: prefix + c for c in df.columns if c != KEY})

    value_cols = [c for c in df.columns if c != KEY and c not in GROUP_COLS]
    num = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in value_cols if c not in num]
    g = df.groupby(KEY, sort=False, observed=True)

    parts = [g.size().rename(f"{prefix}count").astype("int32").to_frame()]
    if num:
        agg = g[num].agg(NUM_STATS)
        agg.columns = [f"{prefix}{c}_{s}" for c, s in agg.columns]
        parts.append(agg.astype("float32"))
    if cat:
        codes = df[[KEY]].copy()
        for c in cat:
            codes[c] = df[c].cat.codes.where(df[c].cat.codes >= 0) if hasattr(df[c], "cat") else df[c]
        nunique = codes.groupby(KEY, sort=False)[cat].nunique()
        nunique.columns = [f"{prefix}{c}_nunique" for c in cat]
        parts.append(nunique.astype("int16"))
    return pd.concat(parts, axis=1).reset_index()


def aggregate_files(paths: list[Path], name: str, decision: pd.Series | None = None) -> pd.DataFrame:
    """Aggregates chunk by chunk (chunks are split on case_id ranges), then concatenates."""
    out = []
    for p in paths:
        chunk = load_chunk(p)
        if decision is not None:
            chunk = dates_to_days(chunk, decision)
        out.append(aggregate_table(chunk, name))
        del chunk
    result = pd.concat(out, ignore_index=True)
    if result[KEY].duplicated().any():  # a case_id split across two chunks
        result = result.groupby(KEY, as_index=False).max()
    return result
