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

import pandas as pd

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


def cast_dates(df: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert *D columns to days relative to date_decision (or to datetimes)."""
    df = df.copy()
    date_cols = [c for c in df.columns if c.endswith("D") or c == "date_decision"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if base is not None and "date_decision" in base.columns:
        ref = base[[KEY, "date_decision"]].copy()
        ref["date_decision"] = pd.to_datetime(ref["date_decision"], errors="coerce")
        df = df.merge(ref, on=KEY, how="left", suffixes=("", "_ref"))
        for c in date_cols:
            if c != "date_decision":
                df[c] = (df[c] - df["date_decision"]).dt.days
        df = df.drop(columns=["date_decision"])
    return df


def _aggregate(df: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c not in keys and c not in GROUP_COLS]
    num = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in value_cols if c not in num]

    parts = []
    g = df.groupby(keys, sort=False)
    if num:
        agg = g[num].agg(NUM_STATS)
        agg.columns = [f"{prefix}{c}_{s}" for c, s in agg.columns]
        parts.append(agg)
    if cat:
        nunique = g[cat].nunique()
        nunique.columns = [f"{prefix}{c}_nunique" for c in cat]
        first = g[cat].first()
        first.columns = [f"{prefix}{c}_first" for c in cat]
        parts += [nunique, first]
    count = g.size().rename(f"{prefix}count").to_frame()
    parts.append(count)
    return pd.concat(parts, axis=1).reset_index()


def aggregate_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reduce a depth 0, 1 or 2 table to one row per case_id."""
    depth = table_depth(name)
    prefix = f"{name}__"
    if depth == 0:
        return df.drop_duplicates(KEY).rename(
            columns={c: prefix + c for c in df.columns if c != KEY}
        )
    if depth == 2:
        df = _aggregate(df, [KEY, "num_group1"], prefix="")
    return _aggregate(df, [KEY], prefix=prefix)


def aggregate_files(paths: list[Path], name: str, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate chunk by chunk (chunks are split on case_id ranges), then concatenate."""
    out = []
    for p in paths:
        chunk = pd.read_parquet(p)
        chunk = cast_dates(chunk, base)
        out.append(aggregate_table(chunk, name))
    result = pd.concat(out, ignore_index=True)
    if result[KEY].duplicated().any():  # safety if a case_id spans two chunks
        result = result.groupby(KEY, as_index=False).first()
    return result
