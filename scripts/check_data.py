"""Checks that the local data folder is complete and prints a short inventory."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataiku" / "python"))

from creditops.aggregations import group_files, table_depth  # noqa: E402


def main() -> int:
    ok = True
    train = ROOT / "homecredit" / "parquet_files" / "train"
    files = list(train.glob("*.parquet"))
    if not files:
        print(f"[missing] Home Credit parquet files in {train}")
        ok = False
    else:
        import pyarrow.parquet as pq

        print(f"Home Credit train: {len(files)} files")
        for name, paths in group_files(files).items():
            rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
            cols = pq.ParquetFile(paths[0]).metadata.num_columns
            print(f"  depth {table_depth(name)}  {name:<28} {len(paths):>2} file(s) {rows:>12,} rows {cols:>4} cols")

    cfpb = sorted(p for p in (ROOT / "cfpb").glob("*") if p.is_dir())
    print(f"CFPB exports: {len(cfpb)}")
    for d in cfpb:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  {d.name:<55} {size / 1e6:>8.1f} MB")
    if not cfpb:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
