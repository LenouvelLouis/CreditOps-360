# Zone 1 - Ingestion: converts the light Home Credit tables (depth 0 and small depth 1)
# from the raw Parquet folder into DSS datasets.
import os

import dataiku
import pandas as pd
from creditops.aggregations import group_files

TABLES = [
    "base", "static_0", "static_cb_0",
    "applprev_1", "person_1", "credit_bureau_b_1", "debitcard_1", "deposit_1",
    "other_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1",
]

folder = dataiku.Folder("hc_raw_parquet")
root = folder.get_path()
groups = group_files(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".parquet"))

for table in TABLES:
    df = pd.concat([pd.read_parquet(p) for p in groups[table]], ignore_index=True)
    dataiku.Dataset(f"hc_{table}").write_with_schema(df)
    print(f"hc_{table}: {len(df):,} rows, {df.shape[1]} cols")
