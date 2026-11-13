# Zone 3 - Features: aggregates the heavy Home Credit tables (credit bureau, depth 2)
# straight from Parquet, chunk by chunk, to one row per case_id.
import os

import dataiku
import pandas as pd
from creditops.aggregations import aggregate_files, group_files

HEAVY = ["credit_bureau_a_1", "credit_bureau_a_2", "credit_bureau_b_2", "applprev_2", "person_2"]

folder = dataiku.Folder("hc_raw_parquet")
root = folder.get_path()
groups = group_files(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".parquet"))

base = pd.read_parquet(groups["base"][0], columns=["case_id", "date_decision"])
decision = pd.to_datetime(base.set_index("case_id")["date_decision"])
features = base[["case_id"]].copy()

for table in HEAVY:
    agg = aggregate_files(groups[table], table, decision=decision)
    features = features.merge(agg, on="case_id", how="left")
    print(f"{table}: {agg.shape[1] - 1} features")
    del agg

dataiku.Dataset("hc_agg_bureau").write_with_schema(features)
