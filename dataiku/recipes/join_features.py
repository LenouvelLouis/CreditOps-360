# Zone 3 - Features: one row per case_id, joining the application tables (depth 0)
# with the aggregated credit bureau features. Dates become days before the decision.
import os

import dataiku
import pandas as pd
from creditops.aggregations import dates_to_days, group_files, load_chunk

folder = dataiku.Folder("hc_raw_parquet")
root = folder.get_path()
groups = group_files(os.path.join(root, f) for f in os.listdir(root) if f.endswith(".parquet"))

base = load_chunk(groups["base"][0])
base["date_decision"] = pd.to_datetime(base["date_decision"].astype(str))
decision = base.set_index("case_id")["date_decision"]

features = base
for table in ["static_0", "static_cb_0"]:
    df = pd.concat([load_chunk(p) for p in groups[table]], ignore_index=True)
    df = dates_to_days(df, decision).drop_duplicates("case_id")
    df = df.rename(columns={c: f"{table}__{c}" for c in df.columns if c != "case_id"})
    features = features.merge(df, on="case_id", how="left")
    del df

bureau = dataiku.Dataset("hc_agg_bureau").get_dataframe()
features = features.merge(bureau, on="case_id", how="left")
del bureau

# Age band: not a model input, used for the fairness analysis (AI Act, Annex III)
age = -features["static_cb_0__dateofbirth_337D"] / 365.25
features["age_band"] = pd.cut(age, [0, 30, 45, 60, 120], right=False,
                              labels=["18-29", "30-44", "45-59", "60+"]).astype(str)
features.loc[age.isna(), "age_band"] = "unknown"

# Columns that are fully empty bring nothing to the model
features = features.dropna(axis=1, how="all")
print(f"hc_features: {len(features):,} rows, {features.shape[1]} cols")
dataiku.Dataset("hc_features").write_with_schema(features)
