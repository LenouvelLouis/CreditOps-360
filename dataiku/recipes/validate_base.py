# Zone 2 - Quality: gate on the base table before feature engineering.
# Fails the build if the key or the target is not usable, so that nothing
# downstream is computed on corrupted data.
import dataiku

df = dataiku.Dataset("hc_base").get_dataframe()

errors = []
if df.empty:
    errors.append("hc_base is empty")
n_dup = int(df["case_id"].duplicated().sum())
if n_dup:
    errors.append(f"{n_dup} duplicated case_id")
for col in ("case_id", "target", "WEEK_NUM"):
    n_null = int(df[col].isna().sum())
    if n_null:
        errors.append(f"{n_null} null values in {col}")
bad_target = sorted(set(df["target"].dropna().unique()) - {0, 1})
if bad_target:
    errors.append(f"unexpected target values: {bad_target[:10]}")

if errors:
    raise ValueError("hc_base quality gate failed: " + "; ".join(errors))

print(f"hc_base_validated: {len(df):,} rows, default rate {df['target'].mean():.4f}, "
      f"weeks {int(df['WEEK_NUM'].min())}-{int(df['WEEK_NUM'].max())}")
dataiku.Dataset("hc_base_validated").write_with_schema(df)
