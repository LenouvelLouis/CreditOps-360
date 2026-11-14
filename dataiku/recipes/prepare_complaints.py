# Zone 6 - Complaints: keeps banking complaints that have a narrative and draws a
# sample stratified by product, used to build and evaluate the LLM triage.
#
# The CFPB product taxonomy changed in April 2017: complaints received before that
# date are tagged "legacy" so that label drift can be studied.
import dataiku
import pandas as pd

SAMPLE_CURRENT = 8000
SAMPLE_LEGACY = 2000
SEED = 42
TAXONOMY_CHANGE = pd.Timestamp("2017-04-24")
BANK_PRODUCTS = {
    # current taxonomy
    "Credit card", "Checking or savings account", "Mortgage", "Vehicle loan or lease",
    "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan, title loan, or personal loan", "Credit card or prepaid card",
    # legacy taxonomy
    "Bank account or service", "Consumer Loan",
}
COLUMNS = {
    "Complaint ID": "complaint_id", "Date received": "date_received", "Product": "product",
    "Sub-product": "sub_product", "Issue": "issue", "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative", "Company": "company", "State": "state",
    "Company response to consumer": "company_response", "Timely response?": "timely_response",
}

parts = []
source = dataiku.Dataset("cfpb_complaints_raw")
for chunk in source.iter_dataframes(chunksize=200_000, columns=list(COLUMNS), infer_with_pandas=False):
    chunk = chunk[chunk["Consumer complaint narrative"].notna() & chunk["Product"].isin(BANK_PRODUCTS)]
    parts.append(chunk.rename(columns=COLUMNS))
df = pd.concat(parts, ignore_index=True).drop_duplicates("complaint_id")
df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
df["taxonomy_era"] = (df["date_received"] < TAXONOMY_CHANGE).map({True: "legacy", False: "current"})
print(df.groupby(["taxonomy_era", "product"]).size().to_string())


def stratified(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Same number of complaints per product where possible, the rest proportionally."""
    per_product = max(n // frame["product"].nunique(), 1)
    sample = frame.groupby("product", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_product), random_state=SEED))
    missing = n - len(sample)
    if missing > 0:
        rest = frame.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(missing, len(rest)), random_state=SEED)])
    return sample


sample = pd.concat([
    stratified(df[df["taxonomy_era"] == "current"], SAMPLE_CURRENT),
    stratified(df[df["taxonomy_era"] == "legacy"], SAMPLE_LEGACY),
], ignore_index=True)
sample["narrative_length"] = sample["narrative"].str.len()
print(f"cfpb_sample: {len(sample):,} complaints")
dataiku.Dataset("cfpb_sample").write_with_schema(sample)
