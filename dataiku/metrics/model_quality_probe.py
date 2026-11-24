# Custom Python probe on hc_prod_scored: overall Gini of the PD model on the
# production period and worst weekly PSI of the score distribution.
# Reference for the PSI: the first production weeks, as in weekly_monitoring.
from creditops.stability import gini, psi

REFERENCE_WEEKS = 4


def process(dataset, partition_id):
    df = dataset.get_dataframe(columns=["WEEK_NUM", "target", "proba_1"])
    weeks = sorted(df["WEEK_NUM"].unique())
    reference = df.loc[df["WEEK_NUM"].isin(weeks[:REFERENCE_WEEKS]), "proba_1"]
    weekly_psi = [psi(reference, g["proba_1"]) for _, g in df.groupby("WEEK_NUM")]
    return {
        "gini_overall": float(gini(df["target"], df["proba_1"])),
        "psi_weekly_max": float(max(weekly_psi)),
        "weeks_scored": len(weeks),
    }
