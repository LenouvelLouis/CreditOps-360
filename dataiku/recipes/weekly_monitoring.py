# Zone 5 - Monitoring: model performance and score drift week by week on the
# simulated production period (weeks after ${train_week_max}).
#
# Reference for the PSI: the first production weeks, i.e. the score distribution
# observed right after deployment.
import dataiku
import pandas as pd
from creditops.stability import gini_by_week, psi, stability_score

REFERENCE_WEEKS = 4
variables = dataiku.get_custom_variables()
psi_threshold = float(variables.get("psi_threshold", 0.2))
gini_drop = float(variables.get("gini_drop_threshold", 0.05))

df = dataiku.Dataset("hc_prod_scored").get_dataframe(columns=["WEEK_NUM", "target", "proba_1"])
weeks = sorted(df["WEEK_NUM"].unique())
reference = df[df["WEEK_NUM"].isin(weeks[:REFERENCE_WEEKS])]

gini = gini_by_week(df, score="proba_1")
ref_gini = gini.loc[weeks[:REFERENCE_WEEKS]].mean()

rows = []
for week, g in df.groupby("WEEK_NUM"):
    week_psi = psi(reference["proba_1"], g["proba_1"])
    week_gini = gini.get(week)
    rows.append({
        "WEEK_NUM": int(week),
        "n_cases": len(g),
        "default_rate": g["target"].mean(),
        "mean_pd": g["proba_1"].mean(),
        "gini": week_gini,
        "psi_score": week_psi,
        "alert_psi": week_psi > psi_threshold,
        "alert_gini": week_gini is not None and week_gini < ref_gini - gini_drop,
    })

weekly = pd.DataFrame(rows)
dataiku.Dataset("hc_weekly_monitoring").write_with_schema(weekly)

summary = pd.DataFrame([{
    "weeks": len(weekly),
    "gini_mean": gini.mean(),
    "gini_reference": ref_gini,
    "stability_score": stability_score(gini),
    "weeks_psi_alert": int(weekly["alert_psi"].sum()),
    "weeks_gini_alert": int(weekly["alert_gini"].sum()),
}])
dataiku.Dataset("hc_stability_summary").write_with_schema(summary)
print(summary.to_string(index=False))
