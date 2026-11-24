# Zone 8 - Agents: light case profile used by the agents' Dataset Lookup tool.
# One row per production case (hc_prod_scored), about twenty business columns renamed in
# French plus the PD score, so the LLM receives a short, readable record instead
# of 500 technical features.
import dataiku
import numpy as np
import pandas as pd

COLUMNS = {
    "case_id": "case_id",
    "date_decision": "date_decision",
    "WEEK_NUM": "semaine",
    "age_band": "tranche_age",
    "static_0__credtype_322L": "type_credit",
    "static_0__credamount_770A": "montant_credit",
    "static_0__annuity_780A": "mensualite",
    "static_0__pmtnum_254L": "nb_echeances",
    "static_0__maininc_215A": "revenu_principal",
    "static_0__currdebt_22A": "dette_courante",
    "static_0__totaldebt_9A": "dette_totale",
    "static_0__maxdpdlast12m_727P": "retard_max_12m_jours",
    "static_0__maxdpdlast24m_143P": "retard_max_24m_jours",
    "static_0__numinstlswithdpd10_728L": "nb_echeances_retard_10j",
    "static_0__numrejects9m_859L": "nb_refus_9m",
    "static_0__lastst_736L": "statut_derniere_demande",
    "credit_bureau_a_1__count": "nb_contrats_bureau",
    "credit_bureau_a_1__debtoverdue_47A_max": "impaye_bureau_max",
    "proba_1": "proba_defaut",
}

df = dataiku.Dataset("hc_prod_scored").get_dataframe(columns=list(COLUMNS))
df = df.rename(columns=COLUMNS)

# Debt-to-income style indicator (monthly payment / monthly income), when income is known
income = df["revenu_principal"].where(df["revenu_principal"] > 0)
df["taux_effort"] = (df["mensualite"] / income).round(3)

# Risk class relative to the production portfolio: deciles of the PD score
df["decile_risque"] = pd.qcut(df["proba_defaut"].rank(method="first"), 10, labels=False) + 1
df["classe_risque"] = np.select(
    [df["decile_risque"] <= 5, df["decile_risque"] <= 9], ["faible", "modéré"], "élevé")
df["proba_defaut"] = df["proba_defaut"].round(4)
for c in ["montant_credit", "mensualite", "revenu_principal", "dette_courante",
          "dette_totale", "impaye_bureau_max"]:
    df[c] = df[c].round(0)
df["case_id"] = df["case_id"].astype("int64").astype(str)

# The Dataset Lookup tool drops empty fields: list them explicitly so the LLM states
# that they are missing instead of guessing a value
key_fields = ["revenu_principal", "taux_effort", "mensualite", "montant_credit",
              "retard_max_12m_jours", "nb_contrats_bureau", "tranche_age"]
df["donnees_manquantes"] = df[key_fields].apply(
    lambda r: ", ".join(c for c in key_fields if pd.isna(r[c]) or r[c] == "unknown") or "aucune",
    axis=1)

dataiku.Dataset("hc_case_profile").write_with_schema(df)
