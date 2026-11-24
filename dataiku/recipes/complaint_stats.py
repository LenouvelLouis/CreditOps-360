# Zone 8 - Agents: complaint summary per product, used by the agents' Dataset
# Lookup tool (one row per French product label, plus a "tous" total row).
import dataiku
import pandas as pd

df = dataiku.Dataset("cfpb_triage").get_dataframe(
    columns=["complaint_id", "produit", "urgence", "montant", "motif"])
df["produit"] = df["produit"].fillna("autre")


def summarize(g: pd.DataFrame, produit: str) -> dict:
    motifs = g["motif"].dropna().value_counts()
    return {
        "produit": produit,
        "nb_reclamations": len(g),
        "nb_urgence_haute": int((g["urgence"] == "haute").sum()),
        "nb_urgence_moyenne": int((g["urgence"] == "moyenne").sum()),
        "nb_urgence_basse": int((g["urgence"] == "basse").sum()),
        "montant_moyen_usd": round(float(g["montant"].mean()), 0) if g["montant"].notna().any() else None,
        "motifs_frequents": ", ".join(motifs.index[:3]),
    }


rows = [summarize(g, p) for p, g in df.groupby("produit")]
rows.append(summarize(df, "tous"))
dataiku.Dataset("cfpb_stats").write_with_schema(pd.DataFrame(rows))
