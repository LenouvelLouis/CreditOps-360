"""Model stability metrics: Gini per week, competition stability score, PSI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def gini_by_week(df: pd.DataFrame, week="WEEK_NUM", target="target", score="score") -> pd.Series:
    """Gini per week, weeks with a single class are skipped."""
    out = {}
    for w, g in df.groupby(week):
        if g[target].nunique() == 2:
            out[w] = gini(g[target], g[score])
    return pd.Series(out, name="gini").sort_index()


def stability_score(weekly_gini: pd.Series) -> float:
    """Competition metric: mean Gini, penalised by a falling trend and by volatility.

    score = mean(gini) + 88 * min(0, slope) - 0.5 * std(residuals)
    """
    x = weekly_gini.index.to_numpy(dtype=float)
    y = weekly_gini.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    return float(y.mean() + 88.0 * min(0.0, slope) - 0.5 * residuals.std())


def psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of a numeric variable.

    Bins are the quantiles of the reference sample. Rule of thumb:
    < 0.1 stable, 0.1 to 0.2 moderate shift, > 0.2 significant shift.
    """
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / max(len(expected), 1)
    a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
