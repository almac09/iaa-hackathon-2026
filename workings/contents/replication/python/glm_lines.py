"""Helpers for the split three-line model (audit A16): GLM factor banding, the claim-history credibility
factor, the small/large split of the target and the generalised Pareto tail for large claims.

The GLMs themselves are models.fit_tweedie with models.design_matrix, as for the contents pack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Claim-history groups of sql/12: (count column, expected-count column) per group.
HISTORY_GROUPS = {"own": ("h_own_n", "e_own"), "home": ("h_home_n", "e_home"), "cross": ("h_cross_n", "e_cross")}


def credibility_factor(n: np.ndarray, expected: np.ndarray, k: float) -> np.ndarray:
    """Poisson-gamma experience factor (k + n) / (k + E): n claims seen in the row's history against E expected
    by a model without history. A row with no history (n = E = 0) gets 1; k = inf switches the group off."""
    if not np.isfinite(k):
        return np.ones(len(n))
    return (k + np.nan_to_num(n)) / (k + np.nan_to_num(expected))


def fit_bands(train: pd.DataFrame, numeric: list[str], n_bands: int = 10) -> dict[str, np.ndarray]:
    """Inner quantile edges per numeric column from the training rows (duplicates dropped)."""
    qs = np.linspace(0, 1, n_bands + 1)[1:-1]
    return {c: np.unique(np.quantile(train[c].dropna().to_numpy(dtype=float), qs)) for c in numeric}


def _band(x: pd.Series, edges: np.ndarray) -> pd.Series:
    """'qNN' counts the edges strictly below the value, as sql/08 does; missing is its own level (D-06)."""
    v = x.to_numpy(dtype=float)
    lab = pd.Series(np.char.add("q", np.char.zfill(np.searchsorted(edges, v, side="left").astype(str), 2)),
                    index=x.index)
    return lab.where(~np.isnan(v), "MISSING")


class FactorSpec:
    """Turns raw columns into GLM factors: numerics banded and rare categorical levels merged.

    Everything is fitted on the training rows passed to the constructor and then applied unchanged.
    """

    def __init__(self, train: pd.DataFrame, numeric: list[str], categorical: list[str],
                 min_level_share: float = 0.005, n_bands: int = 10):
        self.numeric, self.categorical = numeric, categorical
        self.edges = fit_bands(train, numeric, n_bands)
        share = {c: train.groupby(train[c].astype("string").fillna("MISSING"), dropna=False).exposure.sum()
                 / train.exposure.sum() for c in categorical}
        self.keep = {c: set(s.index[s >= min_level_share]) for c, s in share.items()}

    def factors(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            out[c] = _band(df[c], self.edges[c])
        for c in self.categorical:
            s = df[c].astype("string").fillna("MISSING")
            out[c] = s.where(s.isin(self.keep[c]), "OTHER")
        return pd.DataFrame(out, index=df.index).astype(str)


def split_targets(claims: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Developed small and large parts per row. Each claim is split at `threshold` on its reported cost and both
    parts are then developed with the claim's factor, so development never pushes a claim over the threshold."""
    c = claims.net_cost.to_numpy()
    f = claims.dev_factor.to_numpy()
    parts = pd.DataFrame({"row_id": claims.row_id.to_numpy(), "small_dev": np.minimum(c, threshold) * f,
                          "excess_dev": np.maximum(c - threshold, 0) * f, "n_large": (c > threshold).astype(int)})
    return parts.groupby("row_id").sum()


def gpd_tail(claims: pd.DataFrame, threshold: float) -> dict:
    """Generalised Pareto fit to reported cost in excess of `threshold`. `adjustment` scales the empirical mean
    excess to the fitted one; with shape >= 1 the mean is infinite, so no adjustment is made and it is flagged."""
    x = claims.net_cost.to_numpy()
    x = x[x > threshold] - threshold
    shape, _, scale = stats.genpareto.fit(x, floc=0)
    fitted = scale / (1 - shape) if shape < 1 else np.inf
    return {"threshold": threshold, "n_large": len(x), "shape": shape, "scale": scale,
            "mean_excess_empirical": x.mean(), "mean_excess_fitted": fitted,
            "adjustment": fitted / x.mean() if np.isfinite(fitted) else 1.0, "infinite_mean": not np.isfinite(fitted)}
