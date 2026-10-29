"""Helpers for the split three-line model (audit A16): GLM factor banding, the claim-history credibility
factor, the small/large split of the target and the generalised Pareto tail for large claims.

The GLMs themselves are models.fit_tweedie with models.design_matrix, as for the contents pack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import SplineTransformer

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
    """Turns raw columns into GLM inputs: numerics banded or smoothed, rare categorical levels merged.

    Numerics in `smooth` get a cubic B-spline basis instead of bands (knots at training quantiles, flat beyond the
    outer knots, which sit at the `knot_range` quantiles: the training range by default; one basis column dropped
    against the intercept) plus a missing indicator, so a curve of a few parameters replaces one free coefficient
    per band (A17). Everything is fitted on the training rows passed to the constructor and then applied unchanged.
    """

    def __init__(self, train: pd.DataFrame, numeric: list[str], categorical: list[str],
                 min_level_share: float = 0.005, n_bands: int = 10, smooth: list[str] = (), n_knots: int = 4,
                 knot_range: tuple[float, float] = (0, 1)):
        # knot_range: the quantiles of the outer knots, beyond which a curve is flat (the full range by default)
        self.categorical = categorical
        self.splines = {}
        for c in smooth:
            x = train[c].dropna().to_numpy(dtype=float)
            knots = np.unique(np.quantile(x, np.linspace(*knot_range, n_knots)))
            if len(knots) >= 3:  # too few distinct values for a curve: keep the column banded
                self.splines[c] = SplineTransformer(degree=3, knots=knots.reshape(-1, 1), extrapolation="constant",
                                                    include_bias=False).fit(x.reshape(-1, 1))
        self.has_missing = {c: bool(train[c].isna().any()) for c in self.splines}
        self.numeric = [c for c in numeric if c not in self.splines]
        self.edges = fit_bands(train, self.numeric, n_bands)
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

    def basis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spline columns for the smoothed numerics (all zero where the value is missing, flagged separately)."""
        out = {}
        for c, st in self.splines.items():
            x = df[c].to_numpy(dtype=float)
            miss = np.isnan(x)
            b = st.transform(np.where(miss, np.nanmedian(x) if (~miss).any() else 0.0, x).reshape(-1, 1))
            b[miss] = 0.0
            for j in range(b.shape[1]):
                out[f"{c}_s{j}"] = b[:, j]
            if self.has_missing[c]:
                out[f"{c}_missing"] = miss.astype(float)
        return pd.DataFrame(out, index=df.index)


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
