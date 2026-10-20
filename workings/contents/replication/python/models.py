"""Metric, design matrix, one-way scoring and the GLM structures compared in audit A07 and A09.

R/run_all.R implements the same functions independently; reconcile/reconcile.py compares them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor


def tweedie_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    """Unit Tweedie deviance for 1 < p < 2 (valid for y = 0)."""
    return 2 * (np.power(y, 2 - p) / ((1 - p) * (2 - p)) - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p))


def d2_tweedie(y: np.ndarray, pred: np.ndarray, p: float) -> float:
    """Competition metric: 1 - D(y, pred) / D(y, mean(y)), with the unweighted mean as baseline."""
    return 1 - tweedie_deviance(y, pred, p).sum() / tweedie_deviance(y, np.full_like(y, y.mean()), p).sum()


def factor_levels(train: pd.DataFrame, factors: list[str]) -> dict[str, list[str]]:
    """Sorted by code point (C locale) so R's radix sort gives the same order and reference level."""
    return {f: sorted(train[f].unique()) for f in factors}


def design_matrix(df: pd.DataFrame, levels: dict[str, list[str]], numeric: list[str]) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for f, lv in levels.items():
        x = df[f].to_numpy()
        for level in lv[1:]:  # first level is the reference
            cols.append((x == level).astype(float))
            names.append(f"{f}={level}")
    for n in numeric:
        cols.append(df[n].to_numpy(dtype=float))
        names.append(n)
    return np.column_stack(cols), names


def oneway_table(df: pd.DataFrame, f: str) -> pd.DataFrame:
    g = df.groupby(f).agg(exposure=("exposure", "sum"), n_claims=("n_claims", "sum"), cost=("claim_cost", "sum"))
    overall = df.claim_cost.sum() / df.exposure.sum()
    g["exposure_pct"] = 100 * g.exposure / g.exposure.sum()
    g["freq_per_1000"] = 1000 * g.n_claims / g.exposure
    g["severity"] = g.cost / g.n_claims.where(g.n_claims > 0)
    g["loss_cost"] = g.cost / g.exposure
    g["relativity"] = g.loss_cost / overall
    for label, part in (("rel_2021_22", df[df.yr <= 2022]), ("rel_2023_24", df[df.yr >= 2023])):
        p = part.groupby(f).agg(e=("exposure", "sum"), c=("claim_cost", "sum"))
        g[label] = (p.c / p.e) / (part.claim_cost.sum() / part.exposure.sum())
    return g.reset_index().rename(columns={f: "level"}).assign(factor=f)


def oneway_d2_gain(fit: pd.DataFrame, val: pd.DataFrame, f: str, p: float) -> float:
    """Out-of-time gain over exposure-only: level rates from `fit` applied to `val` as rate x exposure."""
    base = fit.claim_cost.sum() / fit.exposure.sum()
    r = fit.groupby(f).agg(e=("exposure", "sum"), c=("claim_cost", "sum"))
    rate = val[f].map(r.c / r.e).fillna(base).to_numpy()
    y = val.claim_cost.to_numpy()
    e = val.exposure.to_numpy()
    return d2_tweedie(y, rate * e, p) - d2_tweedie(y, base * e, p)


# GLMs. Exposure enters exactly as a log offset would: the target is y / exposure with sample
# weight exposure^(2 - p), because the Tweedie unit deviance satisfies d(cy, c mu) = c^(2-p) d(y, mu).

# Newton-Cholesky is exact for n >> p and converges in a handful of iterations (audit A11: L-BFGS
# and Newton reach the same objective, but Newton gets there reliably on near-separated levels).
SOLVER = dict(solver="newton-cholesky", max_iter=1000, tol=1e-10)


def fit_tweedie(X, y, expo, p, alpha):
    m = TweedieRegressor(power=p, link="log", alpha=alpha, **SOLVER)
    m.fit(X, y / expo, sample_weight=expo ** (2 - p))
    return m


def fit_poisson(X, n, expo, alpha):
    m = PoissonRegressor(alpha=alpha, **SOLVER)
    m.fit(X, n / expo, sample_weight=expo)
    return m


def fit_gamma(X, mean_sev, n, alpha):
    m = GammaRegressor(alpha=alpha, **SOLVER)
    m.fit(X, mean_sev, sample_weight=n)
    return m


def coef_frame(component: str, m, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"component": component, "param": ["(intercept)"] + names,
                         "coef": np.r_[m.intercept_, m.coef_]})
