"""Mack's distribution-free chain ladder: reserve and its standard error (Mack 1993), for the level uncertainty (A25).

A triangle is a square array of cumulative amounts, rows = origin periods (oldest first), columns = development
periods; row i is observed up to column n - 1 - i and the rest is NaN. The standard error covers process and
parameter error of the chain ladder; there is no tail beyond the last development period.
"""
from __future__ import annotations

import numpy as np


def chain_ladder(tri: np.ndarray) -> dict:
    n = tri.shape[0]
    f = np.ones(n - 1)
    sigma2 = np.zeros(n - 1)
    col_sum = np.zeros(n - 1)
    for k in range(n - 1):
        rows = np.arange(n - 1 - k)                       # origins with column k + 1 observed
        c0, c1 = tri[rows, k], tri[rows, k + 1]
        use = c0 > 0
        col_sum[k] = c0[use].sum()
        f[k] = c1[use].sum() / col_sum[k] if col_sum[k] > 0 else 1.0
        if use.sum() > 1:
            sigma2[k] = (c0[use] * (c1[use] / c0[use] - f[k]) ** 2).sum() / (use.sum() - 1)
    # The last factor has a single observation: Mack's extrapolation.
    if n >= 3:
        a, b = sigma2[n - 3], sigma2[n - 4] if n >= 4 else sigma2[n - 3]
        sigma2[n - 2] = min(a ** 2 / b, b, a) if b > 0 else a
    full = tri.copy()
    for i in range(n):
        for k in range(n - 1 - i, n - 1):
            full[i, k + 1] = full[i, k] * f[k]
    latest = np.array([tri[i, n - 1 - i] for i in range(n)])
    return {"f": f, "sigma2": sigma2, "col_sum": col_sum, "full": full, "latest": latest,
            "ultimate": full[:, -1], "reserve": full[:, -1] - latest}


def mack_mse(cl: dict, origins) -> float:
    """Mean squared error of the total reserve over `origins` (indices into the triangle), with the covariance
    between origins that share estimated factors."""
    f, s2, S, full, ult = cl["f"], cl["sigma2"], cl["col_sum"], cl["full"], cl["ultimate"]
    n = full.shape[0]
    w = np.divide(s2, f ** 2, out=np.zeros_like(s2), where=f > 0)
    mse = 0.0
    origins = sorted(origins)
    for i in origins:
        ks = range(n - 1 - i, n - 1)
        mse += ult[i] ** 2 * sum(w[k] * (1 / full[i, k] + 1 / S[k]) for k in ks if full[i, k] > 0 and S[k] > 0)
    for a, i in enumerate(origins):
        for j in origins[a + 1:]:
            older = min(i, j)                              # the older origin needs the fewer projections
            ks = range(n - 1 - older, n - 1)
            mse += 2 * ult[i] * ult[j] * sum(w[k] / S[k] for k in ks if S[k] > 0)
    return float(mse)
