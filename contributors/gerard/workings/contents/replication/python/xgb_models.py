"""XGBoost helpers for the contents pack (audit A13): feature matrices, the exposure offset and the bootstrap.

Unlike models.py, this has no R twin: an R xgboost run would not agree with Python closely enough
to reconcile, so the model is checked against GLM A on the same split instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

import models

# Never valid as features: claim and target columns, and columns only known after the period.
POST_PERIOD = {"renewal_decision", "claim_cost"}
POST_PERIOD_PREFIXES = ("cost_", "n_")
# Excluded while the matching decision applies (raw column and its banded version).
EXCLUDED_BY_DECISION = {
    "D04_exclude_contacts": {"customer_service_contacts", "f_cs_contacts"},
    "D05_exclude_written_premium": {"written_premium", "earned_premium", "f_wp_band"},
}


def check_features(features: list[str], cfg: dict) -> None:
    for f in features:
        if f in POST_PERIOD or f.startswith(POST_PERIOD_PREFIXES):
            raise ValueError(f"{f} is a claim or post-period column and cannot be a feature.")
        for key, cols in EXCLUDED_BY_DECISION.items():
            if f in cols and cfg["decisions"][key]["apply"]:
                raise ValueError(f"{f} is excluded by {key}.")


def _as_level(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("MISSING")


def category_levels(train: pd.DataFrame, categorical: list[str]) -> dict[str, list[str]]:
    """Levels from the training rows, missing as its own level (D-06), in code-point order as in models.py."""
    return {c: sorted(_as_level(train[c]).unique()) for c in categorical}


def matrix(df: pd.DataFrame, numeric: list[str], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Numeric columns as float (NaN left to XGBoost); categoricals fixed to the training levels."""
    out = {n: df[n].astype(float).to_numpy() for n in numeric}
    for c, levels in categories.items():
        s = _as_level(df[c])
        out[c] = pd.Categorical(s.where(s.isin(levels)).to_numpy(dtype=object), categories=levels)
    return pd.DataFrame(out, index=df.index)


def _dmatrix(X: pd.DataFrame, expo: np.ndarray, base_rate: float, y: np.ndarray | None = None,
             extra: np.ndarray | None = None) -> xgb.DMatrix:
    # Exposure enters as a log offset. The base rate goes into the offset too: with base_margin set,
    # XGBoost does not add base_score, so otherwise the first trees would be spent learning the intercept.
    # `extra` is any further fixed log-scale term, such as the calendar-year trend.
    margin = np.log(expo) + np.log(base_rate) + (0.0 if extra is None else extra)
    return xgb.DMatrix(X, label=y, base_margin=margin, enable_categorical=True)


def fit(X, y, expo, base_rate, power, params, n_rounds, seed, nthread, val=None, early_stopping=None,
        extra=None) -> xgb.Booster:
    """Tweedie boosting with a log-exposure offset. `val` = (X, y, expo) enables early stopping on it."""
    full = {"objective": "reg:tweedie", "tweedie_variance_power": power, "eval_metric": f"tweedie-nloglik@{power}",
            "tree_method": "hist", "seed": seed, "nthread": nthread, **params}
    evals = [(_dmatrix(val[0], val[2], base_rate, val[1]), "val")] if val else []
    return xgb.train(full, _dmatrix(X, expo, base_rate, y, extra), num_boost_round=n_rounds, evals=evals,
                     early_stopping_rounds=early_stopping, verbose_eval=False)


def predict(booster: xgb.Booster, X: pd.DataFrame, expo: np.ndarray, base_rate: float,
            n_trees: int | None = None, extra: np.ndarray | None = None) -> np.ndarray:
    """Expected claim cost for the row's exposure, using the first `n_trees` trees (all by default)."""
    return booster.predict(_dmatrix(X, expo, base_rate, extra=extra), iteration_range=(0, n_trees or 0))


def _year_slope(y, mu, yr, centre) -> float:
    """Slope of log(actual / expected) by year on (year - centre), weighted by expected cost."""
    g = pd.DataFrame({"y": y, "mu": mu, "yr": yr}).groupby("yr").sum()
    t, w = g.index.to_numpy(dtype=float) - centre, g.mu.to_numpy()
    z = np.log(g.y.to_numpy() / w)
    t_bar = np.average(t, weights=w)
    return float(np.sum(w * (t - t_bar) * (z - np.average(z, weights=w))) / np.sum(w * (t - t_bar) ** 2))


def fit_with_trend(X, y, expo, yr, base_rate, power, params, n_rounds, seed, nthread, n_iter=3):
    """Boosting plus a log-linear calendar-year term, fitted jointly by backfitting.

    Trees cannot extrapolate a year feature, so the trend sits in the offset as slope x (year - centre).
    Each pass fits the trees with the current slope, measures the trend left in actual/expected by year and
    adds it to the slope; a final fit uses the converged slope. Returns (booster, slope, centre, history).
    """
    yr = np.asarray(yr, dtype=float)
    centre = float(np.average(yr, weights=expo))
    slope, history = 0.0, []
    for i in range(n_iter + 1):
        extra = slope * (yr - centre)
        b = fit(X, y, expo, base_rate, power, params, n_rounds, seed, nthread, extra=extra)
        residual = _year_slope(y, predict(b, X, expo, base_rate, extra=extra), yr, centre)
        history.append({"iteration": i, "slope": slope, "residual_slope": residual})
        if i < n_iter:
            slope += residual
    return b, slope, centre, history


def paired_bootstrap_d2(y, pred_a, pred_b, p, n_boot, seed) -> np.ndarray:
    """D2(a) - D2(b) on `n_boot` row resamples, both models scored on the same resample each time."""
    rng = np.random.default_rng(seed)
    dev_a, dev_b = models.tweedie_deviance(y, pred_a, p), models.tweedie_deviance(y, pred_b, p)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        null = models.tweedie_deviance(y[idx], np.full(len(idx), y[idx].mean()), p).sum()
        out[i] = (dev_b[idx].sum() - dev_a[idx].sum()) / null
    return out
