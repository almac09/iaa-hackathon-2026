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


def _dmatrix(X: pd.DataFrame, expo: np.ndarray, base_rate: float, y: np.ndarray | None = None) -> xgb.DMatrix:
    # Exposure enters as a log offset. The base rate goes into the offset too: with base_margin set,
    # XGBoost does not add base_score, so otherwise the first trees would be spent learning the intercept.
    return xgb.DMatrix(X, label=y, base_margin=np.log(expo) + np.log(base_rate), enable_categorical=True)


def fit(X, y, expo, base_rate, power, params, n_rounds, seed, nthread, val=None, early_stopping=None) -> xgb.Booster:
    """Tweedie boosting with a log-exposure offset. `val` = (X, y, expo) enables early stopping on it."""
    full = {"objective": "reg:tweedie", "tweedie_variance_power": power, "eval_metric": f"tweedie-nloglik@{power}",
            "tree_method": "hist", "seed": seed, "nthread": nthread, **params}
    evals = [(_dmatrix(val[0], val[2], base_rate, val[1]), "val")] if val else []
    return xgb.train(full, _dmatrix(X, expo, base_rate, y), num_boost_round=n_rounds, evals=evals,
                     early_stopping_rounds=early_stopping, verbose_eval=False)


def predict(booster: xgb.Booster, X: pd.DataFrame, expo: np.ndarray, base_rate: float,
            n_trees: int | None = None) -> np.ndarray:
    """Expected claim cost for the row's exposure, using the first `n_trees` trees (all by default)."""
    return booster.predict(_dmatrix(X, expo, base_rate), iteration_range=(0, n_trees or 0))


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
