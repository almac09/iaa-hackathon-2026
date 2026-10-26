"""A31 variation v6: buildings-specific treatment of the portable developed model (C0).

Buildings only; contents and motor return C0 unchanged. Two multiplicative changes to C0, then a rebalance to C0's
total on the scored rows (the 2025 level is set separately, so this variation changes ranking only):

1. Missing-level neutralisation (outcome-free). Nine buildings fields are missing on about 12% of emerging rows,
   at random, against about 2.5% in the developed market, where missingness partly carries information. C0 still
   applies the developed missing-level relativities (river distance x1.10, soil x0.93, energy rating x1.05, ...).
   Their average effect is measured by regressing log(C0 rate) on the nine missing indicators (exposure-weighted,
   with year intercepts) over the training rows' features only, and divided out. Flood zone is excluded: it is
   missing on about 3% of rows in both markets and its missingness is informative in the developed data.
2. Construction recalibration. A Tweedie (p = 1.5) GLM with the neutralised C0 as offset, unpenalised year
   intercepts and ridge-penalised construction levels, fitted to the training target capped at CAP per row, so the
   few fire total losses do not drive the correction. CAP and LAMBDA were chosen by customer-grouped five-fold
   cross-validation within emerging 2023 (cap 25k beat 10k, 50k and none); LAMBDA is set above the CV optimum
   (1e3) on purpose, keeping about 80% of the CV gain with smaller corrections, because buildings is noisy and
   within-year CV cannot see year-specific shocks.

Wildfire: C0 is flat in wildfire_index (its fitted gradient is x1.000 per 0.1 in both markets, and tree splits
cannot extrapolate), so clamping at the developed maximum and leaving it give the same C0. This variation does
not extrapolate wildfire, does not use region, written_premium, gender, customer_since_year or
customer_service_contacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

P = 1.5
LINE = "buildings"
# Frozen configuration (chosen on emerging 2023 only).
CAP = 25_000.0
LAMBDA = 1e4
MISSING_FIELDS = ("days_unoccupied", "smoke_alarms", "marital_status", "household_size", "years_at_address",
                  "distance_to_river_km", "soil_type", "roof_type", "energy_rating")
CONSTRUCTION_LEVELS = ("brick", "non_standard", "stone", "timber")
MIN_EXPOSURE = 1e-3

FINAL = {"neutralise": True, "construction": True, "cap": CAP, "lam": LAMBDA,
         "extra": None, "extra_cap": None, "extra_lam": None}


def _rate_log(d: pd.DataFrame) -> np.ndarray:
    return np.log(d.c0.to_numpy(float) / np.maximum(d.exposure_plain.to_numpy(float), MIN_EXPOSURE))


def _missing_matrix(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([d[c].isna().to_numpy(float) for c in MISSING_FIELDS])


def _year_matrix(d: pd.DataFrame, years: list[int]) -> np.ndarray:
    yr = d.yr.to_numpy()
    return np.column_stack([(yr == y).astype(float) for y in years])


def missing_effects(train: pd.DataFrame) -> np.ndarray:
    """C0's average log relativity for each missing indicator, from the training rows' features (no outcomes)."""
    years = sorted(train.yr.unique())
    X = np.column_stack([_year_matrix(train, years), _missing_matrix(train)])
    w = np.maximum(train.exposure_plain.to_numpy(float), MIN_EXPOSURE)
    xw = X * w[:, None]
    coef = np.linalg.solve(X.T @ xw, xw.T @ _rate_log(train))
    return coef[len(years):]


def _fit_offset_tweedie(X: np.ndarray, pen: np.ndarray, y: np.ndarray, base: np.ndarray, lam: float) -> np.ndarray:
    """Minimise the Tweedie deviance of y against base * exp(X b) plus lam * sum(pen * b^2)."""
    lb = np.log(base)

    def f(b):
        eta = lb + X @ b
        m1, m2 = np.exp((1 - P) * eta), np.exp((2 - P) * eta)
        val = 2 * (-y * m1 / (1 - P) + m2 / (2 - P)).sum() + lam * (pen * b * b).sum()
        g = 2 * (-y * m1 + m2)
        return val, X.T @ g + 2 * lam * pen * b

    b0 = np.zeros(X.shape[1])
    r = minimize(f, b0, jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    if not r.success:
        raise RuntimeError(f"correction fit did not converge: {r.message}")
    return r.x


def _onehot(s: pd.Series, levels) -> np.ndarray:
    v = s.astype(object).to_numpy()
    return np.column_stack([(v == lv).astype(float) for lv in levels])


def _extra_design(d: pd.DataFrame, name: str, ctx: dict) -> np.ndarray:
    """Design columns for labelled exploration terms (never used by FINAL)."""
    if name == "region":
        return _onehot(d.region, ctx["region_levels"])
    if name == "wildfire_hinge":  # wildfire beyond the developed maximum, per 0.1 (not clamped)
        return (np.maximum(d.wildfire_index.to_numpy(float) - ctx["wmax"], 0.0) * 10)[:, None]
    if name == "gender":
        return _onehot(d.gender, ["F", "M"])
    if name == "log_wp_rel":
        return (np.log(d.written_premium.to_numpy(float) / d.c0.to_numpy(float)) - ctx["wp_centre"])[:, None]
    raise KeyError(name)


def buildings(train: pd.DataFrame, score: pd.DataFrame, dev: pd.DataFrame | None, cfg: dict = FINAL,
              return_parts: bool = False):
    """Buildings predictions for the score rows under configuration cfg."""
    c0_tr, c0_sc = train.c0.to_numpy(float), score.c0.to_numpy(float)
    base_tr, base_sc = c0_tr.copy(), c0_sc.copy()
    parts = {}
    if cfg["neutralise"]:
        gam = missing_effects(train)
        base_tr = base_tr * np.exp(-_missing_matrix(train) @ gam)
        base_sc = base_sc * np.exp(-_missing_matrix(score) @ gam)
        parts["missing_relativities_removed"] = dict(zip(MISSING_FIELDS, np.exp(gam).round(4).tolist()))
    years = sorted(train.yr.unique())
    ctx = {}
    if cfg.get("extra"):  # context for the labelled exploration terms only; FINAL needs none of it
        ctx = {"region_levels": sorted(train.region.dropna().unique()),
               "wmax": float(dev.wildfire_index.max()) if dev is not None else None,
               "wp_centre": float(np.median(np.log(train.written_premium / train.c0)))}
    terms = []
    if cfg["construction"]:
        terms.append(("construction", cfg["cap"], cfg["lam"]))
    if cfg.get("extra"):
        terms.append((cfg["extra"], cfg["extra_cap"], cfg["extra_lam"]))
    y_tr = train.y_full.to_numpy(float)
    for name, cap, lam in terms:
        # each term is fitted in turn on the base so far, with its own cap and penalty
        yt = y_tr if cap is None else np.minimum(y_tr, cap)
        if name == "construction":
            Xt, Xs = _onehot(train.construction_type, CONSTRUCTION_LEVELS), _onehot(score.construction_type,
                                                                                     CONSTRUCTION_LEVELS)
        else:
            Xt, Xs = _extra_design(train, name, ctx), _extra_design(score, name, ctx)
        Y = _year_matrix(train, years)
        X = np.column_stack([Y, Xt])
        pen = np.concatenate([np.zeros(len(years)), np.ones(Xt.shape[1])])
        b = _fit_offset_tweedie(X, pen, yt, base_tr, lam)[len(years):]
        base_tr = base_tr * np.exp(Xt @ b)
        base_sc = base_sc * np.exp(Xs @ b)
        parts[f"{name}_log_coef"] = np.round(b, 4).tolist()
    pred = base_sc * (c0_sc.sum() / base_sc.sum())  # keep C0's total on the scored rows
    if not (np.all(np.isfinite(pred)) and np.all(pred > 0)):
        raise ValueError("buildings predictions must be finite and positive")
    return (pred, parts) if return_parts else pred


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    """line -> positive predictions aligned to score[line]; buildings changed, other lines C0 as is."""
    out = {ln: score[ln].c0.to_numpy(float) for ln in score}
    if LINE in score:
        out[LINE] = buildings(train[LINE], score[LINE], dev.get(LINE) if dev else None, FINAL)
    return out
