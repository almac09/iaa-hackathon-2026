"""v7: ridge-penalised offset corrections on C0, fitted on capped (or robustly weighted) targets.

    log mu = log c0 + a_year + X b,  X = one-hot factors (numerics in five training-quantile bands),
    b ~ ridge, a_year unpenalised (one per training year, so a year-level A/E difference never leaks into b).

The target is the row's claim cost y_full, optionally capped at the line's 95th or 99th percentile of positive
claim cost in the training rows, or at the large-claim threshold, and then rescaled by one factor so the capped
total equals the uncapped total. Optionally the rows with extreme Pearson residuals are down-weighted (Huber,
positive side only). The ridge penalty is expressed in "rows of prior information": lam = 100 shrinks every
coefficient as if 100 average training rows (Fisher information mean(mu^(2-p))) supported b = 0.

Predictions keep C0's level on the training book: pred = c0 * exp(X b) * K, with K chosen so that
sum_train c0 exp(X b) K = sum_train c0. The 2025 level is set elsewhere; the correction changes only the ranking
(and the mix-driven level on a scored book whose mix differs from training).

Field handling: categorical levels with under 0.5% of training exposure are merged into one OTHER level; missing
values get their own level (shrunk towards zero, as they are missing at random); levels or values never seen in
training get a zero row, i.e. no correction. Numerics outside the training range fall into the end bands.
customer_since_year, renewal_count, customer_service_contacts, written_premium and gender are never used in the
technical model (written_premium and gender only in the labelled diagnostics in explore.py).

Result (explore.py; 19 configurations scored on A2024, every choice made by customer-grouped CV on 2023): no
correction, capped, robust or forward-selected, beat C0 on A2024 pooled. Capping made the corrections less harmful
(e.g. targeted + main factors: gain -0.0017 uncapped, -0.0001 capped at p95) but never turned one into a gain. The
corrections 2023 CV liked most (motor new business x1.54/x0.65, buildings age band, construction and region,
contents urban density and AD cover) were the most harmful in 2024: the effects are year-specific, which a
within-year CV cannot see. Under the rule decided before scoring, recorded here afterwards (a line keeps its CV-chosen correction only if it is
not worse than C0 on A2024 level-matched D2), every line reverts to C0. CV_CHOICE keeps the rejected configuration.
Labelled diagnostics only (owner decisions, never adopted): C0 + gender +0.0016 [-0.0007, +0.0039]; C0 + written
premium relative to C0 -0.0059.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = 1.5
LARGE_THRESHOLD = {"buildings": 50000.0, "contents": None, "motor": 50000.0}  # meta.json excess_threshold
AGE = {"buildings": "policyholder_age_band", "contents": "policyholder_age_band", "motor": "driver_age_band"}
TARGETED = {ln: ["region", "new_business", AGE[ln], "channel"] for ln in AGE}
MAIN = {  # the A30 C2b main rating factors (numeric, categorical)
    "buildings": (["flood_zone", "sum_insured", "elevation_m", "year_built", "distance_to_river_km"],
                  ["construction_type", "soil_type", "occupancy", "energy_rating", "urban_density"]),
    "contents": (["elevation_m", "distance_to_river_km", "log_si"],
                 ["f_flood_zone", "f_tenure", "f_prior_claims", "f_vol_xs", "f_security", "f_hvi", "f_ad_cover",
                  "f_urban", "f_prop_type"]),
    "motor": (["ncb_years", "years_licensed", "vehicle_age", "vehicle_value_group", "voluntary_excess",
               "prior_claims_5y"],
              ["cover_type", "urban_density", "vehicle_use", "payment_method", "marketing_segment"]),
}
# A30's buildings list also had dry_summer_index, freezing_days and rainfall_mm. They are weather-year values (book
# means move year to year; dry_summer_index correlates only 0.59 for the same address 2023 v 2024; PSI on 2023 bands
# 0.74 / 0.28 / 0.03 against 2024 and 0.46 / 0.37 / 0.40 against 2025), so bands fixed on one year mean something else
# the next: excluded from the corrections (C0 keeps them).
EXCLUDED = {"buildings": ["dry_summer_index", "freezing_days", "rainfall_mm"], "contents": [], "motor": []}
POOL = {ln: (MAIN[ln][0], MAIN[ln][1] + TARGETED[ln]) for ln in AGE}
NEVER = {"customer_service_contacts", "customer_since_year", "renewal_count", "written_premium", "gender"}
N_BANDS = 5
MIN_SHARE = 0.005
MAX_LOG_CORRECTION = 3.0  # safety clip on X b; reported if it ever binds

# The 2023 CV choice per line (largest out-of-fold gain over the whole grid and the forward selections), frozen for
# reference. It lost to C0 on A2024 on every line (level-matched D2 0.0851/0.0913/0.0073 against C0's
# 0.1087/0.0972/0.0270), so it is not used.
CV_CHOICE: dict[str, dict] = {
    "buildings": dict(numeric=[], categorical=["policyholder_age_band", "construction_type", "region"], cap="thr",
                      lam=300.0, robust=None),
    "contents": dict(numeric=[], categorical=["f_urban", "f_ad_cover"], cap=None, lam=1000.0, robust=None),
    "motor": dict(numeric=["voluntary_excess"], categorical=["new_business"], cap=None, lam=300.0, robust=None),
}
# Frozen final configuration. None = C0 unchanged for that line (the pre-declared guard rejected every correction).
FINAL: dict[str, dict | None] = {"buildings": None, "contents": None, "motor": None}


# ----------------------------------------------------------------------------------------------- encoding
class Encoder:
    """One-hot design on banded numerics and merged categoricals, fixed on the training rows."""

    def __init__(self, train: pd.DataFrame, numeric: list[str], categorical: list[str]):
        assert not set(numeric + categorical) & (NEVER - {"written_premium", "gender"}), "forbidden field"
        self.numeric, self.categorical = list(numeric), list(categorical)
        e = train.exposure_plain.to_numpy(float)
        self.edges, self.levels, self.keep, self.seen = {}, {}, {}, {}
        for c in self.numeric:
            x = train[c].to_numpy(float)
            ok = ~np.isnan(x)
            self.edges[c] = np.unique(np.quantile(x[ok], np.linspace(0, 1, N_BANDS + 1)[1:-1])) if ok.any() else np.array([])
            self.levels[c] = list(np.unique(self._num(c, train[c])))
        for c in self.categorical:
            s = self._raw_cat(train[c])
            share = pd.Series(e).groupby(s.to_numpy()).sum() / e.sum()
            self.seen[c] = set(share.index)
            self.keep[c] = set(share.index[(share >= MIN_SHARE) | (share.index == "__NA__")])
            self.levels[c] = sorted(set(self._cat(c, train[c])))
        self.columns = [(c, lv) for c in self.numeric + self.categorical for lv in self.levels[c]]

    @staticmethod
    def _raw_cat(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("__NA__").reset_index(drop=True)

    def _num(self, c: str, s: pd.Series) -> np.ndarray:
        x = s.to_numpy(float)
        b = np.searchsorted(self.edges[c], x, side="right").astype(str)
        return np.where(np.isnan(x), "__NA__", b)

    def _cat(self, c: str, s: pd.Series) -> np.ndarray:
        """Kept level as is; a level seen in training but rare -> OTHER; a level never seen -> UNSEEN (zero row)."""
        r = self._raw_cat(s)
        out = np.where(r.isin(self.keep[c]), r.to_numpy(), "__OTHER__").astype(object)
        out[~r.isin(self.seen[c]).to_numpy()] = "__UNSEEN__"
        return out.astype(str)

    def matrix(self, d: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(d), len(self.columns)))
        j = 0
        for c in self.numeric + self.categorical:
            v = self._num(c, d[c]) if c in self.numeric else self._cat(c, d[c])
            for lv in self.levels[c]:
                X[:, j] = v == lv
                j += 1
        return X


# ------------------------------------------------------------------------------------------------ fitting
def _loss_grad_hess(y, mu, family):
    if family == "poisson":
        return mu - y * np.log(mu), mu - y, mu
    return 2 * y * mu ** -0.5 + 2 * mu ** 0.5, mu ** 0.5 - y * mu ** -0.5, 0.5 * mu ** 0.5 + 0.5 * y * mu ** -0.5


def fit_offset(X, G, y, offset, lam, w=None, family="tweedie", beta0=None, max_iter=50, tol=1e-8):
    """Penalised Newton for log mu = offset + G a + X b with penalty 0.5 lam I0 ||b||^2, where I0 = mean(w mu0^(2-p))
    is one average row's information at the starting point (mu0 = exp(offset + G a0))."""
    n, kg = G.shape
    Z = np.hstack([G, X])
    w = np.ones(n) if w is None else w
    a0 = np.log(np.array([(w * y * g).sum() / (w * np.exp(offset) * g).sum() for g in G.T]))
    mu0 = np.exp(offset + G @ a0)
    info = float((w * (mu0 if family == "poisson" else mu0 ** (2 - P))).mean())
    pen = np.r_[np.zeros(kg), np.full(X.shape[1], lam * info)]
    beta = np.r_[a0, np.zeros(X.shape[1])] if beta0 is None else beta0.copy()

    def obj(b):
        mu = np.exp(np.clip(offset + Z @ b, -50, 50))
        return (w * _loss_grad_hess(y, mu, family)[0]).sum() + 0.5 * (pen * b * b).sum()

    f = obj(beta)
    for _ in range(max_iter):
        mu = np.exp(np.clip(offset + Z @ beta, -50, 50))
        _, g, hss = _loss_grad_hess(y, mu, family)
        grad = Z.T @ (w * g) + pen * beta
        H = (Z * (w * hss)[:, None]).T @ Z + np.diag(pen) + 1e-10 * np.eye(len(beta))
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-6:
            nb = beta - t * step
            fn = obj(nb)
            if fn <= f + 1e-12 * abs(f):
                break
            t *= 0.5
        beta, done = nb, abs(f - fn) <= tol * (abs(f) + 1)
        f = fn
        if done:
            break
    return beta[:kg], beta[kg:]


def cap_target(y: np.ndarray, cap: str | None, line: str) -> tuple[np.ndarray, float | None]:
    """Cap at the training 95th/99th percentile of positive cost ('p95', 'p99') or the large-claim threshold
    ('thr'), then rescale so the capped total equals the uncapped total."""
    if cap is None:
        return y, None
    if cap == "thr":
        c = LARGE_THRESHOLD[line]
        if c is None:
            return y, None
    else:
        c = float(np.quantile(y[y > 0], {"p95": 0.95, "p99": 0.99}[cap]))
    yc = np.minimum(y, c)
    return yc * (y.sum() / yc.sum()), c


def year_design(yr: np.ndarray, years: list) -> np.ndarray:
    return np.column_stack([(yr == t).astype(float) for t in years])


class Correction:
    """cfg: numeric, categorical (lists), cap (None/'p95'/'p99'/'thr'), lam (float), robust (None or Huber c),
    family ('tweedie'/'poisson')."""

    def __init__(self, line: str, cfg: dict):
        self.line, self.cfg = line, cfg

    def fit(self, d: pd.DataFrame, X: np.ndarray | None = None, enc: "Encoder | None" = None) -> "Correction":
        cfg = self.cfg
        self.enc = enc or Encoder(d, cfg["numeric"], cfg["categorical"])
        X = self.enc.matrix(d) if X is None else X
        y = d.y_full.to_numpy(float)
        yc, self.cap_value = cap_target(y, cfg.get("cap"), self.line)
        self.years = sorted(d.yr.unique())
        G = year_design(d.yr.to_numpy(), self.years)
        off = np.log(d.c0.to_numpy(float))
        fam = cfg.get("family", "tweedie")
        w = None
        a, b = fit_offset(X, G, yc, off, cfg["lam"], family=fam)
        if cfg.get("robust"):
            for _ in range(3):
                mu = np.exp(off + G @ a + X @ b)
                vpow = 1.0 if fam == "poisson" else P
                phi = float(((yc - mu) ** 2 / mu ** vpow).mean())
                r = (yc - mu) / np.sqrt(phi * mu ** vpow)
                w = np.where(r > cfg["robust"], cfg["robust"] / np.maximum(r, 1e-12), 1.0)
                a, b = fit_offset(X, G, yc, off, cfg["lam"], w=w, family=fam)
        self.b = b
        self.n_downweighted = 0 if w is None else int((w < 1).sum())
        lc = np.clip(X @ b, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)
        c0 = d.c0.to_numpy(float)
        self.K = float(c0.sum() / (c0 * np.exp(lc)).sum())
        return self

    def relativity(self, d: pd.DataFrame, X: np.ndarray | None = None) -> np.ndarray:
        X = self.enc.matrix(d) if X is None else X
        lc = X @ self.b
        self.clip_bound = int((np.abs(lc) > MAX_LOG_CORRECTION).sum())
        return np.exp(np.clip(lc, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)) * self.K

    def predict(self, d: pd.DataFrame, X: np.ndarray | None = None) -> np.ndarray:
        return d.c0.to_numpy(float) * self.relativity(d, X)


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    """line -> positive predictions aligned to score[line]; lines without a frozen correction return C0."""
    out = {}
    for line, s in score.items():
        cfg = FINAL.get(line)
        if cfg is None:
            out[line] = s.c0.to_numpy(float)
        else:
            out[line] = Correction(line, cfg).fit(train[line]).predict(s)
        assert np.all(np.isfinite(out[line])) and np.all(out[line] > 0), f"{line}: invalid predictions"
    return out
