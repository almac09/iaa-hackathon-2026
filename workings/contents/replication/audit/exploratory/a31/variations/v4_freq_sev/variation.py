"""A31 variation v4 (scratch): a frequency x severity two-stage model for the emerging market.

Per line, fitted on the developed market (2021-24) with C0's portable features:
  frequency   Poisson GLM for claim counts (n_claims), log-exposure offset, linear calendar-year term;
  severity    Gamma GLM for the mean small-claim severity min(cost / n, T) on rows with claims, weight n, year term;
  large load  for lines with a large-claim threshold T (buildings, motor): a compact Poisson GLM for the count of
              rows whose mean claim exceeds T, times the developed mean excess per such row (one constant a line);
              it trends with the small-claim frequency and severity slopes.
Numerics are cubic B-splines (4 knots at training quantiles, flat beyond the training range, missing flag), or
5 quantile bands for the large-claim GLMs, as in C0; rare categorical levels (< 0.5% of exposure) merge to OTHER.

Emerging stages, all estimated on `train` (emerging rows) only:
  uncalibrated  the developed model projected to the score year with its own trend;
  level         a frequency level (Poisson MLE of claim counts) and a severity level (Gamma MLE of small-claim
                severity), separately; the large load gets the frequency level times a Poisson-gamma credibility
                factor on large-claim counts, (n_obs + K) / (n_exp + K), K = LARGE_K. In a pure product model the two
                levels change only the level; they change the ranking only through the small/large mix;
  corrected     a ridge-penalised Poisson GLM on the emerging claim counts with the level-calibrated frequency as
                offset (corrections act on frequency only); the penalty is in claim-deviance units (sklearn alpha =
                lambda / expected claims), so one frozen lambda is the same prior whatever the size of `train`.
Missing values: fields missing in at least 5% of emerging rows but under 5% of developed ones (flood zone excepted,
whose missingness is informative) are missing at random in the emerging market (audit A30), so a missing value
takes the field's expected relativity over the emerging rows where it is known, not the developed MISSING level.

Fields whose meaning changes in 2025 are not used as they are: customer_since_year (equal to the policy year on every
new-business row, so 2025 lies outside the developed range on 76% of 2025 rows) enters buildings as tenure =
yr - customer_since_year; campaign_code (missing on 88% of 2023 rows, 17% of 2025 rows: it marks the broker book)
is dropped. Never used: customer_service_contacts (D-04), written_premium (D-05) and gender (D-36), except where a
diagnostic switch says so (never in the frozen configuration).

Frozen configuration (SETTINGS, fixed by rules set on 2023 before any 2024 scoring): buildings and motor use v4
(motor with the emerging frequency correction), contents returns C0. On A2024 it scores pooled level-matched D2
0.11978 against C0's 0.12012 (no ranking gain); its raw pooled D2 is 0.11537 against 0.11000, which comes from the level.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.preprocessing import SplineTransformer

P = 1.5
LINES = ("buildings", "contents", "motor")
SOLVER = dict(solver="newton-cholesky", max_iter=300, tol=1e-8)
NEVER = {"customer_service_contacts"}  # D-04: claim-driven in training (leakage); never a feature
DIAGNOSTIC_ONLY = {"written_premium", "gender"}  # D-05, D-36: owner decisions; labelled diagnostics only
T_SCALE = 10.0  # the year term enters as 10 t, so the ridge penalty barely touches the trend

BUILDINGS_SPLINE = ["sum_insured", "days_unoccupied", "voluntary_excess", "missed_payments_12m", "renewal_count",
                    "days_quote_to_inception", "discount_applied", "household_size", "prior_claims_5y",
                    "licence_points", "years_at_address", "tenure", "distance_to_river_km", "elevation_m",
                    "year_built", "bedrooms", "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
                    "wildfire_index", "rainfall_mm", "mean_temperature_c", "x_clay_dry", "x_fz3_rain"]
BUILDINGS_CAT = ["flood_zone", "construction_type", "occupancy", "smoke_alarms", "accidental_damage_cover", "mortgage",
                 "channel", "payment_method", "new_business", "marital_status", "email_domain", "marketing_consent",
                 "document_preference", "previous_insurer", "branch_id", "marketing_segment", "urban_density",
                 "soil_type", "property_type", "roof_type", "basement", "trees_within_5m", "listed_building",
                 "energy_rating", "solar_panels", "policyholder_age_band"]

# The developed model per line: C0's portable features (buildings: C0's XGBoost list, with tenure for
# customer_since_year and without campaign_code), C0's large-claim features and threshold.
MODEL = {
    "buildings": {"spline": BUILDINGS_SPLINE, "band": [], "categorical": BUILDINGS_CAT, "linear": [],
                  "threshold": 50000.0,
                  "large": {"spline": ["sum_insured", "elevation_m", "year_built", "distance_to_river_km"], "band": [],
                            "categorical": ["flood_zone", "soil_type", "construction_type"],
                            "linear": ["x_clay_dry", "x_fz3_rain"]}},
    "contents": {"spline": ["elevation_m", "distance_to_river_km"], "band": [],
                 "categorical": ["f_flood_zone", "f_tenure", "f_segment", "f_prior_claims", "f_missed_pay", "f_vol_xs",
                                 "f_security", "f_hvi", "f_ad_cover", "f_bike", "f_wfh", "f_lodgers", "f_urban",
                                 "f_prop_type", "policyholder_age_band"],
                 "linear": ["log_si"], "threshold": None, "large": None},
    "motor": {"spline": ["ncb_years", "years_licensed", "vehicle_age", "vehicle_value_group", "voluntary_excess",
                         "prior_claims_5y", "missed_payments_12m"], "band": [],
              "categorical": ["marketing_segment", "cover_type", "urban_density", "channel", "vehicle_use",
                              "payment_method", "driver_age_band"],
              "linear": [], "threshold": 50000.0,
              "large": {"spline": [], "band": ["ncb_years", "years_licensed"], "categorical": ["driver_age_band"],
                        "linear": []}},
}
AGE_BAND = {"buildings": "policyholder_age_band", "contents": "policyholder_age_band", "motor": "driver_age_band"}

# Frozen configuration (set after exploring; see explore.py). Developed ridge alphas are sklearn alphas chosen on
# developed 2021-23 -> 2024 validation; LAMBDA is the emerging correction penalty in claim-deviance units.
SETTINGS = {
    "alpha": {"buildings": {"freq": 1e-6, "sev": 3e-2, "large": 1e-6},
              "contents": {"freq": 1e-5, "sev": 3e-3, "large": None},
              "motor": {"freq": 1e-6, "sev": 1e-4, "large": 1e-5}},
    "mar": True,          # missing-at-random marginalisation of emerging missing values
    "level": True,        # emerging frequency / severity / large levels
    "LARGE_K": 10.0,      # credibility constant for the large-claim level, in claims (an assumption)
    # Frozen by the rule fixed on 2023 before any 2024 scoring (explore_fit23.py): a line takes the correction set and
    # lambda with the best out-of-fold level-matched Tweedie gain on 2023, provided its out-of-fold Poisson count gain
    # is positive too; buildings and contents have none that qualifies.
    "corr": {"buildings": None, "contents": None,
             "motor": {"categorical": ["region", "new_business", "driver_age_band", "channel", "occupation_job"],
                       "lambda": 10.0}},
    # A line uses v4 only if the developed freq x sev model beats C0 on emerging 2023 (level-matched D2, both out of
    # sample there): buildings 0.0728 v 0.0690, motor 0.0387 v 0.0361; contents 0.0814 v 0.0864 keeps C0.
    "lines": ("buildings", "motor"),
}


# ----------------------------------------------------------------------------------------------- preparation
def occupation_job(s: pd.Series) -> pd.Series:
    """About 80 raw occupation strings for about 20 jobs: lower case, prefix 'self-employed' removed."""
    v = s.astype("string").str.lower().str.strip()
    v = v.str.replace(r"^self[- ]employed\s+", "", regex=True).str.replace(r"\s+", " ", regex=True)
    return v


def prepare(d: pd.DataFrame, threshold) -> pd.DataFrame:
    """Derived fields and the frequency / severity targets (targets only where outcomes exist)."""
    out = pd.DataFrame(index=d.index)
    out["yr"] = d["yr"].to_numpy(dtype=float)
    out["E"] = d["exposure_plain"].to_numpy(dtype=float)
    if "customer_since_year" in d:
        out["tenure"] = (d["yr"] - d["customer_since_year"]).to_numpy(dtype=float)
    if "sum_insured" in d and "log_si" not in d:
        out["log_si"] = np.log(d["sum_insured"].clip(lower=1).to_numpy(dtype=float))
    if "occupation" in d:
        out["occupation_job"] = occupation_job(d["occupation"])
        out["self_employed"] = d["occupation"].astype("string").str.lower().str.startswith("self").astype("string")
    if "written_premium" in d:
        out["log_wp"] = np.log(d["written_premium"].clip(lower=1).to_numpy(dtype=float))
    if "y_full" in d and d["y_full"].notna().all():
        y = d["y_full"].to_numpy(dtype=float)
        n = d["n_claims"].to_numpy(dtype=float)
        mean = np.divide(y, n, out=np.zeros_like(y), where=n > 0)
        small_mean = mean if threshold is None else np.minimum(mean, threshold)
        out["y"], out["n"] = y, n
        out["sev"] = small_mean                      # mean small-claim severity on rows with claims
        out["excess"] = np.clip(y - n * small_mean, 0, None)
        out["n_large"] = (out["excess"] > 0).astype(float)
    return out


def column(d: pd.DataFrame, extra: pd.DataFrame, c: str) -> pd.Series:
    if c in NEVER:
        raise ValueError(f"{c} is never a feature (D-04)")
    return extra[c] if c in extra else d[c]


# ----------------------------------------------------------------------------------------------- design matrix
class Spec:
    """Turns raw columns into a sparse GLM design, fitted on reference rows and then applied unchanged. Columns are
    grouped by field so a field's contribution to the linear predictor can be recomputed (MAR marginalisation)."""

    def __init__(self, d, extra, w, spline=(), band=(), categorical=(), linear=(), n_knots=4, n_bands=5,
                 min_share=0.005):
        self.splines, self.has_missing, self.fill = {}, {}, {}
        band = list(band)
        for c in spline:
            x = column(d, extra, c).to_numpy(dtype=float)
            ok = ~np.isnan(x)
            knots = np.unique(np.quantile(x[ok], np.linspace(0, 1, n_knots)))
            if len(knots) < 3:  # too few distinct values for a curve: band it instead
                band.append(c)
                continue
            self.splines[c] = SplineTransformer(degree=3, knots=knots.reshape(-1, 1), extrapolation="constant",
                                                include_bias=False).fit(x[ok].reshape(-1, 1))
            self.has_missing[c] = bool((~ok).any())
            self.fill[c] = float(np.median(x[ok]))
        self.edges = {c: np.unique(np.quantile(column(d, extra, c).dropna().to_numpy(dtype=float),
                                               np.linspace(0, 1, n_bands + 1)[1:-1])) for c in band}
        self.linear = list(linear)
        self.lin_fill = {c: float(np.nanmedian(column(d, extra, c).to_numpy(dtype=float))) for c in self.linear}
        self.levels = {}
        for c in list(categorical) + list(self.edges):
            s = self.labels(d, extra, c)
            share = pd.Series(w).groupby(s.to_numpy()).sum() / w.sum()
            keep = list(share.index[share >= min_share])
            other = bool((share < min_share).any())
            ref = share[keep].idxmax()  # the largest level is the reference
            self.levels[c] = {"ref": ref, "cols": sorted(k for k in keep if k != ref) + (["OTHER"] if other else []),
                              "known": set(keep)}
        self.groups = None

    def labels(self, d, extra, c) -> pd.Series:
        if c in self.edges:
            v = column(d, extra, c).to_numpy(dtype=float)
            lab = pd.Series(np.char.add("q", np.searchsorted(self.edges[c], v, side="left").astype(str)), index=d.index)
            return lab.where(~np.isnan(v), "MISSING")
        s = column(d, extra, c)
        return s.astype("string").fillna("MISSING").astype(str)

    def missing(self, d, extra, c) -> np.ndarray:
        s = column(d, extra, c)
        return (s.isna() | (s.astype("string") == "MISSING")).to_numpy()

    def design(self, d, extra) -> sp.csr_matrix:
        blocks, groups, j = [], {}, 0
        n = len(d)
        for c, st in self.splines.items():
            x = column(d, extra, c).to_numpy(dtype=float)
            miss = np.isnan(x)
            b = st.transform(np.where(miss, self.fill[c], x).reshape(-1, 1))
            b[miss] = 0.0
            if self.has_missing[c]:
                b = np.column_stack([b, miss.astype(float)])
            blocks.append(sp.csr_matrix(b))
            groups[c] = np.arange(j, j + b.shape[1])
            j += b.shape[1]
        for c in self.linear:
            x = column(d, extra, c).to_numpy(dtype=float)
            blocks.append(sp.csr_matrix(np.where(np.isnan(x), self.lin_fill[c], x).reshape(-1, 1)))
            groups[c] = np.array([j])
            j += 1
        for c, lv in self.levels.items():
            s = self.labels(d, extra, c).to_numpy()
            s = np.where(np.isin(s, list(lv["known"])), s, "OTHER" if "OTHER" in lv["cols"] else lv["ref"])
            idx = {k: i for i, k in enumerate(lv["cols"])}
            col = np.array([idx.get(v, -1) for v in s])
            rows = np.nonzero(col >= 0)[0]
            blocks.append(sp.csr_matrix((np.ones(len(rows)), (rows, col[rows])), shape=(n, len(lv["cols"]))))
            groups[c] = np.arange(j, j + len(lv["cols"]))
            j += len(lv["cols"])
        self.groups = groups
        return sp.hstack(blocks, format="csr") if blocks else sp.csr_matrix((n, 0))


class GLM:
    """Poisson or Gamma GLM (log link) on a Spec design with an optional linear year term (t = yr - centre)."""

    def __init__(self, spec: Spec, family: str, alpha: float, trend: bool, centre: float = 0.0):
        self.spec, self.family, self.alpha, self.trend, self.centre = spec, family, alpha, trend, centre

    def _x(self, d, extra, yr=None):
        X = self.spec.design(d, extra)
        if self.trend:
            t = (extra["yr"].to_numpy() if yr is None else np.full(len(d), float(yr))) - self.centre
            X = sp.hstack([X, sp.csr_matrix(T_SCALE * t.reshape(-1, 1))], format="csr")
        return X

    def fit(self, d, extra, y, w):
        Model = PoissonRegressor if self.family == "poisson" else GammaRegressor
        self.model = Model(alpha=self.alpha, **SOLVER).fit(self._x(d, extra), y, sample_weight=w)
        self.slope = T_SCALE * float(self.model.coef_[-1]) if self.trend else 0.0  # per year
        return self

    def eta(self, d, extra, yr=None, mar: dict | None = None) -> np.ndarray:
        """Linear predictor; with `mar`, a missing value in a MAR field takes that field's constant instead."""
        X = self._x(d, extra, yr)
        eta = X @ self.model.coef_ + self.model.intercept_
        for c, const in (mar or {}).items():
            if c not in self.spec.groups:
                continue
            miss = self.spec.missing(d, extra, c)
            if miss.any():
                cols = self.spec.groups[c]
                contrib = X[:, cols] @ self.model.coef_[cols]
                eta = eta + np.where(miss, const - contrib, 0.0)
        return eta

    def mar_constants(self, d, extra, w, fields) -> dict:
        """log of the exposure-weighted mean relativity over rows where the field is known."""
        X = self.spec.design(d, extra)
        out = {}
        for c in fields:
            if c not in self.spec.groups:
                continue
            known = ~self.spec.missing(d, extra, c)
            if known.sum() == 0:
                continue
            cols = self.spec.groups[c]
            contrib = X[known][:, cols] @ self.model.coef_[cols]
            out[c] = float(np.log(np.average(np.exp(contrib), weights=w[known])))
        return out


# ----------------------------------------------------------------------------------------------- developed model
class DevModel:
    def __init__(self, line: str, settings: dict):
        self.line, self.cfg, self.s = line, MODEL[line], settings
        self.T = self.cfg["threshold"]

    def fields(self) -> list[str]:
        c = self.cfg
        out = c["spline"] + c["band"] + c["categorical"] + c["linear"]
        if c["large"]:
            out += c["large"]["spline"] + c["large"]["band"] + c["large"]["categorical"] + c["large"]["linear"]
        return list(dict.fromkeys(out))

    def fit(self, dev: pd.DataFrame) -> "DevModel":
        x = prepare(dev, self.T)
        E = x.E.to_numpy()
        self.centre = float(np.average(x.yr, weights=E))
        c, a = self.cfg, self.s["alpha"][self.line]
        spec = Spec(dev, x, E, c["spline"], c["band"], c["categorical"], c["linear"])
        self.freq = GLM(spec, "poisson", a["freq"], True, self.centre).fit(dev, x, x.n.to_numpy() / E, E)
        cl = (x.n > 0) & (x.y > 0)
        dc, xc = dev[cl.to_numpy()], x[cl]
        spec_s = Spec(dc, xc, xc.n.to_numpy(), c["spline"], c["band"], c["categorical"], c["linear"])
        self.sev = GLM(spec_s, "gamma", a["sev"], True, self.centre).fit(dc, xc, xc.sev.to_numpy(), xc.n.to_numpy())
        if self.T is not None:
            lg = c["large"]
            spec_l = Spec(dev, x, E, lg["spline"], lg["band"], lg["categorical"], lg["linear"])
            self.large = GLM(spec_l, "poisson", a["large"], False).fit(dev, x, x.n_large.to_numpy() / E, E)
            self.mean_excess = float(x.excess.sum() / x.n_large.sum())
            # the large-claim GLM has no year term: centre its level on the developed exposure-weighted mean year
        self.mar = {}
        return self

    def set_mar(self, train: pd.DataFrame, dev: pd.DataFrame):
        """MAR fields: >= 5% missing in emerging train, < 5% in developed, flood zone excepted (as E5)."""
        self.mar = {}
        if not self.s["mar"]:
            return
        xt = prepare(train, self.T)
        fields = []
        for c in self.fields():
            if "flood_zone" in c or c not in train.columns or c not in dev.columns:
                continue
            miss = lambda s: float((s.isna() | (s.astype("string") == "MISSING")).mean())  # noqa: E731
            if miss(train[c]) >= 0.05 and miss(dev[c]) < 0.05:
                fields.append(c)
        self.mar_fields = fields
        w = xt.E.to_numpy()
        self.mar = {"freq": self.freq.mar_constants(train, xt, w, fields),
                    "sev": self.sev.mar_constants(train, xt, w, fields)}
        if self.T is not None:
            self.mar["large"] = self.large.mar_constants(train, xt, w, fields)

    def components(self, d: pd.DataFrame, yr=None) -> dict:
        """Per row: frequency rate f, small severity s, large-claim count rate fl and the large cost rate L (per
        unit exposure) at `yr` (None: each row's own year)."""
        x = prepare(d, self.T)
        f = np.exp(self.freq.eta(d, x, yr, self.mar.get("freq")))
        s = np.exp(self.sev.eta(d, x, yr, self.mar.get("sev")))
        out = {"x": x, "f": f, "s": s, "fl": np.zeros(len(d)), "L": np.zeros(len(d))}
        if self.T is not None:
            t = (x.yr.to_numpy() if yr is None else np.full(len(d), float(yr))) - self.centre
            fl = np.exp(self.large.eta(d, x, None, self.mar.get("large")))
            out["fl"] = fl * np.exp(self.freq.slope * t)
            out["L"] = out["fl"] * self.mean_excess * np.exp(self.sev.slope * t)
        return out


# ----------------------------------------------------------------------------------------------- emerging stages
def corr_spec(train, xt, w, corr):
    return Spec(train, xt, w, corr.get("spline", []), corr.get("band", []), corr.get("categorical", []),
                corr.get("linear", []), n_bands=5)


def fit_correction(train, xt, offset_count, corr, lam):
    """Ridge Poisson GLM for counts with log(offset_count) as offset; sklearn alpha = lam / sum(offset)."""
    spec = corr_spec(train, xt, xt.E.to_numpy(), corr)
    X = spec.design(train, xt)
    m = PoissonRegressor(alpha=lam / offset_count.sum(), **SOLVER)
    m.fit(X, xt.n.to_numpy() / offset_count, sample_weight=offset_count)
    return spec, m


def apply_correction(spec, m, d, x):
    return np.exp(spec.design(d, x) @ m.coef_ + m.intercept_)


class LineFit:
    """Everything for one line: the developed model, the emerging levels and the optional correction."""

    def __init__(self, line: str, settings: dict):
        self.line, self.s = line, settings

    def fit(self, train: pd.DataFrame, dev: pd.DataFrame, corr=None, stage: str = "corrected", dm=None):
        self.stage = stage
        self.dm = DevModel(self.line, self.s).fit(dev) if dm is None else dm
        self.dm.set_mar(train, dev)
        c = self.dm.components(train)
        x, E = c["x"], c["x"].E.to_numpy()
        self.a_f = self.a_s = self.c_L = 1.0
        self.corr = None
        if stage in ("level", "corrected") and self.s["level"]:
            n, sev = x.n.to_numpy(), x.sev.to_numpy()
            self.a_f = n.sum() / (c["f"] * E).sum()
            cl = n > 0
            self.a_s = (n[cl] * sev[cl] / c["s"][cl]).sum() / n[cl].sum()
        g = np.ones(len(train))
        if stage == "corrected" and corr is not None:
            off = self.a_f * c["f"] * E
            self.corr = (corr, *fit_correction(train, x, off, corr, corr["lambda"]))
            g = apply_correction(self.corr[1], self.corr[2], train, x)
        if stage in ("level", "corrected") and self.s["level"] and self.dm.T is not None:
            n_obs = x.n_large.sum()
            n_exp = (self.a_f * g * c["fl"] * E).sum()
            K = self.s["LARGE_K"]
            self.c_L = (n_obs + K) / (n_exp + K)
            self.large_counts = (float(n_obs), float(n_exp))
        return self

    def predict(self, d: pd.DataFrame, yr) -> np.ndarray:
        c = self.dm.components(d, yr)
        x, E = c["x"], c["x"].E.to_numpy()
        g = np.ones(len(d)) if self.corr is None else apply_correction(self.corr[1], self.corr[2], d, x)
        small = self.a_f * c["f"] * self.a_s * c["s"]
        large = self.a_f * self.c_L * c["L"]
        return E * g * (small + large)


LAST_FIT: dict = {}  # the fitted LineFit objects of the latest call, for reporting levels


def run(train: dict, score: dict, dev: dict, year: int, settings: dict | None = None, stage: str = "corrected",
        keep: dict | None = None, dms: dict | None = None) -> dict:
    s = copy.deepcopy(SETTINGS if settings is None else settings)
    out = {}
    LAST_FIT.clear()
    for ln in LINES:
        if ln not in s["lines"]:
            out[ln] = score[ln].c0.to_numpy(dtype=float)
            continue
        lf = LineFit(ln, s).fit(train[ln], dev[ln], s["corr"].get(ln), stage, (dms or {}).get(ln))
        out[ln] = lf.predict(score[ln], year)
        LAST_FIT[ln] = lf
        if keep is not None:
            keep[ln] = lf
    return out


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    """Frozen configuration: line -> positive predictions aligned to score[line]."""
    out = run(train, score, dev, year, SETTINGS, "corrected")
    for ln, v in out.items():
        assert len(v) == len(score[ln]) and np.all(np.isfinite(v)) and np.all(v > 0), f"{ln}: invalid predictions"
    return out
