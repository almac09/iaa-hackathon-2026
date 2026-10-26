"""v8: emerging-only XGBoost Tweedie benchmark, and its geometric blend with C0.

What a strong typical entry would do without the developed transfer: an XGBoost Tweedie (p = 1.5, log-exposure
offset) fitted on the emerging rows alone with every suitable field and no c0, tuned by customer-grouped
cross-validation within emerging 2023 (explore.py). The submitted form blends it with C0 geometrically,
pred = k * c0^(1 - w) * gbm^w, with w per line chosen on out-of-fold 2023 predictions, and k set so that the line's
total prediction on the score rows equals the total c0 (the 2025 level is set elsewhere; this is a ranking change).

Field handling (evidence in datacheck.py / datacheck.txt):
  - excluded by owner decision: written_premium (D-05), gender (D-36), customer_service_contacts (D-04, leakage).
    written_premium and gender can be switched on only as labelled diagnostics (cfg keys "wp", "gender").
  - excluded as unsuitable: customer_since_year (equals yr - renewal_count exactly, and is 2025 on every 2025 new-
    business row; renewal_count carries the same tenure without the year drift); campaign_code (present exactly on
    new business in 2023, but kept by renewing customers later, so 12% -> 26% -> 83% present and its meaning
    changes); claim-history fields h_* (all zero in 2023 because emerging history starts in 2023, so the 2023 fit
    cannot learn them, and their meaning changes with the length of history); renewal_decision (end-of-term
    outcome); ids, names, dates, src, yr; *_raw fields (harmonised versions used: exposure_raw is in days for part
    of 2023-24 and in years in 2025, sum_insured_raw has unit errors); contents f_* bands, z_log_si and log_si
    (duplicates of raw fields, and f_cs_contacts / f_wp_band encode excluded fields); fields 100% missing in the
    emerging market (credit_band, date_of_birth, exact ages, mileage, telematics); compulsory_excess (constant).
  - weather fields (hail, wind, freezing, dry summer, wildfire, rainfall, temperature and the buildings
    interactions x_clay_dry, x_fz3_rain) are realised values for the policy year: the same address has different
    values each year (PSI 0.2-1.8 between years) and region-year shocks are confounded with region in a one-year
    fit. On the developed market, a GBM fitted on one year at emerging size transfers to the next year better
    without them in all three lines (weather_dev_check.txt), so they are dropped ("weather": False).
  - occupation: 80 raw spellings normalised to 20 jobs (case, trailing spaces, a "Self-employed " prefix that also
    appears on retired / student / unemployed, so it is treated as noise and dropped).
  - missing values (about 12% of several fields, at random): left missing, handled by XGBoost's default direction.
  - categorical levels with fewer than RARE_MIN training rows, and levels unseen in training, go to "__other__"
    (or missing when no level was rare); counts of unseen levels are in datacheck.txt.
  - written_premium (diagnostic only) is an annual premium that drifts about 5-6% a year; it is put on the latest
    training year's terms with the median ratio for customers present in both years (premium only, no outcomes).
  - several training years: the offset carries each year's own claim rate, so the trees learn relativities and
    not the year-to-year level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

P = 1.5
LINES = ("buildings", "contents", "motor")
NTHREAD = 2
RARE_MIN = 30
OTHER = "__other__"

# ---------------------------------------------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------------------------------------------
COMMON_NUM = ["missed_payments_12m", "renewal_count", "days_quote_to_inception", "discount_applied", "household_size",
              "prior_claims_5y", "licence_points", "years_at_address", "voluntary_excess"]
COMMON_CAT = ["channel", "payment_method", "new_business", "job", "marital_status", "marketing_segment", "region",
              "urban_density"]
ADMIN_CAT = ["email_domain", "marketing_consent", "document_preference", "previous_insurer", "branch_id"]
ADDRESS_NUM = ["flood_zone", "distance_to_river_km", "elevation_m", "year_built", "bedrooms", "hail_days",
               "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm",
               "mean_temperature_c"]
ADDRESS_CAT = ["soil_type", "property_type", "roof_type", "basement", "trees_within_5m", "listed_building",
               "energy_rating", "solar_panels"]
LINE_NUM = {
    "buildings": ["sum_insured", "days_unoccupied", "x_clay_dry", "x_fz3_rain"],
    "contents": ["sum_insured", "high_value_items"],
    "motor": ["years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "engine_size_cc", "seats"],
}
LINE_CAT = {
    "buildings": ["policyholder_age_band", "occupancy", "smoke_alarms", "accidental_damage_cover", "mortgage",
                  "construction_type"],
    "contents": ["policyholder_age_band", "tenure", "security_level", "accidental_damage_cover", "works_from_home",
                 "lodgers", "bicycle_cover"],
    "motor": ["driver_age_band", "make_model", "body_type", "transmission", "vehicle_colour", "fuel_type",
              "driver_assist", "cover_type", "overnight_parking", "vehicle_use", "named_young_driver"],
}
WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm",
           "mean_temperature_c", "x_clay_dry", "x_fz3_rain"]  # realised annual values: change each year by address
LARGE_THRESHOLD = {"buildings": 50000.0, "contents": None, "motor": 50000.0}

# ---------------------------------------------------------------------------------------------------------------
# Frozen configuration, fixed before any A2024 look. Hyperparameters: best of 20 random settings per line by
# customer-grouped 5-fold CV within emerging 2023 (explore_search.txt). Weather fields dropped on the developed-market
# year-to-year check (weather_dev_check.txt). Rounds and blend weight w (weight on the GBM in log space) re-chosen on
# the 2023 out-of-fold predictions without weather (oof_noweather.json). Buildings: w = 0, i.e. C0 unchanged.
# ---------------------------------------------------------------------------------------------------------------
FROZEN = {
    "buildings": {"params": {"max_depth": 2, "min_child_weight": 3000, "subsample": 0.7, "colsample_bytree": 0.7,
                             "reg_lambda": 10.0, "eta": 0.02}, "rounds": 200, "cap": False, "features": "all",
                  "weather": False, "wp": False, "gender": False, "w": 0.0},
    "contents": {"params": {"max_depth": 2, "min_child_weight": 1000, "subsample": 0.7, "colsample_bytree": 0.4,
                            "reg_lambda": 50.0, "eta": 0.02}, "rounds": 200, "cap": False, "features": "all",
                 "weather": False, "wp": False, "gender": False, "w": 0.15},
    "motor": {"params": {"max_depth": 2, "min_child_weight": 1000, "subsample": 0.7, "colsample_bytree": 0.7,
                         "reg_lambda": 10.0, "eta": 0.02}, "rounds": 150, "cap": True, "features": "all",
              "weather": False, "wp": False, "gender": False, "w": 0.5},
}


def _as_str(s: pd.Series) -> pd.Series:
    """Object series of python str, missing as None (independent of the pandas string dtype in use)."""
    vals = [None if pd.isna(x) else str(x) for x in s.to_numpy(dtype=object)]
    return pd.Series(vals, index=s.index, dtype=object)


def normalise_job(s: pd.Series) -> pd.Series:
    def norm(x):
        if x is None:
            return None
        x = x.strip().lower()
        return x[len("self-employed "):] if x.startswith("self-employed ") else x
    return pd.Series([norm(x) for x in _as_str(s)], index=s.index, dtype=object)


def feature_lists(line: str, features: str = "all", wp: bool = False, gender: bool = False, weather: bool = True):
    num = COMMON_NUM + ADDRESS_NUM + LINE_NUM[line]
    if not weather:
        num = [c for c in num if c not in WEATHER]
    cat = COMMON_CAT + ADDRESS_CAT + LINE_CAT[line] + (ADMIN_CAT if features == "all" else [])
    if wp:
        num = num + ["log_wp"]
    if gender:
        cat = cat + ["gender"]
    return num, cat


def raw_frame(df: pd.DataFrame, line: str, features: str = "all", wp: bool = False, gender: bool = False,
              weather: bool = True):
    num, cat = feature_lists(line, features, wp, gender, weather)
    X = pd.DataFrame(index=df.index)
    for c in num:
        if c == "log_wp":
            continue
        X[c] = pd.to_numeric(df[c].astype("float64"), errors="coerce")
    for c in cat:
        if c == "job":
            v = normalise_job(df["occupation"])
        else:
            v = _as_str(df[c])
        X[c] = v
    return X, num, cat


class Encoder:
    """Categorical levels from the training rows; rare (< RARE_MIN rows) and unseen levels go to OTHER."""

    def fit(self, X: pd.DataFrame, cat: list[str]):
        self.cat, self.levels, self.has_other = cat, {}, {}
        for c in cat:
            vc = X[c].value_counts(dropna=True)
            keep = sorted(vc.index[vc >= RARE_MIN].tolist())
            self.has_other[c] = bool((vc < RARE_MIN).any())
            self.levels[c] = keep + ([OTHER] if self.has_other[c] else [])
        return self

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        X = X.copy()
        unseen = {}
        for c in self.cat:
            v = X[c]
            known = v.isna() | v.isin(self.levels[c])
            unseen[c] = int((~known).sum())
            v = v.where(known, OTHER if self.has_other[c] else None)
            X[c] = pd.Categorical(v, categories=self.levels[c])
        return X, unseen


def wp_index(train: pd.DataFrame, score: pd.DataFrame) -> dict:
    """Premium drift: for every year present, the factor that puts its written_premium on the latest training
    year's terms, from the median premium ratio of customers present in consecutive years (premium only)."""
    allrows = pd.concat([train[["customer_id", "yr", "written_premium"]], score[["customer_id", "yr", "written_premium"]]])
    yrs = sorted(allrows.yr.unique())
    med = allrows.groupby(["customer_id", "yr"]).written_premium.median().unstack()
    step = {}
    for a, b in zip(yrs[:-1], yrs[1:]):
        both = med[[a, b]].dropna()
        step[b] = float(np.median(both[b] / both[a])) if len(both) >= 50 else 1.0
    ref = int(train.yr.max())
    level = {yrs[0]: 1.0}
    for b in yrs[1:]:
        level[b] = level[yrs[yrs.index(b) - 1]] * step[b]
    return {y: level[ref] / level[y] for y in yrs}


def _add_wp(X: pd.DataFrame, df: pd.DataFrame, idx: dict) -> pd.DataFrame:
    X = X.copy()
    X["log_wp"] = np.log(df.written_premium.to_numpy(float) * df.yr.map(idx).to_numpy(float))
    return X


class GBM:
    def __init__(self, line: str, cfg: dict):
        self.line, self.cfg = line, cfg

    def fit(self, train: pd.DataFrame, score_for_wp: pd.DataFrame | None = None):
        cfg, line = self.cfg, self.line
        X, num, cat = raw_frame(train, line, cfg["features"], cfg.get("wp", False), cfg.get("gender", False),
                                cfg.get("weather", True))
        self.wp_idx = None
        if cfg.get("wp", False):
            self.wp_idx = wp_index(train, score_for_wp if score_for_wp is not None else train.iloc[:0])
            X = _add_wp(X, train, self.wp_idx)
        self.enc = Encoder().fit(X, cat)
        Xe, _ = self.enc.transform(X)
        self.cols = num + cat
        y = train.y_full.to_numpy(float)
        thr = LARGE_THRESHOLD[line]
        if cfg.get("cap", False) and thr is not None:
            y = np.minimum(y, thr)
        e = train.exposure_plain.to_numpy(float)
        # each training year's own rate in the offset, so the trees learn relativities, not the year level
        yr = train.yr.to_numpy()
        self.rate = {int(t): float(y[yr == t].sum() / e[yr == t].sum()) for t in np.unique(yr)}
        self.latest = int(yr.max())
        scale = self.rate[self.latest]
        margin = np.log(e) + np.log(np.array([self.rate[int(t)] for t in yr]) / scale)
        d = xgb.DMatrix(Xe[self.cols], label=y / scale, base_margin=margin, enable_categorical=True, nthread=NTHREAD)
        params = {"objective": "reg:tweedie", "tweedie_variance_power": P, "tree_method": "hist",
                  "nthread": NTHREAD, "seed": 31, "verbosity": 0, **cfg["params"]}
        self.scale = scale
        self.booster = xgb.train(params, d, num_boost_round=int(cfg["rounds"]))
        return self

    def predict(self, score: pd.DataFrame, rounds: int | None = None, return_unseen: bool = False):
        X, _, _ = raw_frame(score, self.line, self.cfg["features"], self.cfg.get("wp", False),
                            self.cfg.get("gender", False), self.cfg.get("weather", True))
        if self.wp_idx is not None:
            X = _add_wp(X, score, self.wp_idx)
        Xe, unseen = self.enc.transform(X)
        e = score.exposure_plain.to_numpy(float)
        d = xgb.DMatrix(Xe[self.cols], base_margin=np.log(e), enable_categorical=True, nthread=NTHREAD)
        r = int(rounds or self.cfg["rounds"])
        mu = self.booster.predict(d, iteration_range=(0, r)) * self.scale  # natural level: latest training year
        return (mu, unseen) if return_unseen else mu


def blend(c0: np.ndarray, g: np.ndarray, w: float) -> np.ndarray:
    """Geometric blend c0^(1-w) g^w, rescaled so that its total equals the total c0 on the same rows."""
    c0, g = np.asarray(c0, float), np.asarray(g, float)
    b = np.exp((1 - w) * np.log(c0) + w * np.log(g))
    return b * (c0.sum() / b.sum())


def predict(train: dict, score: dict, dev: dict, year: int, cfg: dict | None = None) -> dict:
    """line -> positive predictions aligned to score[line]. dev is not used (emerging-only by design)."""
    cfg = cfg or FROZEN
    out = {}
    for line in LINES:
        c = cfg[line]
        c0 = score[line].c0.to_numpy(float)
        if c["w"] == 0:
            out[line] = c0.copy()
            continue
        g = GBM(line, c).fit(train[line], score_for_wp=score[line]).predict(score[line])
        out[line] = blend(c0, g, c["w"])
    return out
