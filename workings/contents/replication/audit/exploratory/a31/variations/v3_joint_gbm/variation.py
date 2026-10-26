"""V3: joint developed + emerging GBM (A31 variation, scratch).

One XGBoost Tweedie model per line, fitted on the developed market (2021-24) and the emerging training rows together.

Design
- Target: y_full, capped per row at 50,000 for motor (meta.json's threshold), so the developed 12.55m claim does
  not drive the trees; the part above the cap comes back through the level factor L (below), so it moves the
  level, not the ranking. Buildings is uncapped (the 2023 CV preferred it to the 50,000 cap) and contents has no
  threshold.
- Exposure: exposure_plain enters as a log offset, as in the harness's constant-rate baseline.
- Time trend: an explicit log-linear offset, TREND_SLOPE x (year - 2024), with the developed slopes from meta.json
  (C0's). Trees never see the year. Score rows are put at the prediction year `year`, so 2025 is the 2023-24 level
  carried forward by exp(slope) per year: one year past 2024 for the test book.
- Markets: a market indicator (0 developed, 1 emerging). Fields both markets share are one feature. Developed-only
  fields (exact ages, credit band, mileage, telematics) are NaN on emerging rows (FEATURE_SET "union"), or left out
  ("shared"). Emerging-only levels are their own levels: regions E01-E06 beside R01-R12, and the occupation flag
  "self-employed". Occupation is normalised first (case, trailing spaces, the "Self-employed " prefix), which
  maps the 80 raw emerging values onto the 20 developed jobs. customer_since_year is not used (on new business it
  equals the policy year, so its meaning moves with the year; renewal_count carries tenure exactly). Claim-history
  fields (h_*) are not used (their depth differs by year and market). campaign_code is not used (missing on 88% of
  emerging 2023 rows but 17% of the 2025 book, PSI 2.5: its meaning changes). written_premium (D-05), gender (D-36) and
  customer_service_contacts (D-04) are never features here; the first two can be added only as labelled
  diagnostics through `extra`.
- Missing values stay NaN, so XGBoost learns a default direction. Score-row category levels unseen in training are
  set to NaN (missing) and counted by `suitability.py`.
- Developed rows: every row with a claim, plus a seeded DEV_ZERO_FRAC sample of the claim-free rows, weighted
  1 / DEV_ZERO_FRAC, so the weighted likelihood is unbiased for the full developed market. Emerging rows carry
  weight EMG_WEIGHT[line] relative to a developed row.
- Level: the GBM's prediction is multiplied by L, the Tweedie deviance-optimal scale of 4-fold out-of-fold
  predictions (folds by customer) against the uncapped cost of the emerging training rows at their own years. So
  the level is the emerging training years' level (2023-24 for the 2025 call), carried forward by the trend.
- Output: the joint GBM, or with BLEND[line] = a < 1 the geometric blend c0^(1-a) x gbm^a, rescaled to the GBM's
  mean on the score rows (so the blend changes ranking only). a is chosen out of fold within emerging 2023.

The constants below were frozen after the 2023 cross-validation in explore.py and choose.py (folds grouped by
customer_id). Scored on A2024 it does not beat C0 (pooled level-matched D2 0.11364 against 0.12012); see run_a2024.log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

LINES = ("buildings", "contents", "motor")
P = 1.5
NTHREAD = 2
SEED = 3103
REF_YEAR = 2024

# From cache/meta.json (C0's developed trend), frozen here so the module stands alone.
TREND_SLOPE = {"buildings": 0.04587525922327018, "contents": 0.041818621051095524, "motor": 0.0362598256137392}

# Boosting settings (fixed, not tuned on A2024).
PARAMS = {"max_depth": 3, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}
MIN_CHILD_ROWS = 5000          # min_child_weight = rows x median per-row Tweedie hessian at the base rate
DEV_ZERO_FRAC = 0.2            # share of claim-free developed rows kept (all claim rows kept)

# Frozen after the grouped 4-fold cross-validation within emerging 2023 (explore.py, choose.py): the maximum 2023
# out-of-fold level-matched D2 over feature set {union, shared} x weight {1, 3, 9} x trees {20, 40, ..., 400}
# (buildings also uncapped), then the blend weight a in {0, 0.05, ..., 1} on the same out-of-fold predictions.
FEATURE_SET = {"buildings": "union", "contents": "shared", "motor": "shared"}
CAP = {"buildings": None, "contents": None, "motor": 50000.0}
EMG_WEIGHT = {"buildings": 3.0, "contents": 1.0, "motor": 1.0}
N_TREES = {"buildings": 60, "contents": 60, "motor": 120}
BLEND = {"buildings": 0.85, "contents": 0.55, "motor": 1.0}
LEVEL_FOLDS = 4                # out-of-fold folds (by customer) for the level L

COMMON_NUM = ["voluntary_excess", "missed_payments_12m", "renewal_count", "days_quote_to_inception",
              "discount_applied", "household_size", "prior_claims_5y", "licence_points", "years_at_address",
              "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index",
              "rainfall_mm", "mean_temperature_c"]
COMMON_CAT = ["channel", "payment_method", "new_business", "marital_status", "email_domain", "marketing_consent",
              "document_preference", "previous_insurer", "branch_id", "marketing_segment",
              "urban_density", "region", "occ_job", "occ_self_employed"]
LINE_NUM = {
    "buildings": ["sum_insured", "days_unoccupied", "flood_zone", "distance_to_river_km", "elevation_m", "year_built",
                  "bedrooms", "x_clay_dry", "x_fz3_rain"],
    "contents": ["sum_insured", "high_value_items", "flood_zone", "distance_to_river_km", "elevation_m", "year_built",
                 "bedrooms"],
    "motor": ["years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "engine_size_cc", "seats"],
}
LINE_CAT = {
    "buildings": ["construction_type", "occupancy", "smoke_alarms", "accidental_damage_cover", "mortgage", "soil_type",
                  "property_type", "roof_type", "basement", "trees_within_5m", "listed_building", "energy_rating",
                  "solar_panels", "policyholder_age_band"],
    "contents": ["tenure", "security_level", "accidental_damage_cover", "works_from_home", "lodgers", "bicycle_cover",
                 "soil_type", "property_type", "roof_type", "basement", "trees_within_5m", "listed_building",
                 "energy_rating", "solar_panels", "policyholder_age_band"],
    "motor": ["make_model", "body_type", "transmission", "vehicle_colour", "fuel_type", "driver_assist", "cover_type",
              "overnight_parking", "vehicle_use", "named_young_driver", "driver_age_band"],
}
DEV_ONLY_NUM = {"buildings": ["policyholder_age"], "contents": ["policyholder_age"],
                "motor": ["driver_age", "annual_mileage", "telematics_score"]}
DEV_ONLY_CAT = {"buildings": ["credit_band"], "contents": ["credit_band"], "motor": ["credit_band", "telematics_opt_in"]}
# campaign_code: missing on 88% of emerging 2023 rows but 17% of the 2025 book (PSI 2.5), so its meaning changes.
NEVER = {"customer_service_contacts", "written_premium", "gender", "customer_since_year", "campaign_code"}
DIAGNOSTIC = {"gender", "log_written_premium"}  # owner decisions D-36 and D-05: measured, never adopted here


def feature_lists(line: str, feature_set: str = "union", extra: tuple = ()) -> tuple[list, list]:
    num = COMMON_NUM + LINE_NUM[line] + ["market"]
    cat = COMMON_CAT + LINE_CAT[line]
    if feature_set == "union":
        num, cat = num + DEV_ONLY_NUM[line], cat + DEV_ONLY_CAT[line]
    assert not (set(num) | set(cat)) & NEVER, "an excluded field is in the feature list"
    assert set(extra) <= DIAGNOSTIC, "only the labelled diagnostics may be added"
    for x in extra:  # labelled diagnostics only, never in the frozen configuration
        (num if x == "log_written_premium" else cat).append(x)
    return num, cat


def _occupation(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    t = s.astype("string").str.strip().str.lower()
    self_emp = t.str.startswith("self-employed ")
    job = t.str.replace(r"^self-employed\s+", "", regex=True)
    return job, self_emp.astype("string")


def raw_features(df: pd.DataFrame, line: str, market: int, extra: tuple = ()) -> pd.DataFrame:
    """Unencoded features for one market's rows: numerics as float (NaN kept), categoricals as strings."""
    num, cat = feature_lists(line, "union", extra)
    out = {}
    job, self_emp = _occupation(df["occupation"])
    derived = {"occ_job": job, "occ_self_employed": self_emp if market == 1 else pd.Series(pd.NA, index=df.index,
                                                                                          dtype="string"),
               "market": pd.Series(float(market), index=df.index)}
    if "log_written_premium" in extra:
        derived["log_written_premium"] = np.log(df["written_premium"].astype(float).clip(lower=1.0))
    dev_only = set(DEV_ONLY_NUM[line]) | set(DEV_ONLY_CAT[line])
    for c in num:
        v = derived[c] if c in derived else df[c]
        v = pd.to_numeric(v, errors="coerce").astype(float)
        out[c] = pd.Series(np.nan, index=df.index) if (c in dev_only and market == 1) else v
    for c in cat:
        v = derived[c] if c in derived else df[c]
        v = v.astype("string")
        if c in dev_only and market == 1:
            v = pd.Series(pd.NA, index=df.index, dtype="string")
        out[c] = v
    return pd.DataFrame(out, index=df.index)


def encode(raw: pd.DataFrame, num: list, cat: list, levels: dict) -> pd.DataFrame:
    out = {c: raw[c].to_numpy(dtype=float) for c in num}
    for c in cat:
        s = raw[c]
        out[c] = pd.Categorical(s.where(s.isin(levels[c])).to_numpy(dtype=object), categories=levels[c])
    return pd.DataFrame(out, index=raw.index)


def target(df: pd.DataFrame, cap: float | None) -> np.ndarray:
    y = df["y_full"].to_numpy(dtype=float)
    return y if cap is None else np.minimum(y, cap)


def customer_folds(cust, k: int, seed: int = SEED) -> np.ndarray:
    u = pd.unique(np.asarray(cust).astype(str))
    perm = np.random.default_rng(seed).permutation(len(u))
    f = dict(zip(u[perm], np.arange(len(u)) % k))
    return np.array([f[c] for c in np.asarray(cust).astype(str)])


def dev_sample(dev: pd.DataFrame, frac: float = DEV_ZERO_FRAC, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """All developed claim rows plus a seeded share of claim-free rows; weights make it unbiased for the full market."""
    pos = dev["y_full"].to_numpy(float) > 0
    rng = np.random.default_rng(seed)
    keep = pos | (rng.random(len(dev)) < frac)
    w = np.where(pos, 1.0, 1.0 / frac)[keep]
    return dev.loc[keep], w


class JointModel:
    def __init__(self, line, booster, num, cat, levels, base_rate, level, extra):
        self.line, self.booster, self.num, self.cat, self.levels = line, booster, num, cat, levels
        self.base_rate, self.level, self.extra = base_rate, level, extra

    def margin(self, df: pd.DataFrame, years: np.ndarray) -> np.ndarray:
        return (np.log(df["exposure_plain"].to_numpy(float)) + np.log(self.base_rate)
                + TREND_SLOPE[self.line] * (np.asarray(years, float) - REF_YEAR))

    def predict(self, df: pd.DataFrame, years, n_trees: int | None = None, raw: pd.DataFrame | None = None):
        """Expected uncapped cost (level L applied) for emerging rows, at `years` (scalar or per row)."""
        years = np.broadcast_to(np.asarray(years, float), (len(df),))
        raw = raw_features(df, self.line, 1, self.extra) if raw is None else raw
        dm = xgb.DMatrix(encode(raw, self.num, self.cat, self.levels), base_margin=self.margin(df, years),
                         enable_categorical=True, nthread=NTHREAD)
        return self.level * self.booster.predict(dm, iteration_range=(0, n_trees or 0))


def fit_line(line: str, train: pd.DataFrame, dev_rows: pd.DataFrame, dev_w: np.ndarray, emg_weight: float,
             n_trees: int, feature_set: str = "union", extra: tuple = (), dev_raw: pd.DataFrame | None = None,
             train_raw: pd.DataFrame | None = None, cap="line", level_folds: int = 0) -> JointModel:
    """Fit the joint model on (sampled, weighted) developed rows and emerging training rows.

    level_folds = 0 leaves the level at 1 (enough for ranking). level_folds = K sets it to the deviance-optimal scale
    of K-fold out-of-fold predictions on the emerging training rows (folds grouped by customer): an in-sample level
    is useless here, because the boosted fit is Tweedie-balanced in sample but not out of sample.
    """
    cap = CAP[line] if cap == "line" else cap
    dev_raw = raw_features(dev_rows, line, 0, extra) if dev_raw is None else dev_raw
    train_raw = raw_features(train, line, 1, extra) if train_raw is None else train_raw
    model = _fit(line, train, dev_rows, dev_w, emg_weight, n_trees, feature_set, extra, dev_raw, train_raw, cap)
    if level_folds:
        fold = customer_folds(train["customer_id"], level_folds)
        oof = np.zeros(len(train))
        for k in range(level_folds):
            te = fold == k
            m = _fit(line, train[~te], dev_rows, dev_w, emg_weight, n_trees, feature_set, extra, dev_raw,
                     train_raw[~te], cap)
            oof[te] = m.predict(train[te], train["yr"].to_numpy(float)[te], raw=train_raw[te])
        y = train["y_full"].to_numpy(float)
        model.level = float((y * oof ** (1 - P)).sum() / (oof ** (2 - P)).sum())
    return model


def _fit(line, train, dev_rows, dev_w, emg_weight, n_trees, feature_set, extra, dev_raw, train_raw, cap):
    num, cat = feature_lists(line, feature_set, extra)
    raw = pd.concat([dev_raw[num + cat], train_raw[num + cat]], ignore_index=True)
    levels = {c: sorted(raw[c].dropna().unique().tolist()) or ["__none__"] for c in cat}
    X = encode(raw, num, cat, levels)
    y = np.concatenate([target(dev_rows, cap), target(train, cap)])
    e = np.concatenate([dev_rows["exposure_plain"].to_numpy(float), train["exposure_plain"].to_numpy(float)])
    yrs = np.concatenate([dev_rows["yr"].to_numpy(float), train["yr"].to_numpy(float)])
    w = np.concatenate([dev_w, np.full(len(train), float(emg_weight))])
    trend = TREND_SLOPE[line] * (yrs - REF_YEAR)
    base_rate = float((w * y).sum() / (w * e * np.exp(trend)).sum())
    margin = np.log(e) + np.log(base_rate) + trend
    hess = 0.5 * np.sqrt(base_rate * np.median(e))  # Tweedie p = 1.5 hessian of a claim-free row at the base rate
    params = {"objective": "reg:tweedie", "tweedie_variance_power": P, "tree_method": "hist", "seed": SEED,
              "nthread": NTHREAD, "min_child_weight": MIN_CHILD_ROWS * hess, **PARAMS}
    dm = xgb.DMatrix(X, label=y, weight=w, base_margin=margin, enable_categorical=True, nthread=NTHREAD)
    booster = xgb.train(params, dm, num_boost_round=int(n_trees))
    return JointModel(line, booster, num, cat, levels, base_rate, 1.0, extra)


def blend(gbm: np.ndarray, c0: np.ndarray, a: float) -> np.ndarray:
    """Geometric blend c0^(1-a) gbm^a, rescaled to the GBM's mean (the blend changes ranking, not level)."""
    if a >= 1.0:
        return gbm
    b = np.exp((1 - a) * np.log(c0) + a * np.log(gbm))
    return b * gbm.sum() / b.sum()


def predict_parts(train: dict, score: dict, dev: dict, year: int) -> dict:
    """line -> (joint GBM prediction, final prediction) for the score rows; `predict` returns the second."""
    out = {}
    for line in LINES:
        dev_rows, dev_w = dev_sample(dev[line])
        model = fit_line(line, train[line], dev_rows, dev_w, EMG_WEIGHT[line], N_TREES[line],
                         feature_set=FEATURE_SET[line], cap=CAP[line], level_folds=LEVEL_FOLDS)
        gbm = model.predict(score[line], year)
        pred = blend(gbm, score[line]["c0"].to_numpy(float), BLEND[line])
        assert np.all(np.isfinite(pred)) and np.all(pred > 0), f"{line}: invalid predictions"
        out[line] = (gbm, pred)
    return out


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    """line -> positive predictions aligned to score[line] rows, for policy year `year`."""
    return {line: parts[1] for line, parts in predict_parts(train, score, dev, year).items()}
