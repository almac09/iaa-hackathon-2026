"""A31 variation v2: emerging residual GBM on C0 (scratch, not committed).

Model, per line: y ~ Tweedie(p = 1.5, log link), log mu = log(c0) + log(k_yr) + f(x), where
  - y is the row's total claim cost (y_full) and c0 is C0's expected total cost for the row, exposure included
    (log c0 is proportional to log exposure with slope 1.00); so exposure enters through the offset and every row has
    weight 1. That is the "y with base_margin including log exposure" formulation, and its loss is the leaderboard's
    unweighted Tweedie deviance on row totals.
  - k_yr is the deviance-optimal emerging level (sum y c0^(1-p) / sum c0^(2-p)) of each training (year, new-business)
    cell, fixed in the offset so that the trees fit relativities rather than the emerging A/E or the new-business
    loading.
  - f is a heavily regularised XGBoost Tweedie model of emerging fields, including those C0 cannot use (region,
    occupation cleaned to its 20 jobs, marital status), plus C0's factors in their raw form.
The number of trees is chosen by five-fold customer-grouped cross-validation on the training rows passed in (2023 when
scored on 2024; 2023 + 2024 when predicting 2025); a line where no tree count beats C0 in CV returns C0. The returned
prediction is c0 * exp(f), rescaled so that on the training rows it has C0's total within new business and within
renewals: the variation changes risk ranking within each segment and leaves the overall emerging level, and C0's
new-business / renewal balance, to be set separately.

Excluded on purpose: written_premium (D-05) and gender (D-36), both available only as labelled diagnostics through
`extra`; customer_service_contacts (D-04, leakage); renewal_decision (end-of-period outcome, absent on test rows); the
h_* claim-history counts (near zero in 2023 because emerging history starts in 2023, so their meaning changes by
year); customer_since_year (equals the policy year on every new-business row, so 2025 values lie outside training;
replaced by tenure_years = yr - customer_since_year, which is 0 for new business in every year); identifiers, surname,
dates; fields entirely missing in the emerging market (exact ages, credit band, mileage, telematics). See
profile_fields.py (log: profile_fields.log) for the field suitability checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

P = 1.5
LINES = ("buildings", "contents", "motor")
NTHREAD = 2
SEED = 2031
N_FOLDS = 5
MAX_TREES = 500
STEP = 10

# ---- frozen configuration -------------------------------------------------------------------------------------------
# Hyperparameters ("B"): the best pooled CV gain of 24 configurations compared by customer-grouped CV on 2023 only
# (explore.py, explore.log): depth-2 trees, eta 0.02, column subsampling 0.5, one-vs-rest categorical splits (partition
# splits over 25-300 levels overfit badly), and a very large min_child_weight and L2 penalty (mcw_rows / lambda_rows
# are in row-equivalents of the mean expected hessian at the offset). Uncapped target; the tree count minimises the
# level-matched held-out deviance, as the harness scores it.
# segment=True (chosen AFTER B was scored on A2024; counted): the offset carries the emerging level of each
# (year, new-business) cell and the result is re-anchored to C0 within new business and within renewals, so the trees
# fit relativities within each segment and the variation never moves C0's new-business / renewal balance. Without it,
# B learnt a +55% motor new-business loading (mean relativity 1.55 against 0.96 for renewals) from 2023, where six
# new-business claims above 25k carried 43% of new-business cost (relative A/E 1.81 in 2023, 0.93 in 2024), and lost
# 0.0043 of motor D2 on A2024; the 2025 book is 77% new business.
BASE = {"max_depth": 2, "eta": 0.02, "subsample": 0.7, "colsample_bytree": 0.5, "mcw_rows": 3000, "lambda_rows": 3000,
        "max_cat_to_onehot": 1000, "max_delta_step": 0.0, "cap": None, "target": "cost", "criterion": "level_matched",
        "log_c0": False, "keep": None, "drop": (), "segment": True}
CONFIG = {"buildings": dict(BASE), "contents": dict(BASE), "motor": dict(BASE)}

CUSTOMER = ["channel", "payment_method", "missed_payments_12m", "new_business", "renewal_count",
            "days_quote_to_inception", "discount_applied", "household_size", "prior_claims_5y", "licence_points",
            "years_at_address", "tenure_years", "email_domain", "marketing_consent", "document_preference",
            "previous_insurer", "campaign_code", "branch_id", "marketing_segment", "occ_job", "occ_self_employed",
            "marital_status", "voluntary_excess"]
WEATHER = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm",
           "mean_temperature_c"]
GEO = ["region", "urban_density"] + WEATHER
SITE = ["flood_zone", "distance_to_river_km", "elevation_m", "soil_type"]
PROPERTY = ["year_built", "property_type", "roof_type", "basement", "trees_within_5m", "listed_building", "bedrooms",
            "energy_rating", "solar_panels"]
FEATURES = {
    "buildings": ["policyholder_age_band"] + CUSTOMER + GEO + SITE + PROPERTY
                 + ["sum_insured", "days_unoccupied", "occupancy", "smoke_alarms", "accidental_damage_cover", "mortgage",
                    "construction_type", "x_clay_dry", "x_fz3_rain"],
    "contents": ["policyholder_age_band"] + CUSTOMER + GEO + SITE + PROPERTY
                + ["sum_insured", "high_value_items", "tenure", "security_level", "accidental_damage_cover",
                   "works_from_home", "lodgers", "bicycle_cover"],
    "motor": ["driver_age_band"] + CUSTOMER + GEO
             + ["years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "make_model", "engine_size_cc",
                "body_type", "transmission", "seats", "vehicle_colour", "fuel_type", "driver_assist", "cover_type",
                "overnight_parking", "vehicle_use", "named_young_driver"],
}
TREND = {"buildings": 0.04587525922327018, "contents": 0.041818621051095524, "motor": 0.0362598256137392}


# ---- features -------------------------------------------------------------------------------------------------------
def clean_occupation(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """80 raw values -> 20 jobs: strip whitespace, lower-case, and split off a 'self-employed ' prefix (which the
    emerging insurer applies even to 'retired', 'student' and 'unemployed', and the developed market never uses)."""
    t = s.astype("string").str.strip().str.lower()
    return t.str.replace(r"^self-employed ", "", regex=True), t.str.startswith("self-employed ")


def engineer(df: pd.DataFrame, line: str, cfg: dict | None = None, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Raw frame -> model frame: numerics as float (missing kept as NaN), categoricals as strings."""
    cfg = CONFIG[line] if cfg is None else cfg
    job, self_emp = clean_occupation(df["occupation"])
    cols = {}
    keep = FEATURES[line] if cfg["keep"] is None else [f for f in FEATURES[line] if f in cfg["keep"]]
    for c in [f for f in keep if f not in cfg["drop"]] + list(extra):
        if c == "tenure_years":
            cols[c] = df["yr"] - df["customer_since_year"]
        elif c == "occ_job":
            cols[c] = job
        elif c == "occ_self_employed":
            cols[c] = self_emp.astype("Float64")
        elif c == "year_built" and "year_built_raw" in df:
            # the cleaned field imputes the 1900 placeholder (3% of rows in every split); keep it missing instead
            cols[c] = df["year_built"].where(df["year_built_raw"] == df["year_built"])
        elif c in ("sum_insured", "written_premium"):
            cols[c] = np.log(df[c].astype(float))
        elif c == "branch_id":
            cols[c] = df[c].astype("string")
        else:
            cols[c] = df[c]
    if cfg["log_c0"]:
        cols["log_c0_rate"] = np.log(df["c0"] / df["exposure_plain"]) - TREND[line] * (df["yr"] - 2023)
    out = pd.DataFrame(cols, index=df.index)
    for c in out.columns:
        dt = str(out[c].dtype)
        if dt in ("bool", "boolean"):
            out[c] = out[c].astype("Float64").astype(float)
        elif dt in ("object", "str") or dt.startswith("string"):
            out[c] = out[c].astype("string")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def align(train_x: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Categoricals take the training levels; levels unseen in training become missing (and are counted)."""
    a, b, unseen = train_x.copy(), other.copy(), {}
    for c in a.columns:
        if str(a[c].dtype).startswith("string"):
            cats = sorted(a[c].dropna().unique().tolist())
            known = b[c].isin(cats).fillna(False).astype(bool)
            unseen[c] = int((b[c].notna() & ~known).sum())
            a[c] = pd.Categorical(a[c], categories=cats)
            b[c] = pd.Categorical(b[c].where(known), categories=cats)
    return a, b, unseen


# ---- model ----------------------------------------------------------------------------------------------------------
def level(y, mu) -> float:
    """Deviance-optimal scale of mu for y."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float((y * mu ** (1 - P)).sum() / (mu ** (2 - P)).sum())


def year_levels(y, c0, yr) -> dict:
    return {v: level(y[yr == v], c0[yr == v]) for v in np.unique(yr)}


def nloglik(y, mu) -> float:
    return float(np.mean(-y * mu ** (1 - P) / (1 - P) + mu ** (2 - P) / (2 - P)))


def booster_params(mu_scaled: np.ndarray, cfg: dict) -> dict:
    count = cfg["target"] == "count"
    # expected hessian at the offset (Poisson: mu; Tweedie: mu^(2-p) at E[y] = mu, times (2-p) for the observed form)
    mean_h = float(np.mean(mu_scaled if count else (2 - P) * mu_scaled ** (2 - P)))
    obj = {"objective": "count:poisson", "eval_metric": "poisson-nloglik"} if count else \
        {"objective": "reg:tweedie", "tweedie_variance_power": P, "eval_metric": f"tweedie-nloglik@{P}"}
    return obj | {"base_score": 1.0, "tree_method": "hist", "max_cat_to_onehot": cfg["max_cat_to_onehot"],
                  "nthread": NTHREAD, "seed": SEED, "max_delta_step": cfg["max_delta_step"],
                  "max_depth": cfg["max_depth"], "eta": cfg["eta"], "subsample": cfg["subsample"],
                  "colsample_bytree": cfg["colsample_bytree"], "min_child_weight": cfg["mcw_rows"] * mean_h,
                  "lambda": cfg["lambda_rows"] * mean_h}


def customer_folds(cust: np.ndarray, n: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    u = np.unique(cust)
    f = pd.Series(np.arange(len(u)) % n, index=u[np.random.default_rng(seed).permutation(len(u))])
    return f.reindex(cust).to_numpy()


def segments(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Segment code per row: 0 for every row, or the new-business flag when cfg['segment'] is set."""
    return df["new_business"].astype(int).to_numpy() if cfg["segment"] else np.zeros(len(df), int)


def training_inputs(y, n, c0, yr, seg, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """Label, offset mean and scale for the booster. Cost target: y capped at cfg['cap'], offset c0 times the Tweedie
    level of each (year, segment), both divided by mean(c0) so hessians are of order one. Count target: claim counts,
    offset c0 times the Poisson level of each (year, segment) (c0 as a frequency index), unscaled."""
    key = yr * 10 + seg
    if cfg["target"] == "count":
        lv = {v: n[key == v].sum() / c0[key == v].sum() for v in np.unique(key)}
        return n.astype(float), c0 * np.array([lv[v] for v in key]), 1.0
    lab = np.minimum(y, cfg["cap"]) if cfg["cap"] else y
    lv = year_levels(lab, c0, key)
    return lab, c0 * np.array([lv[v] for v in key]), float(np.mean(c0))


def anchors(c0, rel, seg) -> dict:
    """Per segment, the factor that gives c0 * rel C0's total on the training rows."""
    return {g: float(np.sum(c0[seg == g] * rel[seg == g]) / np.sum(c0[seg == g])) for g in np.unique(seg)}


def _dmatrix(x, lab, mu, s):
    return xgb.DMatrix(x, label=None if lab is None else lab / s, base_margin=np.log(mu / s), enable_categorical=True)


def cv_trees(x: pd.DataFrame, y: np.ndarray, n: np.ndarray, c0: np.ndarray, yr: np.ndarray, seg: np.ndarray,
             cust: np.ndarray, cfg: dict) -> dict:
    """Five-fold customer-grouped CV. In each fold the trees are fitted on the fold's training part (offset levels and
    segment anchors estimated there too); held-out rows are scored at every STEP trees as c0 * exp(f) / anchor. The
    curves are the held-out mean Tweedie nloglik on the uncapped target (the leaderboard's) and on the capped target,
    each at the per-year level re-fitted on the fold's training part, and on the uncapped target level-matched on the
    held-out fold itself (as the harness scores). Returns them and the tree count minimising cfg['criterion']."""
    folds = customer_folds(cust)
    yc = np.minimum(y, cfg["cap"]) if cfg["cap"] else y
    sc = float(np.mean(c0))
    ks = np.arange(0, MAX_TREES + 1, STEP)
    unc, cap, lm = np.zeros(len(ks)), np.zeros(len(ks)), np.zeros(len(ks))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        lab, mu_tr, s = training_inputs(y[tr], n[tr], c0[tr], yr[tr], seg[tr], cfg)
        xa, xb, _ = align(x[tr], x[te])
        bst = xgb.train(booster_params(mu_tr / s, cfg), _dmatrix(xa, lab, mu_tr, s), MAX_TREES)
        z_tr = xgb.DMatrix(xa, base_margin=np.zeros(tr.sum()), enable_categorical=True)
        z_te = xgb.DMatrix(xb, base_margin=np.zeros(te.sum()), enable_categorical=True)
        for i, k in enumerate(ks):
            r_tr = np.exp(bst.predict(z_tr, output_margin=True, iteration_range=(0, k))) if k else np.ones(tr.sum())
            r_te = np.exp(bst.predict(z_te, output_margin=True, iteration_range=(0, k))) if k else np.ones(te.sum())
            anc = anchors(c0[tr], r_tr, seg[tr])
            r_tr = r_tr / np.array([anc[g] for g in seg[tr]])
            r_te = r_te / np.array([anc.get(g, 1.0) for g in seg[te]])
            for target, acc in ((y, unc), (yc, cap)):
                lk = year_levels(target[tr], c0[tr] * r_tr, yr[tr])
                mu = c0[te] * r_te * np.array([lk.get(v, 1.0) for v in yr[te]])
                acc[i] += nloglik(target[te] / sc, mu / sc) * te.sum()
            mu = c0[te] * r_te
            lm[i] += nloglik(y[te] / sc, mu * level(y[te], mu) / sc) * te.sum()
    curves = {"uncapped": unc / len(y), "capped": cap / len(y), "level_matched": lm / len(y)}
    best = int(ks[np.argmin(curves[cfg["criterion"]])])
    return {"n_trees": best, "ks": ks, "scale": sc} | curves


def fit_line(train: pd.DataFrame, line: str, cfg: dict | None = None, extra: tuple[str, ...] = (),
             n_trees: int | None = None) -> dict:
    """Fit the residual model on `train` (needs y_full, n_claims, c0, yr, customer_id, new_business); n_trees None
    means choose it by CV on `train`."""
    cfg = CONFIG[line] if cfg is None else cfg
    x = engineer(train, line, cfg, extra)
    y, n, c0 = (train[c].to_numpy(float) for c in ("y_full", "n_claims", "c0"))
    yr, cust, seg = train["yr"].to_numpy(), train["customer_id"].astype(str).to_numpy(), segments(train, cfg)
    cv = None
    if n_trees is None:
        cv = cv_trees(x, y, n, c0, yr, seg, cust, cfg)
        n_trees = cv["n_trees"]
    model = {"bst": None, "x": x, "n_trees": n_trees, "anchor": {g: 1.0 for g in (0, 1)}, "cfg": cfg, "extra": extra,
             "cv": cv}
    if n_trees == 0:  # CV found no tree count that beats C0: the variation returns C0 for this line
        return model
    lab, mu, s = training_inputs(y, n, c0, yr, seg, cfg)
    xa, _, _ = align(x, x.iloc[:0])
    d = _dmatrix(xa, lab, mu, s)
    model["bst"] = bst = xgb.train(booster_params(mu / s, cfg), d, n_trees)
    model["anchor"] = anchors(c0, np.exp(bst.predict(d, output_margin=True) - np.log(mu / s)), seg)
    return model


def relativity(model: dict, score: pd.DataFrame, line: str) -> tuple[np.ndarray, dict]:
    """exp(f) / anchor for the score rows, and the count of unseen categorical levels per field."""
    xs = engineer(score, line, model["cfg"], model["extra"])
    _, xb, unseen = align(model["x"], xs)
    if model["bst"] is None:
        return np.ones(len(xb)), unseen
    f = model["bst"].predict(xgb.DMatrix(xb, base_margin=np.zeros(len(xb)), enable_categorical=True),
                             output_margin=True)
    anc = np.array([model["anchor"][g] for g in segments(score, model["cfg"])])
    return np.exp(f) / anc, unseen


def predict_line(model: dict, score: pd.DataFrame, line: str) -> np.ndarray:
    r, _ = relativity(model, score, line)
    return score["c0"].to_numpy(float) * r


def predict(train: dict, score: dict, dev: dict, year: int, extra: tuple[str, ...] = ()) -> dict:
    """train/score/dev: line -> DataFrame. Returns line -> positive predictions aligned to the score rows.
    dev is not used: the residual is fitted on the emerging market only, on top of C0. year is not needed beyond the
    c0 that the score rows carry for it."""
    return {ln: predict_line(fit_line(train[ln], ln, extra=extra), score[ln], ln) for ln in LINES}
