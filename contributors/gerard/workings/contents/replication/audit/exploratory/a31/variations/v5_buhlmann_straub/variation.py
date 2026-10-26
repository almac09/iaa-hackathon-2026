"""A31 variation v5 (scratch): Buhlmann-Straub credibility of emerging relativities on top of C0.

For each line, every selected factor's levels get an emerging actual-to-expected ratio against C0, after C0 has been
rescaled to the emerging level separately for each training year (so the year-to-year level swings, 2023 buildings
A/E 1.19 against 2024 0.83, are not read as factor effects). The Buhlmann-Straub structural parameters are estimated
from the data for each factor: the expected process variance (EPV) from the variation of customers' A/E within a level
(weights = expected cost), the variance of hypothetical means (VHM) from the spread of the level means. k = EPV / VHM,
Z = w / (w + k) with w the level's expected cost, and the relativity is Z * A/E + (1 - Z) * 1, rescaled so the factor is
level-neutral on the training book. Correlated factors are handled by backfitting: each factor is re-estimated against
C0 times the other factors' current relativities until the relativities stop moving. Claims are capped at a quantile of
the line's positive claim costs while estimating A/E (not when scoring). The product of relativities is renormalised so
the adjustment is level-neutral on the training book: the line's level is left to whoever sets the 2025 level.

Data handling (see explore.py for the checks behind it):
- missing values (about 12% of several emerging fields, at random) are their own level, credibility-weighted like any
  other; levels never seen in training get relativity 1;
- numeric factors are banded on the training rows (quantile bands, or the values themselves when there are few, with
  thin tails merged); values outside the training range fall into the end bands;
- occupation's 80 raw spellings are cleaned to about 20 jobs (case, spaces and a 'self-employed ' prefix);
- customer_since_year, renewal_count, customer_service_contacts, written_premium and gender are not used.

For factors that C0 dropped but both markets record (occupation), the developed relativity against C0 can serve as the
complement of credibility instead of 1 (FINAL uses it for motor only; dev needs yr, occupation, y_full, c0_in_sample and
customer_id).

predict(train, score, dev, year) -> {line: positive numpy array aligned to score[line]}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LINES = ("buildings", "contents", "motor")
MIN_SHARE = 0.02      # an ordinal value needs this share of training rows, else it is merged into its neighbour
N_BANDS = 5           # quantile bands for numeric factors with many distinct values
MAX_ORD_LEVELS = 8    # numerics with at most this many distinct values keep their values as levels
MAX_ITER = 30
TOL = 1e-5

# Candidate factors per line: name -> (kind, source column). kind: cat = categorical, num = numeric banded on train.
CANDIDATES = {
    "buildings": {
        "region": ("cat", "region"), "new_business": ("cat", "new_business"), "age_band": ("cat", "policyholder_age_band"),
        "channel": ("cat", "channel"), "payment_method": ("cat", "payment_method"), "occupancy": ("cat", "occupancy"),
        "urban_density": ("cat", "urban_density"), "property_type": ("cat", "property_type"),
        "construction_type": ("cat", "construction_type"), "occupation": ("cat", "occupation_job"),
        "sum_insured": ("num", "sum_insured"), "year_built": ("num", "year_built"),
        "wildfire_index": ("num", "wildfire_index"), "flood_zone": ("num", "flood_zone"),
        "distance_to_river_km": ("num", "distance_to_river_km"), "prior_claims_5y": ("num", "prior_claims_5y"),
        "voluntary_excess": ("num", "voluntary_excess"),
    },
    "contents": {
        "region": ("cat", "region"), "new_business": ("cat", "new_business"), "age_band": ("cat", "policyholder_age_band"),
        "channel": ("cat", "channel"), "payment_method": ("cat", "payment_method"), "tenure": ("cat", "tenure"),
        "urban_density": ("cat", "urban_density"), "property_type": ("cat", "property_type"),
        "security_level": ("cat", "security_level"), "occupation": ("cat", "occupation_job"),
        "sum_insured": ("num", "sum_insured"), "high_value_items": ("num", "high_value_items"),
        "flood_zone": ("num", "flood_zone"), "prior_claims_5y": ("num", "prior_claims_5y"),
        "voluntary_excess": ("num", "voluntary_excess"),
    },
    "motor": {
        "region": ("cat", "region"), "new_business": ("cat", "new_business"), "age_band": ("cat", "driver_age_band"),
        "channel": ("cat", "channel"), "payment_method": ("cat", "payment_method"), "cover_type": ("cat", "cover_type"),
        "vehicle_use": ("cat", "vehicle_use"), "urban_density": ("cat", "urban_density"),
        "property_type": ("cat", "property_type"), "occupation": ("cat", "occupation_job"),
        "vehicle_value_group": ("num", "vehicle_value_group"), "ncb_years": ("num", "ncb_years"),
        "years_licensed": ("num", "years_licensed"), "vehicle_age": ("num", "vehicle_age"),
        "prior_claims_5y": ("num", "prior_claims_5y"), "voluntary_excess": ("num", "voluntary_excess"),
        "engine_size_cc": ("num", "engine_size_cc"),
    },
}
# Labelled diagnostics only (owner decisions D-05 and D-36): never part of FINAL.
DIAGNOSTIC = {"written_premium": ("num", "wp_ratio"), "gender": ("cat", "gender")}

# Frozen configuration, chosen by customer-grouped cross-validation within 2023 only (explore.py: forward selection
# over cap {none, 99%, 95%} x k multiplier {1, 2, 4}; confirm.py and round2.py: fresh folds, a factor kept only if
# it adds more than 1e-4 pooled D2 there). No 2024 outcome was used.
FINAL = {
    # age band: the only buildings factor that survived the fresh folds (region +0.00003, construction type -0.00024).
    "buildings": {"factors": ["age_band"], "cap_q": 0.95, "sub": "customer", "k_mult": 1.0},
    "contents": {"factors": ["occupation"], "cap_q": 0.95, "sub": "customer", "k_mult": 1.0},
    # occupation: developed relativity against C0 as the complement of credibility (dev claims capped at the 99th
    # percentile), the emerging estimate credibility-weighted on top of it (Z = 0 on 2023 alone).
    "motor": {"factors": ["occupation"], "dev_factors": ["occupation"], "dev_cap_q": 0.99, "cap_q": None,
              "sub": "customer", "k_mult": 1.0},
}


def clean_occupation(s: pd.Series) -> pd.Series:
    """80 raw spellings -> about 20 jobs: lower case, collapse spaces, drop a 'self-employed ' prefix."""
    t = s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return t.str.replace(r"^self-employed ", "", regex=True)


def source_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The source columns the factors read, with derived ones added."""
    out = pd.DataFrame(index=df.index)
    for spec in list(CANDIDATES.values()) + [DIAGNOSTIC]:
        for kind, col in spec.values():
            if col in out:
                continue
            if col == "occupation_job":
                out[col] = clean_occupation(df["occupation"])
            elif col == "wp_ratio":
                if "written_premium" in df and "c0" in df:
                    out[col] = df["written_premium"].astype(float) / df["c0"].astype(float)
            elif col in df:
                out[col] = df[col]
    return out


class Bander:
    """Maps one factor's raw values to level labels, with the banding fitted on the training rows only."""

    def __init__(self, kind: str):
        self.kind, self.edges, self.lo, self.hi = kind, None, None, None

    def fit(self, x: pd.Series) -> "Bander":
        if self.kind == "num":
            v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
            u = np.unique(v)
            if len(u) <= MAX_ORD_LEVELS:  # few values: keep them, merging thin tails into their neighbours
                share = np.array([(v == a).mean() for a in u])
                lo = 0
                while lo < len(u) - 1 and share[: lo + 1].sum() < MIN_SHARE:
                    lo += 1
                hi = len(u) - 1
                while hi > lo and share[hi:].sum() < MIN_SHARE:
                    hi -= 1
                self.lo, self.hi = u[lo], u[hi]
            else:
                self.edges = np.unique(np.quantile(v, np.linspace(0, 1, N_BANDS + 1)[1:-1]))
        return self

    def transform(self, x: pd.Series) -> np.ndarray:
        if self.kind == "cat":
            s = x.astype("string")
            return s.fillna("NA").to_numpy(object)
        v = pd.to_numeric(x, errors="coerce").to_numpy(float)
        if self.edges is None:
            lab = np.array([f"v{a:g}" for a in np.clip(v, self.lo, self.hi)], dtype=object)
        else:
            lab = np.array([f"b{i}" for i in np.searchsorted(self.edges, v, side="left")], dtype=object)
        lab[np.isnan(v)] = "NA"
        return lab


def bs_parameters(lev: np.ndarray, sub: np.ndarray, y: np.ndarray, m: np.ndarray):
    """Buhlmann-Straub on A/E = y / m with weights m: risks = levels, observations = sub-units within a level.
    Returns (level means, level weights, EPV, VHM, k)."""
    nl = int(lev.max()) + 1
    key, kinv = np.unique(lev.astype(np.int64) * (int(sub.max()) + 1) + sub, return_inverse=True)
    w_t = np.bincount(kinv, weights=m)
    a_t = np.bincount(kinv, weights=y)
    lev_t = (key // (int(sub.max()) + 1)).astype(int)
    w_l = np.bincount(lev_t, weights=w_t, minlength=nl)
    a_l = np.bincount(lev_t, weights=a_t, minlength=nl)
    n_l = np.bincount(lev_t, minlength=nl)
    xbar_l = np.where(w_l > 0, a_l / np.where(w_l > 0, w_l, 1), 1.0)
    ok = w_t > 0
    epv = float(((a_t[ok] - xbar_l[lev_t[ok]] * w_t[ok]) ** 2 / w_t[ok]).sum() / max((n_l - 1).clip(0).sum(), 1))
    w = w_l.sum()
    xbar = a_l.sum() / w
    nlev = int((w_l > 0).sum())
    vhm = float(((w_l * (xbar_l - xbar) ** 2).sum() - (nlev - 1) * epv) / (w - (w_l ** 2).sum() / w))
    k = epv / vhm if vhm > 0 else np.inf
    return xbar_l, w_l, epv, vhm, k


def _year_normalise(m: np.ndarray, y: np.ndarray, yr: np.ndarray) -> np.ndarray:
    out = m.copy()
    for g in np.unique(yr):
        s = yr == g
        out[s] *= y[s].sum() / m[s].sum()
    return out


def dev_relativities(dev: pd.DataFrame, line: str, name: str, cap_q: float | None = 0.99) -> dict:
    """Developed-market relativity of a categorical factor that C0 leaves out (occupation): Buhlmann-Straub A/E of the
    developed rows against C0 in sample, year-normalised, claims capped at cap_q. Used as an offset on C0, i.e. as the
    complement of credibility for the emerging estimate. Returns {level: relativity, '_info': ...}."""
    kind, col = CANDIDATES[line][name]
    assert kind == "cat"
    src = source_frame(dev)
    lab = Bander(kind).fit(src[col]).transform(src[col])
    codes, u = pd.factorize(lab)
    y = dev["y_full"].to_numpy(float)
    if cap_q is not None:
        y = np.minimum(y, float(np.quantile(y[y > 0], cap_q)))
    m = _year_normalise(dev["c0_in_sample"].to_numpy(float), y, dev["yr"].to_numpy())
    sub = pd.factorize(dev["customer_id"].astype(str))[0]
    xbar_l, w_l, epv, vhm, k = bs_parameters(codes, sub, y, m)
    z = w_l / (w_l + k) if np.isfinite(k) else np.zeros_like(w_l)
    r = z * xbar_l + (1 - z)
    r *= w_l.sum() / (w_l * r).sum()
    out = dict(zip(np.asarray(u, dtype=object), r))
    out["NA"] = 1.0  # missing in emerging: neutral offset
    out["_info"] = {"k": k, "z_min": float(z.min()), "z_max": float(z.max())}
    return out


def _offset(dev_rel: dict, src: pd.DataFrame, line: str) -> np.ndarray:
    off = np.ones(len(src))
    for name, rel in dev_rel.items():
        kind, col = CANDIDATES[line][name]
        lab = Bander(kind).transform(src[col])
        off *= np.array([rel.get(a, 1.0) for a in lab])  # unseen level -> 1
    return off


def fit_line(train: pd.DataFrame, line: str, cfg: dict, extra: dict | None = None, src: pd.DataFrame | None = None,
             dev: pd.DataFrame | None = None, dev_rel: dict | None = None) -> dict:
    """Fit the credibility relativities for one line. cfg: factors, cap_q, sub ('customer' or 'month'), k_mult,
    dev_factors (factors whose developed relativity is applied as an offset on C0 first; needs dev or dev_rel)."""
    specs = {**CANDIDATES[line], **(extra or {})}
    factors = list(cfg["factors"])
    src = source_frame(train) if src is None else src
    if cfg.get("dev_factors"):
        if dev_rel is None:
            dev_rel = {f: dev_relativities(dev, line, f, cfg.get("dev_cap_q", 0.99)) for f in cfg["dev_factors"]}
    else:
        dev_rel = {}
    off = _offset(dev_rel, src, line)
    y = train["y_full"].to_numpy(float)
    if cfg.get("cap_q") is not None:
        cap = float(np.quantile(y[y > 0], cfg["cap_q"]))
        y = np.minimum(y, cap)
    else:
        cap = None
    c0 = train["c0"].to_numpy(float) * off
    yr = train["yr"].to_numpy()
    if cfg.get("sub", "customer") == "month":
        sub = pd.to_datetime(train["period_start"]).dt.to_period("M").astype(str)
        sub = pd.factorize(sub.astype(str) + "_" + pd.Series(yr, index=train.index).astype(str))[0]
    else:
        sub = pd.factorize(train["customer_id"].astype(str))[0]
    banders, codes, labels = {}, {}, {}
    for f in factors:
        kind, col = specs[f]
        b = Bander(kind).fit(src[col])
        lab = b.transform(src[col])
        c, u = pd.factorize(lab)
        banders[f], codes[f], labels[f] = b, c, np.asarray(u, dtype=object)
    rel = {f: np.ones(len(labels[f])) for f in factors}
    info = {}
    for it in range(MAX_ITER):
        change = 0.0
        for f in factors:
            other = np.ones_like(c0)
            for g in factors:
                if g != f:
                    other *= rel[g][codes[g]]
            m = _year_normalise(c0 * other, y, yr)
            xbar_l, w_l, epv, vhm, k = bs_parameters(codes[f], sub, y, m)
            k = k * cfg.get("k_mult", 1.0)
            z = w_l / (w_l + k) if np.isfinite(k) else np.zeros_like(w_l)
            r = z * xbar_l + (1 - z)
            r *= w_l.sum() / (w_l * r).sum()
            change = max(change, float(np.abs(np.log(r) - np.log(rel[f])).max()))
            rel[f] = r
            info[f] = {"k": k, "epv": epv, "vhm": vhm, "z_min": float(z.min()), "z_max": float(z.max()),
                       "levels": dict(zip(labels[f], np.round(r, 4))), "w": dict(zip(labels[f], np.round(w_l, 1)))}
        if change < TOL:
            break
    prod = np.ones_like(c0)
    for f in factors:
        prod *= rel[f][codes[f]]
    c0_plain = train["c0"].to_numpy(float)
    norm = c0_plain.sum() / (c0 * prod).sum()  # the whole adjustment (offset and factors) is level-neutral on train
    return {"line": line, "factors": factors, "specs": specs, "banders": banders, "dev_rel": dev_rel,
            "rel": {f: dict(zip(labels[f], rel[f])) for f in factors}, "norm": norm, "cap": cap,
            "iterations": it + 1, "info": info}


def predict_line(model: dict, score: pd.DataFrame, src: pd.DataFrame | None = None) -> np.ndarray:
    src = source_frame(score) if src is None else src
    out = score["c0"].to_numpy(float) * model["norm"] * _offset(model["dev_rel"], src, model["line"])
    for f in model["factors"]:
        kind, col = model["specs"][f]
        lab = model["banders"][f].transform(src[col])
        r = model["rel"][f]
        out = out * np.array([r.get(a, 1.0) for a in lab])  # unseen level -> 1
    return out


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    out = {}
    for line in LINES:
        cfg = FINAL[line]
        if not cfg["factors"] and not cfg.get("dev_factors"):
            out[line] = score[line]["c0"].to_numpy(float)
            continue
        model = fit_line(train[line], line, cfg, dev=dev.get(line))
        p = predict_line(model, score[line])
        assert np.all(np.isfinite(p)) and np.all(p > 0), f"{line}: invalid predictions"
        out[line] = p
    return out
