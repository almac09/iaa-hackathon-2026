"""A31 variation v9 (scratch): grouped emerging corrections on C0 where data are scarce.

Model: log mu = log c0 + intercept(line, year) + sum of grouped factor effects, fitted by penalised Tweedie (p = 1.5)
maximum likelihood (Fisher scoring) on the emerging training rows. Every effect has a ridge penalty expressed in
pseudo-claims: a group with n claims keeps roughly n / (n + k) of its own deviation from C0.

Grouping (the point of this variation):
- a priori: occupation's 80 raw spellings are cleaned to 20 jobs plus 'missing' (case, trailing spaces and a
  'self-employed ' prefix), with an optional self-employed flag;
- data-driven: within each factor, levels are merged hierarchically on the training rows until every group has at
  least MIN_CLAIMS claims: the group with the fewest claims joins its neighbour with the closest actual-to-expected
  against C0 (neighbours in the factor's natural order for ordinal factors, in A/E order for nominal ones; the
  'missing' level may join any group). The thresholds were chosen by customer-grouped cross-validation within 2023;
- pooled across lines: a term may span several lines with one shared set of groups and coefficients (borrowing
  strength), optionally with penalised line-specific deviations or a line-specific multiplier (kappa_l * gamma_g,
  fitted by alternating passes; the term's reference line, by default its first, keeps kappa = 1).

Unseen levels (in a later training set or the 2025 rows) get no group and so a zero effect: C0 times the line level.
For prediction the line's intercept is the training-exposure-weighted mean of its year intercepts.

predict(train, score, dev, year) -> {line: positive numpy array aligned to score[line]}. dev is used only by the
optional developed-relativity occupation term, which the frozen configuration does not use.

Outcome (recorded after the frozen configuration was scored once on A2024; the configuration was not changed):
within-2023 customer-grouped CV gain +0.0112 (se 0.005); A2024 pooled level-matched gain -0.0193 (95% -0.032 to
-0.008). The 2023 effects selected (buildings region and construction, motor new business) are carried by a few
large 2023 claims that recur across CV folds within the year but not into 2024. Not recommended for adoption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

P = 1.5
LINES = ("buildings", "contents", "motor")
MULT_ITER = 4  # alternating passes for shared terms with line-specific multipliers

# ------------------------------------------------------------------------------------------------------------------
# factor definitions: name -> (labeller(df, line) -> pd.Series of str, ordinal order or None)
# ------------------------------------------------------------------------------------------------------------------
MISSING = "missing"
URBAN_ORDER = ["dense_urban", "urban", "suburban", "rural"]
AGE_ORDER = ["17-24", "25-34", "35-49", "50-64", "65+"]


def _lab(v: pd.Series) -> pd.Series:
    return v.astype(object).where(v.notna(), MISSING).astype(str)


def _num_lab(v: pd.Series, edges=None) -> pd.Series:
    """Ordinal labels for a numeric field: the value itself (as a sortable string) or a band from fixed edges."""
    x = pd.to_numeric(v, errors="coerce").astype(float)
    if edges is not None:
        idx = np.searchsorted(np.asarray(edges, float), x.to_numpy(), side="right")
        out = pd.Series([f"b{int(i):02d}" for i in idx], index=v.index)
    else:
        out = pd.Series([f"{val:012.3f}" for val in x.to_numpy()], index=v.index)
    return out.where(x.notna(), MISSING)


def occupation_job(df: pd.DataFrame) -> pd.Series:
    s = df["occupation"].astype(object).where(df["occupation"].notna(), "").astype(str).str.strip().str.lower()
    s = s.str.replace("self-employed ", "", regex=False)
    return s.where(s != "", MISSING)


def occupation_se(df: pd.DataFrame) -> pd.Series:
    s = df["occupation"].astype(object).where(df["occupation"].notna(), "").astype(str).str.strip().str.lower()
    out = np.where(s == "", MISSING, np.where(s.str.startswith("self-employed"), "self_employed", "employed"))
    return pd.Series(out, index=df.index)


def _age(df, line):
    return _lab(df["driver_age_band" if line == "motor" else "policyholder_age_band"])


def _nb(df):
    return df["new_business"].astype(bool).map({True: "nb", False: "ren"})


FACTORS = {
    "region": (lambda d, ln: _lab(d["region"]), None),
    "occ_raw": (lambda d, ln: _lab(d["occupation"]), None),
    "occ_job": (lambda d, ln: occupation_job(d), None),
    "occ_se": (lambda d, ln: occupation_se(d), None),
    "occ_job_se": (lambda d, ln: occupation_job(d) + "|" + occupation_se(d), None),
    "nb": (lambda d, ln: _nb(d), None),
    "age": (_age, AGE_ORDER),
    "age_nb": (lambda d, ln: _age(d, ln) + "|" + _nb(d), None),
    "channel": (lambda d, ln: _lab(d["channel"]), None),
    "channel_nb": (lambda d, ln: _lab(d["channel"]) + "|" + _nb(d), None),
    "urban": (lambda d, ln: _lab(d["urban_density"]), URBAN_ORDER),
    "marital": (lambda d, ln: _lab(d["marital_status"]), None),
    "property_type": (lambda d, ln: _lab(d["property_type"]), None),
    "construction": (lambda d, ln: _lab(d["construction_type"]), None),
    "roof": (lambda d, ln: _lab(d["roof_type"]), None),
    "soil": (lambda d, ln: _lab(d["soil_type"]), None),
    "energy": (lambda d, ln: _lab(d["energy_rating"]), "sorted"),
    "flood_zone": (lambda d, ln: _num_lab(d["flood_zone"]), "sorted"),
    "bedrooms": (lambda d, ln: _num_lab(d["bedrooms"]), "sorted"),
    "tenure": (lambda d, ln: _lab(d["tenure"]), None),
    "security": (lambda d, ln: _lab(d["security_level"]), None),
    "hvi": (lambda d, ln: _num_lab(d["high_value_items"]), "sorted"),
    "make_model": (lambda d, ln: _lab(d["make_model"]), None),
    "vvg": (lambda d, ln: _num_lab(d["vehicle_value_group"]), "sorted"),
    "ncb": (lambda d, ln: _num_lab(d["ncb_years"]), "sorted"),
    "yrs_lic": (lambda d, ln: _num_lab(d["years_licensed"], [1, 2, 3, 5, 8, 12, 20, 30, 40]), "sorted"),
    "veh_age": (lambda d, ln: _num_lab(d["vehicle_age"]), "sorted"),
    "body": (lambda d, ln: _lab(d["body_type"]), None),
    "fuel": (lambda d, ln: _lab(d["fuel_type"]), None),
    "cover": (lambda d, ln: _lab(d["cover_type"]), None),
    "use": (lambda d, ln: _lab(d["vehicle_use"]), None),
    "parking": (lambda d, ln: _lab(d["overnight_parking"]), None),
    "young_driver": (lambda d, ln: _lab(d["named_young_driver"]), None),
    "payment": (lambda d, ln: _lab(d["payment_method"]), None),
}

# Fields each factor reads (for the data checks).
FACTOR_FIELDS = {
    "region": ["region"], "occ_raw": ["occupation"], "occ_job": ["occupation"], "occ_se": ["occupation"],
    "occ_job_se": ["occupation"], "nb": ["new_business"], "age": ["age_band"],
    "age_nb": ["age_band", "new_business"], "channel": ["channel"], "channel_nb": ["channel", "new_business"],
    "urban": ["urban_density"], "marital": ["marital_status"], "property_type": ["property_type"],
    "construction": ["construction_type"], "roof": ["roof_type"], "soil": ["soil_type"], "energy": ["energy_rating"],
    "flood_zone": ["flood_zone"], "bedrooms": ["bedrooms"], "tenure": ["tenure"], "security": ["security_level"],
    "hvi": ["high_value_items"], "make_model": ["make_model"], "vvg": ["vehicle_value_group"], "ncb": ["ncb_years"],
    "yrs_lic": ["years_licensed"], "veh_age": ["vehicle_age"], "body": ["body_type"], "fuel": ["fuel_type"],
    "cover": ["cover_type"], "use": ["vehicle_use"], "parking": ["overnight_parking"],
    "young_driver": ["named_young_driver"], "payment": ["payment_method"],
}


def labels(df: pd.DataFrame, line: str, factor: str) -> pd.Series:
    return FACTORS[factor][0](df, line).astype(str)


def _order(levels, spec):
    if spec is None:
        return None
    if spec == "sorted":
        return sorted(lv for lv in levels if lv != MISSING)
    return [lv for lv in spec if lv in set(levels)] + sorted(set(levels) - set(spec) - {MISSING})


# ------------------------------------------------------------------------------------------------------------------
# data-driven merging of sparse levels
# ------------------------------------------------------------------------------------------------------------------
def merge_levels(lab: np.ndarray, y: np.ndarray, mu: np.ndarray, nclaims: np.ndarray, min_claims: float,
                 order=None) -> dict:
    """Map each training level to a group id so that every group has >= min_claims claims (if the data allow).

    mu is the expected cost (C0 at the training level). The group with the fewest claims is merged into the
    neighbour with the closest log A/E; A/E is smoothed by one pseudo-claim at A/E = 1 so empty levels are finite.
    Ordinal factors merge only with adjacent levels ('missing' may join any group); nominal factors merge with
    the neighbours in A/E order (equivalent to one-dimensional agglomerative clustering)."""
    df = pd.DataFrame({"l": lab, "y": y, "m": mu, "n": nclaims}).groupby("l", sort=False).sum()
    sev = max(float(df.y.sum()) / max(float(df.n.sum()), 1.0), 1e-9)
    groups = [{"lv": [lv], "y": r.y, "m": r.m, "n": r.n} for lv, r in df.iterrows()]

    def lae(g):
        return np.log((g["y"] + sev) / (g["m"] + sev))

    miss = [g for g in groups if g["lv"] == [MISSING]]
    rest = [g for g in groups if g["lv"] != [MISSING]]
    if order is not None:
        pos = {lv: i for i, lv in enumerate(order)}
        rest.sort(key=lambda g: pos.get(g["lv"][0], 1e9))
    else:
        rest = rest + miss
        miss = []
        rest.sort(key=lae)

    def join(a, b):
        return {"lv": a["lv"] + b["lv"], "y": a["y"] + b["y"], "m": a["m"] + b["m"], "n": a["n"] + b["n"]}

    while len(rest) > 1 and min(g["n"] for g in rest) < min_claims:
        i = int(np.argmin([g["n"] for g in rest]))
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(rest)]
        j = min(cands, key=lambda j: abs(lae(rest[j]) - lae(rest[i])))
        a, b = sorted((i, j))
        rest[a:b + 1] = [join(rest[a], rest[b])]
        if order is None:
            rest.sort(key=lae)
    for g in miss:  # ordinal factors: the missing level stands alone if credible, else joins the closest group
        if g["n"] >= min_claims or not rest:
            rest.append(g)
        else:
            j = min(range(len(rest)), key=lambda j: abs(lae(rest[j]) - lae(g)))
            rest[j] = join(rest[j], g)
    return {lv: gi for gi, g in enumerate(rest) for lv in g["lv"]}


def raw_groups(lab: np.ndarray) -> dict:
    return {lv: i for i, lv in enumerate(pd.unique(lab))}


# ------------------------------------------------------------------------------------------------------------------
# penalised Tweedie GLM with an offset (Fisher scoring)
# ------------------------------------------------------------------------------------------------------------------
def _obj(y, eta, b, pen):
    mu = np.exp(eta)
    return 2 * (-y * mu ** (1 - P) / (1 - P) + mu ** (2 - P) / (2 - P)).sum() + (pen * b * b).sum()


def fit_glm(X: sp.csr_matrix, y: np.ndarray, off: np.ndarray, pen: np.ndarray, b0=None, max_iter=100, tol=1e-7):
    k = X.shape[1]
    b = np.zeros(k) if b0 is None else b0.copy()
    eta = off + X @ b
    f = _obj(y, eta, b, pen)
    for _ in range(max_iter):
        mu = np.exp(eta)
        w = 2 * mu ** (2 - P)
        g = X.T @ (2 * (mu ** (2 - P) - y * mu ** (1 - P))) + 2 * pen * b
        H = (X.T @ sp.diags(w) @ X).toarray() + np.diag(2 * pen + 1e-10)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            bn = b - t * step
            etan = off + X @ bn
            fn = _obj(y, etan, bn, pen)
            if fn <= f + 1e-12 * abs(f) or t < 1e-6:
                break
            t *= 0.5
        b, eta, f = bn, etan, fn
        if np.max(np.abs(t * step)) < tol:
            break
    return b


# ------------------------------------------------------------------------------------------------------------------
# model: build the design from a configuration, fit, predict
# ------------------------------------------------------------------------------------------------------------------
def _dev_occ_rel(dev: dict, line: str) -> dict:
    d = dev[line]
    job = d["occupation"].astype(str).str.strip().str.lower()
    g = pd.DataFrame({"j": job, "y": d["y_full"].to_numpy(float), "c": d["c0_in_sample"].to_numpy(float)})
    g = g.groupby("j").sum()
    return (np.log(g.y / g.c) - np.log(g.y.sum() / g.c.sum())).to_dict()


def fit_predict(train: dict, score: dict, dev: dict, config: list, return_model=False):
    """config: list of terms {factor, lines, group: ('raw',) | ('data', m) | ('slope_dev',), k, dev_k (optional)}.

    Fits one stacked model over all lines named in the config and returns {line: predictions} for every line in
    score (lines without terms return c0)."""
    lines = [ln for ln in LINES if any(ln in t["lines"] for t in config)]
    out = {ln: score[ln]["c0"].to_numpy(float).copy() for ln in score}
    if not lines:
        return (out, None) if return_model else out
    tr = {ln: train[ln] for ln in lines}
    n_off = np.cumsum([0] + [len(tr[ln]) for ln in lines])
    row0 = {ln: n_off[i] for i, ln in enumerate(lines)}
    N = int(n_off[-1])
    y = np.concatenate([tr[ln]["y_full"].to_numpy(float) for ln in lines])
    c0 = np.concatenate([tr[ln]["c0"].to_numpy(float) for ln in lines])
    ncl = np.concatenate([tr[ln]["n_claims"].to_numpy(float) for ln in lines])
    yr = np.concatenate([tr[ln]["yr"].to_numpy(int) for ln in lines])
    expo = np.concatenate([tr[ln]["exposure_plain"].to_numpy(float) for ln in lines])
    line_of = np.concatenate([[ln] * len(tr[ln]) for ln in lines])
    off = np.log(c0)
    # C0 at the training level of each (line, year): the base for A/E and for the information scale
    mu_lvl = c0.copy()
    int_keys = []
    for ln in lines:
        for yv in sorted(set(tr[ln]["yr"].to_numpy(int))):
            m = (line_of == ln) & (yr == yv)
            mu_lvl[m] *= y[m].sum() / c0[m].sum()
            int_keys.append((ln, yv))
    info = mu_lvl ** (2 - P)

    def info_per_claim(mask):
        return info[mask].sum() / max(ncl[mask].sum(), 1.0)

    rows, cols, vals, pen = [], [], [], []
    for j, (ln, yv) in enumerate(int_keys):
        idx = np.where((line_of == ln) & (yr == yv))[0]
        rows.append(idx), cols.append(np.full(len(idx), j)), vals.append(np.ones(len(idx)))
        pen.append(0.0)
    ncol = len(int_keys)
    terms = []
    for t in config:
        tl = [ln for ln in lines if ln in t["lines"]]
        mask = np.isin(line_of, tl)
        idx = np.where(mask)[0]
        if t["group"][0] == "slope_dev":
            rel = {ln: _dev_occ_rel(dev, ln) for ln in tl}
            x = np.concatenate([occupation_job(tr[ln]).map(rel[ln]).fillna(0.0).to_numpy(float) for ln in tl])
            rows.append(idx), cols.append(np.full(len(idx), ncol)), vals.append(x)
            pen.append(t["k"] * info_per_claim(mask))
            terms.append({"t": t, "kind": "slope", "col": ncol, "rel": rel, "lines": tl})
            ncol += 1
            continue
        lab = np.concatenate([labels(tr[ln], ln, t["factor"]).to_numpy() for ln in tl])
        if t["group"][0] == "raw":
            gmap = raw_groups(lab)
        else:
            gmap = merge_levels(lab, y[idx], mu_lvl[idx], ncl[idx], t["group"][1],
                                _order(pd.unique(lab), FACTORS[t["factor"]][1]))
        ng = max(gmap.values()) + 1
        gid = pd.Series(lab).map(gmap).to_numpy(int)
        term = {"t": t, "kind": "group", "col": ncol, "gmap": gmap, "ng": ng, "lines": tl, "dev_col": {},
                "kappa": {ln: 1.0 for ln in tl}}
        if t.get("mult") and len(tl) > 1:  # remember where this term's entries sit, to rescale them by line
            term["vals_pos"], term["entry_line"], term["entry_gid"] = len(vals), line_of[idx], gid
        rows.append(idx), cols.append(ncol + gid), vals.append(np.ones(len(idx)))
        pen += [t["k"] * info_per_claim(mask)] * ng
        ncol += ng
        if t.get("dev_k") is not None and len(tl) > 1:
            for ln in tl:
                lm = line_of[idx] == ln
                rows.append(idx[lm]), cols.append(ncol + gid[lm]), vals.append(np.ones(lm.sum()))
                pen += [t["dev_k"] * info_per_claim(line_of == ln)] * ng
                term["dev_col"][ln] = ncol
                ncol += ng
        terms.append(term)
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, ncol))
    pen = np.asarray(pen, float)
    b0 = np.zeros(ncol)
    for j, (ln, yv) in enumerate(int_keys):
        m = (line_of == ln) & (yr == yv)
        b0[j] = np.log(y[m].sum() / c0[m].sum())
    b = fit_glm(X, y, off, pen, b0=b0)
    mult_terms = [tm for tm in terms if "vals_pos" in tm]
    for _ in range(MULT_ITER if mult_terms else 0):
        # line-specific multipliers on a shared effect: kappa_l * gamma_g, fitted by alternating between the
        # multipliers (one-dimensional Fisher scoring per line, the rest held fixed) and the full penalised fit
        eta = off + X @ b
        for tm in mult_terms:
            gam = b[tm["col"] + tm["entry_gid"]]
            idx = rows[tm["vals_pos"]]
            ref = tm["t"].get("ref", tm["lines"][0])
            for ln in [x for x in tm["lines"] if x != ref]:  # the reference line keeps kappa = 1 (identifies scale)
                m = tm["entry_line"] == ln
                r, sh = idx[m], gam[m]
                base = eta[r] - tm["kappa"][ln] * sh
                kap = tm["kappa"][ln]
                for _ in range(20):
                    mu = np.exp(base + kap * sh)
                    g1 = (2 * sh * (mu ** (2 - P) - y[r] * mu ** (1 - P))).sum()
                    h1 = (2 * sh * sh * mu ** (2 - P)).sum() + 1e-12
                    kap -= g1 / h1
                    if abs(g1 / h1) < 1e-8:
                        break
                tm["kappa"][ln] = float(kap)
                eta[r] = base + kap * sh
            vals[tm["vals_pos"]] = np.array([tm["kappa"][ln] for ln in tm["entry_line"]])
        X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, ncol))
        b = fit_glm(X, y, off, pen, b0=b)
    model = {"b": b, "terms": terms, "int_keys": int_keys}
    for ln in lines:
        if ln not in score:
            continue
        s = score[ln]
        js = [(j, yv) for j, (l2, yv) in enumerate(int_keys) if l2 == ln]
        w = np.array([expo[(line_of == ln) & (yr == yv)].sum() for _, yv in js])
        eta = np.log(s["c0"].to_numpy(float)) + float((b[[j for j, _ in js]] * w).sum() / w.sum())
        for term in terms:
            if ln not in term["lines"]:
                continue
            if term["kind"] == "slope":
                x = occupation_job(s).map(term["rel"][ln]).fillna(0.0).to_numpy(float)
                eta = eta + b[term["col"]] * x
                continue
            g = labels(s, ln, term["t"]["factor"]).map(term["gmap"])
            seen = g.notna().to_numpy()
            gi = g.fillna(0).to_numpy(int)
            eff = np.where(seen, term["kappa"][ln] * b[term["col"] + gi], 0.0)
            if ln in term["dev_col"]:
                eff = eff + np.where(seen, b[term["dev_col"][ln] + gi], 0.0)
            eta = eta + eff
        out[ln] = np.exp(eta)
    return (out, model) if return_model else out


# ------------------------------------------------------------------------------------------------------------------
# frozen configuration (chosen by customer-grouped cross-validation within 2023; see explore.py)
# ------------------------------------------------------------------------------------------------------------------
# Stage 2 (forward stepwise on single-factor candidates, penalties scaled by 0.5) chose these four terms; stage 3
# found no cross-line pooling (shared, shared + deviations, shared x line multiplier) that beat them in CV.
# k is in pseudo-claims; ('data', m) merges levels until every group has at least m claims.
FINAL_CONFIG: list = [
    {"factor": "region", "lines": ("buildings",), "group": ("data", 100), "k": 50.0},
    {"factor": "construction", "lines": ("buildings",), "group": ("data", 50), "k": 15.0},
    {"factor": "nb", "lines": ("motor",), "group": ("raw",), "k": 50.0},
    {"factor": "fuel", "lines": ("motor",), "group": ("data", 200), "k": 500.0},
]


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    preds = fit_predict(train, score, dev, FINAL_CONFIG)
    for ln, p in preds.items():
        p = np.asarray(p, float)
        assert p.shape == (len(score[ln]),) and np.all(np.isfinite(p)) and np.all(p > 0), ln
        preds[ln] = p
    return preds
