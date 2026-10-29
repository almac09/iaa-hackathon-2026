"""Export for the Excel replica of the Week 2 model: C0 with the segment levels of the final file (public 0.15805).

The final file, work/submissions/submission_emerging_level_midpoint_motornb105_20261026.csv, was built in three steps:
  1. C0, the portable developed model, as it is (A30 E4, emerging.py): the as-is file;
  2. each segment (line x new business or renewal) times its halfway scale (A31, audit/exploratory/a31/levels/
     candidates.py): the midpoint file;
  3. motor new business times 1.05 and divided by its halfway scale, so it ends at 1.05 of as is (A31 write4.py).
Both A31 scripts read their input file with pandas' default CSV parser, which returns some values a few units in the
last place away from what was written; the export replays that, so every step reproduces its file bit for bit.

This runner refits C0 exactly as E4 did, stops if the refit is more than rounding away from the as-is file (relative
1e-14) or if the level steps do not reproduce the midpoint and final files bit for bit, and then writes export/excel/
(see export/excel/README.md):
  terms.csv, spline_pieces.csv, input_mappings.csv, scalars.csv, reconciliation.csv  aggregate, committed
  buildings_xgboost.json, buildings_xgboost_features.csv, buildings_xgboost_nodes.csv  the trees, committed
  rows_components.parquet, rows_inputs_<line>.parquet, rows_glm_terms_<line>.parquet,
  reference_office_premium.parquet                                                    row-level, never committed
The row-level files hold competition data (rules section 5b), so they stay out of git (.gitignore). Last, it
re-reads the exported files and re-evaluates every GLM from them alone, as the spreadsheet will, by both the spline
lookups and the cubic pieces, and the buildings trees from their tables, against the model and the final file.

Usage (from the pack root): python python/export_excel.py [--run-label "<label>"]
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.interpolate import BSpline, PPoly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import emerging  # noqa: E402
import glm_lines  # noqa: E402
import models  # noqa: E402
import office_premium  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import xgb_models  # noqa: E402
from gap_tests import settings  # noqa: E402

RUNNER = "export_excel"
OUT = common.PACK_ROOT / "export" / "excel"
EMG = common.PACK_ROOT / "data" / "emerging"
FILES = {"as_is": "submission_emerging_as_is_20261026.csv",
         "midpoint": "submission_emerging_level_midpoint_20261026.csv",
         "final": "submission_emerging_level_midpoint_motornb105_20261026.csv"}
LINES = ("buildings", "contents", "motor")
YEAR = 2025
MOTOR_NB_LEVEL = 1.05  # A31 write4.py: motor new business at 1.05 of as is
GRID_MAX = 20_000  # a spline gets a lookup at its recorded precision up to this many values, else the values seen
SEEN = "the values that occur in the emerging data (a full grid would be too long)"
CONTINUOUS = {"log_si": ("ln(max(sum_insured, 1)), with sum insured after D-02 (emerging.load_market)",
                         ["sum_insured"]),
              "x_clay_dry": ("coalesce((soil_type = 'clay') x dry_summer_index, 0) (sql/11, A28)",
                             ["soil_type", "dry_summer_index"]),
              "x_fz3_rain": ("coalesce((coalesce(flood_zone, 99) >= 3) x rainfall_mm / 1000, 0): a missing flood "
                             "zone counts as 3 or above (sql/11, A28)", ["flood_zone", "rainfall_mm"])}


# ---- splines -----------------------------------------------------------------------------------------------------

def recorded_power(v: np.ndarray, finest: int = -6, coarsest: int = 6) -> int:
    """The recorded precision of a column as a power of ten: the coarsest 10^p every value is a multiple of."""
    for p in range(coarsest, finest - 1, -1):
        s = 10.0 ** p
        if np.allclose(np.round(v / s) * s, v, rtol=0, atol=1e-9 * max(1.0, s)):
            return p
    return finest


def lookup_values(v: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, str]:
    """Every value from the smallest to the largest in `v` at its recorded precision or, if that is too many, the
    values seen (which are competition data, so they go to a row-level file, not terms.csv). Every value in `seen`
    must be on the grid, so the grid never carries a data value it does not already contain."""
    p = recorded_power(v)
    lo, hi = round(v.min() / 10.0 ** p), round(v.max() / 10.0 ** p)
    if hi - lo + 1 > GRID_MAX:
        return np.unique(np.concatenate([v, seen])), SEEN
    steps = np.arange(lo, hi + 1, dtype=float)
    grid = steps / 10.0 ** -p if p < 0 else steps * 10.0 ** p  # divide for decimals, so 3 / 10 gives the double 0.3
    if not np.isin(seen, grid).all():
        raise ValueError("a value in the data is not on its recorded-precision grid")
    return grid, f"every value from {v.min():g} to {v.max():g} in steps of {10.0 ** p:g}"


def spline_function(st, coef: np.ndarray) -> tuple[BSpline, float, float]:
    """The curve sum_j coef_j B_j(x) of a fitted one-feature SplineTransformer as one B-spline, with the range of
    its knots (outside which the transformer extrapolates as a constant)."""
    bs = st.bsplines_[0]
    t, k = bs.t, bs.k
    n = len(t) - k - 1
    x = np.linspace(t[k], t[n], 513)
    full = BSpline.design_matrix(x, t, k).toarray()
    out = st.transform(x.reshape(-1, 1))
    gap = np.array([np.abs(full - out[:, [j]]).max(0) for j in range(out.shape[1])])
    idx = gap.argmin(1)
    if len(set(idx)) != len(idx) or gap.min(1).max() > 1e-12:
        raise ValueError("could not match the transformer's columns to its B-spline basis")
    c = np.zeros(n)
    c[idx] = coef
    return BSpline(t, c, k), float(t[k]), float(t[n])


def spline_pieces(spline: BSpline, lo: float, hi: float) -> list[dict]:
    """The curve as cubic pieces f(x) = a0 + a1 d + a2 d^2 + a3 d^3, d = x - origin, one per knot interval of
    [lo, hi], with the constants below lo and above hi. Bounds say whether each end is included."""
    pp = PPoly.from_spline(spline)
    inner = [i for i in range(len(pp.x) - 1) if pp.x[i + 1] > pp.x[i] and pp.x[i] >= lo and pp.x[i + 1] <= hi]
    rows = [{"lower": -np.inf, "lower_included": False, "upper": lo, "upper_included": False, "origin": lo,
             "a0": float(spline(lo)), "a1": 0.0, "a2": 0.0, "a3": 0.0, "note": "below the first knot: constant"}]
    for i in inner:
        c = pp.c[:, i]  # highest power first
        rows.append({"lower": pp.x[i], "lower_included": True, "upper": pp.x[i + 1], "upper_included": i == inner[-1],
                     "origin": pp.x[i], "a0": c[3], "a1": c[2], "a2": c[1], "a3": c[0], "note": "cubic piece"})
    rows.append({"lower": hi, "lower_included": False, "upper": np.inf, "upper_included": False, "origin": hi,
                 "a0": float(spline(hi)), "a1": 0.0, "a2": 0.0, "a3": 0.0, "note": "above the last knot: constant"})
    return rows


def eval_pieces(p: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """Evaluate exported cubic pieces at x (no missing values), as a spreadsheet would."""
    out = np.full(len(x), np.nan)
    for r in p.itertuples():
        m = ((x > r.lower) | (r.lower_included & (x == r.lower))) & ((x < r.upper) | (r.upper_included & (x == r.upper)))
        d = x[m] - r.origin
        out[m] = r.a0 + r.a1 * d + r.a2 * d ** 2 + r.a3 * d ** 3
    return out


# ---- one GLM: export and re-evaluation ---------------------------------------------------------------------------

def term_row(line, component, term, kind, level="", value=np.nan, lower=np.nan, upper=np.nan, contribution=np.nan,
             note=""):
    return {"line": line, "component": component, "term": term, "kind": kind, "level": level, "value": value,
            "band_lower": lower, "band_upper": upper, "contribution": contribution, "note": note}


def export_glm(name: str, label: str, glm: rgl.GLM, d: pd.DataFrame, values: pd.DataFrame, trend_slope: float,
               scale: float | None, fitted_on: str = "developed") -> dict:
    """Terms, spline pieces, level maps and per-row contributions of one GLM. `d` is the scored rows, `values` the
    emerging rows whose values the spline lookups must cover, `trend_slope` the slope the component's prediction
    uses (its own for a small-claim GLM; the small-claim model's for the large-claim GLM), `scale` the Pareto
    adjustment of a large-claim GLM, `fitted_on` the rows whose exposure decided which levels to keep."""
    spec, coef = glm.spec, dict(zip(glm.names, glm.model.coef_))
    terms = [term_row(name, label, "(intercept)", "intercept", contribution=float(glm.model.intercept_))]
    pieces, maps, checks = [], [], []
    for f in spec.categorical:
        lv = glm.levels[f]
        for i, level in enumerate(lv):
            terms.append(term_row(name, label, f, "categorical", level, contribution=coef.get(f"{f}={level}", 0.0),
                                  note="base level" if i == 0 else ""))
        terms.append(term_row(name, label, f, "categorical", "(any other level)", contribution=0.0,
                              note="a level the fit did not see (including MISSING where no row is listed) scores "
                                   "as the base level"))
        raw = d[f].astype("string").fillna("MISSING")
        for value in sorted(raw.unique()):
            mapped = value if value in spec.keep[f] else "OTHER"
            note = ("kept" if mapped == value else f"under 0.5% of {fitted_on} exposure: merged into OTHER") + \
                   ("" if mapped in lv else "; not a fitted level, so it scores as the base level")
            maps.append({"section": "categorical level", "line": name, "component": label, "term": f,
                         "source_value": value, "mapped_value": mapped, "note": note})
    for f in spec.numeric:  # banded numerics: 'qNN' counts the band edges strictly below the value
        edges, lv = spec.edges[f], glm.levels[f]
        labels = [f"q{i:02d}" for i in range(len(edges) + 1)] + ["MISSING"]
        for i, level in enumerate(labels):
            lower = np.nan if level == "MISSING" else (-np.inf if i == 0 else edges[i - 1])
            upper = np.nan if level == "MISSING" else (np.inf if i == len(edges) else edges[i])
            note = ("x missing" if level == "MISSING" else "lower < x <= upper") + \
                   ("; base level" if level == lv[0] else "") + \
                   ("" if level in lv else "; not seen in fitting, so it scores as the base level")
            terms.append(term_row(name, label, f, "band", level, lower=lower, upper=upper,
                                  contribution=coef.get(f"{f}={level}", 0.0), note=note))
    for c, st in spec.splines.items():
        cols = [n for n in glm.names if re.fullmatch(rf"{re.escape(c)}_s\d+", n)]
        cvec = np.array([coef[n] for n in cols])
        v = values[c].dropna().to_numpy(dtype=float)
        grid, how = lookup_values(v, np.unique(d[c].dropna().to_numpy(dtype=float)))
        contrib = st.transform(grid.reshape(-1, 1)) @ cvec
        for x, cv in zip(grid, contrib):
            terms.append(term_row(name, label, c, "spline value", value=x, contribution=cv, note=how))
        miss = f"{c}_missing" in coef
        terms.append(term_row(name, label, c, "spline missing", "MISSING", contribution=coef.get(f"{c}_missing", 0.0),
                              note="a missing value takes this instead of the curve" if miss else
                              f"no missing values in the {fitted_on} fit, so a missing value contributes 0"))
        sp, lo, hi = spline_function(st, cvec)
        mine = spline_pieces(sp, lo, hi)
        pieces += [{"line": name, "component": label, "term": c, **p} for p in mine]
        x = np.concatenate([np.linspace(lo - (hi - lo), hi + (hi - lo), 4001), grid])
        gap = float(np.abs(eval_pieces(pd.DataFrame(mine), x) - st.transform(x.reshape(-1, 1)) @ cvec).max())
        checks.append({"check": "spline pieces against the fitted curve, max absolute difference (log scale)",
                       "line": name, "component": label, "term": c, "value": gap, "passed": bool(gap < 1e-10)})
    for c in glm.continuous:
        terms.append(term_row(name, label, c, "continuous", "per unit", contribution=coef[c],
                              note=CONTINUOUS.get(c, ("",))[0]))
    centre = glm.centre
    own = "its own year term" if glm.trend else \
        "no year term of its own: the small-claim model's slope, from this GLM's origin"
    terms += [term_row(name, label, "trend", "trend", "coefficient per year", contribution=trend_slope, note=own),
              term_row(name, label, "trend", "trend", "origin (exposure-weighted mean fitting year)", value=centre),
              term_row(name, label, "trend", "trend", f"log contribution in {YEAR}", value=YEAR - centre,
                       contribution=trend_slope * (YEAR - centre),
                       note=f"value = {YEAR} - origin; multiplier {float(np.exp(trend_slope * (YEAR - centre)))!r}")]
    terms.append(term_row(name, label, "ln(exposure)", "offset", contribution=np.nan,
                          note="add ln(exposure in years), exposure = (period end - period start + 1) / 365 (D-01)"))
    if scale is not None:
        terms.append(term_row(name, label, "Pareto adjustment", "scale", value=scale, contribution=np.log(scale),
                              note="large-claim GLM scaled to the generalised Pareto mean excess"))
    # Per-row contributions from the model's own design matrix.
    f = glm._frame(d, YEAR if glm.trend else None)
    X, names = models.design_matrix(f, glm.levels, glm.numeric)
    assert names == glm.names
    parts = X * glm.model.coef_
    key = [n.split("=")[0] if "=" in n else ("trend" if n == "t" else re.sub(r"_(s\d+|missing)$", "", n)
                                             if n not in glm.continuous else n) for n in names]
    rows = pd.DataFrame({"row_id": d.row_id.to_numpy(), f"{label}__intercept": float(glm.model.intercept_)})
    for k in dict.fromkeys(key):
        rows[f"{label}__{k}"] = parts[:, [i for i, x in enumerate(key) if x == k]].sum(axis=1)
    if not glm.trend:
        rows[f"{label}__trend"] = trend_slope * (YEAR - centre)
    rows[f"{label}__ln_exposure"] = np.log(d.exposure.to_numpy())
    if scale is not None:
        rows[f"{label}__ln_pareto_adjustment"] = np.log(scale)
    eta = rows.drop(columns="row_id").sum(axis=1).to_numpy()
    model = glm.predict(d, YEAR if glm.trend else None)
    if scale is not None:
        model = model * scale * np.exp(trend_slope * (YEAR - centre))
    gap = float(np.abs(np.exp(eta) / model - 1).max())
    checks.append({"check": "per-row term contributions sum to the model's log prediction, max relative difference",
                   "line": name, "component": label, "term": "", "value": gap, "passed": bool(gap < 1e-12)})
    rows[f"{label}__total_log"] = eta
    levels = spec.factors(d).add_prefix(f"{label}__").add_suffix("__level")
    return {"terms": terms, "pieces": pieces, "maps": maps, "checks": checks, "rows": rows, "model": model,
            "levels": levels.set_axis(d.row_id.to_numpy())}


def evaluate_glm(t: pd.DataFrame, pieces: pd.DataFrame, maps: pd.DataFrame, x: pd.DataFrame, route: str,
                 seen: pd.DataFrame | None = None) -> np.ndarray:
    """One GLM component's prediction for the rows of `x` from its exported terms alone, as a spreadsheet would:
    levels through the level map, bands through their edges, splines through the lookups (`route` 'lookup', with the
    seen values for an ungridded spline) or the cubic pieces ('pieces'), the trend's 2025 contribution, the offset
    and the Pareto scale. `t`, `pieces` and `maps` are already cut to the one line and component."""
    eta = np.full(len(x), float(t[t.kind == "intercept"].contribution.iloc[0]))
    for term, g in t[t.kind == "categorical"].groupby("term"):
        m = maps[maps.term == term]
        level = x[term].astype("string").fillna("MISSING").map(dict(zip(m.source_value, m.mapped_value)))
        if level.isna().any():
            raise ValueError(f"{term}: values with no entry in the level map")
        coef = dict(zip(g.level, g.contribution))
        eta += level.map(lambda v: coef.get(v, 0.0)).to_numpy(dtype=float)
    for term, g in t[t.kind == "band"].groupby("term"):
        v = x[term].to_numpy(dtype=float)
        add = np.full(len(v), np.nan)
        for r in g.itertuples():
            m = np.isnan(v) if r.level == "MISSING" else (v > r.band_lower) & (v <= r.band_upper)
            add[m] = r.contribution
        if np.isnan(add).any():
            raise ValueError(f"{term}: values in no band")
        eta += add
    for term in t[t.kind == "spline value"].term.unique():
        v = x[term].to_numpy(dtype=float)
        out = np.full(len(v), float(t[(t.kind == "spline missing") & (t.term == term)].contribution.iloc[0]))
        ok = ~np.isnan(v)
        if route == "pieces":
            out[ok] = eval_pieces(pieces[pieces.term == term], v[ok])
        else:
            g = t[(t.kind == "spline value") & (t.term == term) & t.value.notna()]
            if seen is not None:
                g = pd.concat([g, seen[seen.term == term]])
            found = pd.Series(v[ok]).map(dict(zip(g.value, g.contribution)))
            if found.isna().any():
                raise ValueError(f"{term}: {int(found.isna().sum())} values missing from the lookup")
            out[ok] = found.to_numpy()
        eta += out
    for r in t[t.kind == "continuous"].itertuples():
        v = (np.log(np.maximum(x.sum_insured.to_numpy(dtype=float), 1)) if r.term == "log_si"
             else x[r.term].to_numpy(dtype=float))
        eta += r.contribution * v
    eta += t[(t.kind == "trend") & (t.level == f"log contribution in {YEAR}")].contribution.iloc[0]
    eta += t[t.kind == "scale"].contribution.sum()
    return np.exp(eta) * x.exposure.to_numpy(dtype=float)


# ---- the XGBoost trees: export and re-evaluation -----------------------------------------------------------------

def double_cut(t: float) -> float:
    """XGBoost holds inputs and thresholds in single precision and sends x left when float32(x) < t. The double c
    with x < c exactly when that holds, for every double x: the smallest double that rounds to t or above."""
    t32 = np.float32(t)
    mid = (float(np.nextafter(t32, np.float32(-np.inf))) + float(t32)) / 2  # exact in double precision
    return mid if np.float32(mid) >= t32 else float(np.nextafter(mid, np.inf))


def tree_offset(exposure: np.ndarray, base: float, slope: float, centre: float) -> np.ndarray:
    """The XGBoost margin before the trees in YEAR, in xgb_models._dmatrix's order of operations."""
    return np.log(exposure) + np.log(base) + slope * (YEAR - centre)


def export_trees(path: Path, s: rgl.XGBSmall, d: pd.DataFrame, small: np.ndarray) -> dict:
    """Save an XGBoost small-claim model to `path` (booster.save_model) and describe it as tables, from the saved file:
    its features in the model's order, with each categorical's codes, and every node of every tree. Checks that the
    file, reloaded, reproduces `small` (the model's prediction for `d` in YEAR) bit for bit."""
    s.booster.save_model(path)
    m = json.loads(path.read_text())
    learner, gb = m["learner"], m["learner"]["gradient_booster"]["model"]
    names = learner["feature_names"]
    if names != list(s.numeric) + list(s.cats) or len(gb["trees"]) != s.n_trees:
        raise RuntimeError(f"{path.name}: the saved features or tree count differ from the model's")
    enc = gb["cats"]["enc"]  # XGBoost's own copy of the training levels: UTF-8 bytes, with each level's start and end
    for i, f in enumerate(names):
        o, v = enc[i]["offsets"], enc[i]["values"]
        stored = [bytes(v[a:b]).decode() for a, b in zip(o[:-1], o[1:])]
        if stored != s.cats.get(f, []):
            raise RuntimeError(f"{path.name}: the levels XGBoost stored for {f} differ from the model's")
    features = []
    for i, f in enumerate(names):
        if f in s.cats:
            features += [{"position": i, "feature": f, "type": "categorical", "code": c, "level": lv, "note": ""}
                         for c, lv in enumerate(s.cats[f])]
            features.append({"position": i, "feature": f, "type": "categorical", "code": np.nan,
                             "level": "(any other level)", "note": "no code: missing, so each split's default branch"})
        else:
            note = CONTINUOUS[f][0] if f in CONTINUOUS else \
                "after D-02 (divided by 1,000 at 10m or more)" if f == "sum_insured" else ""
            features.append({"position": i, "feature": f, "type": "numeric", "code": np.nan, "level": "",
                             "note": note})
    nodes = []
    for k, t in enumerate(gb["trees"]):
        cat = {n: t["categories"][a:a + z] for n, a, z in
               zip(t["categories_nodes"], t["categories_segments"], t["categories_sizes"])}
        for n, (left, right) in enumerate(zip(t["left_children"], t["right_children"])):
            row = {"tree": k, "node": n, "kind": "leaf", "position": np.nan, "feature": "", "threshold_float32": np.nan,
                   "threshold": np.nan, "right_codes": "", "right_levels": "", "missing_goes": "", "left": np.nan,
                   "right": np.nan, "leaf_value": np.nan}
            value = float(np.float32(t["split_conditions"][n]))  # exactly as XGBoost holds it (single precision)
            if left == -1:
                row["leaf_value"] = value
            else:
                if min(left, right) <= n:  # score_trees walks each tree once in node order
                    raise RuntimeError(f"{path.name}: tree {k} lists a child before its parent")
                f = names[t["split_indices"][n]]
                row.update(position=t["split_indices"][n], feature=f, left=left, right=right,
                           missing_goes="left" if t["default_left"][n] else "right")
                if t["split_type"][n] == 1:
                    row.update(kind="category split", right_codes=";".join(map(str, cat[n])),
                               right_levels=";".join(s.cats[f][c] for c in cat[n]))
                else:
                    row.update(kind="split", threshold_float32=value, threshold=double_cut(value))
            nodes.append(row)
    expo = d.exposure.to_numpy()
    reloaded = xgb_models.predict(xgb.Booster(model_file=str(path)), xgb_models.matrix(d, s.numeric, s.cats), expo,
                                  s.base, extra=s.slope * (YEAR - s.centre))
    bad = int((reloaded != small).sum())
    if bad:
        raise RuntimeError(f"{path.name}: the reloaded model differs from the fitted one on {bad} rows")
    return {"features": pd.DataFrame(features).astype({"code": "Int64"}),
            "nodes": pd.DataFrame(nodes).astype({"position": "Int64", "left": "Int64", "right": "Int64"}),
            "base_score": float(learner["learner_model_param"]["base_score"].strip("[]"))}


def read_trees(nodes_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The exported tree tables, with every number exactly as written and the levels as text."""
    nodes = pd.read_csv(nodes_path, float_precision="round_trip", keep_default_na=False, na_values=[""],
                        dtype={"right_codes": str, "right_levels": str, "feature": str, "missing_goes": str})
    features = pd.read_csv(features_path, float_precision="round_trip", keep_default_na=False, na_values=[""],
                           dtype={"level": str, "feature": str})
    return nodes, features


def score_trees(nodes: pd.DataFrame, features: pd.DataFrame, x: pd.DataFrame, offset: np.ndarray,
                single: bool) -> np.ndarray:
    """Expected cost from the exported tree tables alone: each row starts at its offset, adds one leaf per tree in
    tree order and takes exp. `single` does both in single precision, as XGBoost does, and reproduces it bit for bit
    where numpy's float32 exp matches XGBoost's expf (checked on macOS, where both call the system's); otherwise
    double precision, as a spreadsheet would. A missing value, or a level with no code, takes each split's default
    branch."""
    cols = {}
    for f, g in features.groupby("position"):
        name = g.feature.iloc[0]
        if g.type.iloc[0] == "categorical":
            codes = g[g.code.notna()]
            level = x[name].astype("string").fillna("MISSING")  # a missing level is the text MISSING (D-06)
            cols[f] = level.map(dict(zip(codes["level"], codes.code))).astype(float).to_numpy()
        else:
            cols[f] = x[name].astype(float).to_numpy()
    X = np.column_stack([cols[f] for f in sorted(cols)])
    total = offset.astype(np.float32) if single else offset.astype(float)
    for _, t in nodes.sort_values(["tree", "node"]).groupby("tree"):
        t = t.reset_index(drop=True)  # nodes 0..n-1 in order
        node = np.zeros(len(X), dtype=int)
        for n in range(len(t)):  # children follow their parent, so one pass in node order reaches every leaf
            r = t.iloc[n]
            at = np.where(node == n)[0]
            if r.kind == "leaf" or not len(at):
                continue
            v = X[at, int(r.position)]
            if r.kind == "category split":
                goes_left = ~np.isin(v, [float(c) for c in r.right_codes.split(";")])
            else:
                goes_left = v < r.threshold
            goes_left = np.where(np.isnan(v), r.missing_goes == "left", goes_left)
            node[at] = np.where(goes_left, int(r.left), int(r.right))
        leaf = t.leaf_value.to_numpy()[node]
        total = total + (leaf.astype(np.float32) if single else leaf)
    return np.exp(total).astype(float)


# ---- the run -----------------------------------------------------------------------------------------------------

def exact_scale(a: np.ndarray, b: np.ndarray, candidates: list[float]) -> tuple[float, str, int]:
    """The multiplier m with a * m == b on every row, bit for bit: a candidate, or a ratio b / a, or one within four
    units in the last place of either. Returns m, where it came from, and how many of the nine nearest doubles
    reproduce b."""
    ratios, counts = np.unique(b / a, return_counts=True)
    tries = [(c, "computed as A31 candidates.py did") for c in candidates] + \
            [(r, "recovered from the as-is and midpoint files") for r in ratios[np.argsort(-counts)][:50]]
    for c, how in tries:
        near, down, up = [np.float64(c)], np.float64(c), np.float64(c)
        for _ in range(4):
            down, up = np.nextafter(down, -np.inf), np.nextafter(up, np.inf)
            near += [down, up]
        hits = [m for m in near if np.array_equal(a * m, b)]
        if hits:
            m = c if c in hits else hits[0]
            return float(m), how, len(hits)
    raise ValueError("no single multiplier reproduces the midpoint file for this segment")


def frame_inputs(name: str, ctx: emerging.Context, d: pd.DataFrame, glms: dict, levels: list[pd.DataFrame]) -> pd.DataFrame:
    """The scored rows as the model saw them: identifiers, period and exposure, every feature any component uses,
    the sources of its continuous terms and the raw values of fixed features, then the band or level each GLM
    assigned."""
    cols = ["row_id", "policy_id", "yr", "period_start", "period_end", "exposure", "exposure_plain", "new_business"]
    for g in glms.values():
        cols += list(g.spec.categorical) + list(g.spec.numeric) + list(g.spec.splines) + list(g.continuous)
        cols += [s for c in g.continuous for s in CONTINUOUS.get(c, ("", []))[1]]
    if isinstance(ctx.models[name].small, rgl.XGBSmall):
        lm = ctx.pcfg["lines"]["models"][name]
        cols += list(lm["numeric"]) + list(lm["categorical"])
    cols += [f"{c}_raw" for c in list(cols)]
    cols = [c for c in dict.fromkeys(cols) if c in d.columns]
    out = d[cols].reset_index(drop=True)
    if "period_start" not in out:  # contents_model_dev keeps no dates: take them from the test file
        dates = pd.read_parquet(EMG / f"test_{name}.parquet", columns=["row_id", "period_start", "period_end"])
        out = out.merge(dates, on="row_id", how="left")
        out[["period_start", "period_end"]] = out[["period_start", "period_end"]].astype("datetime64[us]")
    for lv in levels:
        out = out.merge(lv.rename_axis("row_id").reset_index(), on="row_id", how="left")
    return out


def run(cfg: dict, rec: dict) -> None:
    # Two readings of each file: `subs` holds the values exactly as written; `read` holds them as the A31 scripts read
    # them, with pandas' default CSV parser, which returns some values a few units in the last place away.
    subs = {k: pd.read_csv(emerging.SUBMISSIONS / f, float_precision="round_trip") for k, f in FILES.items()}
    read = {k: pd.read_csv(emerging.SUBMISSIONS / f) for k, f in FILES.items()}
    rec["submissions_sha256"] = {f: common.sha256_file(emerging.SUBMISSIONS / f) for f in FILES.values()}
    order = subs["final"].row_id
    if not all(s.row_id.equals(order) for s in subs.values()):
        raise ValueError("the three submission files list the rows in different orders")
    if pd.read_csv(emerging.OUT / "ev_w5_missing_values.csv").adopt.any():
        raise RuntimeError("E5 filled missing values: this export assumes C0 without the fill (D-35)")
    ctx = emerging.Context(cfg, rec)
    ref = set(pd.read_parquet(EMG / "reference_policies.parquet").row_id)
    k_level = pd.read_csv(emerging.OUT / "ev_w4_level_2025.csv").set_index("line").level_k  # as candidates.py read it
    OUT.mkdir(parents=True, exist_ok=True)
    terms, pieces, maps, checks, comps, scalars = [], [], [], [], [], []
    for name in LINES:
        line, model = ctx.line(name), ctx.models[name]
        test = line.test.reset_index(drop=True)
        if model.cann is not None:
            raise RuntimeError(f"{name}: C0 has a CANN correction, which this export does not cover")
        e = test.exposure.to_numpy()
        if not np.array_equal(e, test.exposure_plain.to_numpy()):
            raise RuntimeError(f"{name}: emerging rows are not priced on plain exposure")
        small = model.small.predict(test, YEAR)
        c = pd.DataFrame({"row_id": test.row_id.to_numpy(), "line": name,
                          "new_business": test.new_business.astype(bool).to_numpy(), "exposure": e})
        c["segment"] = name + np.where(c.new_business, "_nb", "_ren")
        c["reference_policy"] = c.row_id.isin(ref)
        c["small_claim"] = small
        c["small_trend_multiplier"] = np.exp(model.small.slope * (YEAR - model.small.centre))
        glms, adj = {}, None
        if model.excess is not None:
            trend = np.exp(model.small.slope * (YEAR - model.excess.centre))
            adj = model.tail["adjustment"]
            large_glm = model.excess.predict(test)
            large = large_glm * adj * trend  # LineModel.predict's order of operations
            c["large_claim_glm"], c["pareto_adjustment"], c["large_trend_multiplier"] = large_glm, adj, trend
            c["large_claim"] = large
            as_is = small + large
            glms["large"] = model.excess
        else:
            c["large_claim_glm"], c["pareto_adjustment"], c["large_trend_multiplier"] = np.nan, np.nan, np.nan
            c["large_claim"] = 0.0
            as_is = small
        if isinstance(model.small, rgl.GLM):
            glms = {"small": model.small, **glms}
        if not np.array_equal(as_is, ctx.base(name, test, YEAR)):
            raise RuntimeError(f"{name}: the components do not add to C0's prediction")
        file_as_is = subs["as_is"].set_index("row_id").prediction.reindex(c.row_id).to_numpy()
        n_diff = int((as_is != file_as_is).sum())
        ulps = float((np.abs(as_is - file_as_is) / np.spacing(file_as_is)).max())
        rel = float(np.abs(as_is / file_as_is - 1).max())
        note = "the refit matches the as-is file bit for bit" if n_diff == 0 else \
            "the refit differs from the as-is file in the last bits only"
        checks += [{"check": "refit against the as-is file: rows that differ in any bit", "line": name,
                    "component": "", "term": "", "value": n_diff, "passed": n_diff == 0, "note": note},
                   {"check": "refit against the as-is file: max difference in units in the last place", "line": name,
                    "component": "", "term": "", "value": ulps, "passed": ulps == 0, "note": note},
                   {"check": "refit against the as-is file: max relative difference", "line": name, "component": "",
                    "term": "", "value": rel, "passed": rel == 0, "note": note}]
        if rel >= 1e-14:
            raise RuntimeError(f"{name}: the refit differs from the as-is file by more than rounding ({rel:.2e})")
        c["as_is"], c["as_is_file"] = as_is, file_as_is
        hist = line.expected["final"].reindex(c.row_id.to_numpy())
        for g in rgl.GROUPS[name]:
            n_col, e_col = glm_lines.HISTORY_GROUPS[g]
            c[f"history_n_{g}"] = test[n_col].to_numpy(dtype=float)
            c[f"history_e_{g}"] = hist[e_col].to_numpy()
        c["history_factor"] = 1.0
        levels, row_terms = [], []
        for label, g in glms.items():
            r = export_glm(name, label, g, test, line.df, model.small.slope, adj if label == "large" else None)
            terms += r["terms"]
            pieces += r["pieces"]
            maps += r["maps"]
            checks += r["checks"]
            levels.append(r["levels"])
            row_terms.append(r["rows"])
        if row_terms:
            rt = row_terms[0]
            for extra in row_terms[1:]:
                rt = rt.merge(extra, on="row_id")
            rt.to_parquet(OUT / f"rows_glm_terms_{name}.parquet", index=False)
        frame_inputs(name, ctx, test, glms, levels).to_parquet(OUT / f"rows_inputs_{name}.parquet", index=False)
        comps.append(c)
        scalars += line_scalars(name, model)
        tree_files = [OUT / f"{name}_xgboost.json", OUT / f"{name}_xgboost_features.csv",
                      OUT / f"{name}_xgboost_nodes.csv"]
        if isinstance(model.small, rgl.XGBSmall):
            r = export_trees(tree_files[0], model.small, test, small)
            r["features"].to_csv(tree_files[1], index=False)
            r["nodes"].to_csv(tree_files[2], index=False)
            checks.append({"check": f"XGBoost reloaded from {tree_files[0].name} against the model, rows that differ "
                           "in any bit", "line": name, "component": "small", "term": "", "value": 0, "passed": True})
            scalars.append({"line": name, "segment": "", "item": "XGBoost base_score stored in the model file",
                            "value": r["base_score"], "applied": False,
                            "note": "not used: XGBoost ignores it when the offset is given as base_margin"})
        else:
            for p in tree_files:
                p.unlink(missing_ok=True)
    comp = pd.concat(comps, ignore_index=True).set_index("row_id").reindex(order).reset_index()
    # The level steps as A31 applied them: candidates.py multiplied the as-is file as it read it; write4.py multiplied
    # (or copied) the midpoint file as it read it. Both reproduce the files exactly as written, bit for bit.
    a, mid_f, fin_f = read["as_is"].prediction.to_numpy(), subs["midpoint"].prediction.to_numpy(), \
        subs["final"].prediction.to_numpy()
    mid_read = read["midpoint"].prediction.to_numpy()
    seg = comp.segment.to_numpy()
    step1 = np.empty(len(comp))
    for s in sorted(set(seg)):
        m = seg == s
        ln = s.rsplit("_", 1)[0]
        cands = [float(np.sqrt(k_level[ln]))] if ln in ("buildings", "contents") else []
        scale, how, n_hits = exact_scale(a[m], mid_f[m], cands)
        step1[m] = scale
        source = ("square root of the line's 2023-24 level as written (rounded to 6 dp) in ev_w4_level_2025.csv"
                  if cands else "square root of A31's re-solved motor level with the emerging new-business loading "
                  "x1.296 (candidates.py with_loading; its cache was deleted)")
        scalars.append({"line": ln, "segment": s, "item": "level step 1: halfway scale", "value": scale,
                        "applied": True, "note": f"{source}; {how}; {'exactly one' if n_hits == 1 else n_hits} of the "
                        "nine nearest doubles reproduces the midpoint file"})
    mid = a * step1
    nb = seg == "motor_nb"
    final = np.where(nb, mid_read * MOTOR_NB_LEVEL / step1, mid_read)  # write4.py: (midpoint x 1.05) / halfway scale
    for label, x, f in (("midpoint", mid, mid_f), ("final", final, fin_f)):
        bad = int((x != f).sum())
        checks.append({"check": f"level steps reproduce the {label} file, rows that differ in any bit", "line": "all",
                       "component": "", "term": "", "value": bad, "passed": bad == 0})
        if bad:
            raise RuntimeError(f"the level steps do not reproduce the {label} file on {bad} rows")
    for label, k in (("as-is", "as_is"), ("midpoint", "midpoint"), ("final", "final")):
        shifted = read[k].prediction.to_numpy() != subs[k].prediction.to_numpy()
        checks.append({"check": f"pandas' default CSV parser against the {label} file as written: rows read "
                       "differently", "line": "all", "component": "", "term": "", "value": int(shifted.sum()),
                       "passed": True, "type": "information",
                       "note": "why the components match the final file to rounding rather than bit for bit"})
    comp["level_step1_scale"] = step1
    comp["level_step2_times"] = np.where(nb, MOTOR_NB_LEVEL, 1.0)
    comp["level_step2_divide"] = np.where(nb, step1, 1.0)
    comp["as_is_file_as_read"] = a
    comp["midpoint_prediction"] = mid
    comp["midpoint_file_as_read"] = mid_read
    comp["final_prediction"] = final
    comp["effective_level"] = final / comp.as_is_file.to_numpy()
    scalars.append({"line": "motor", "segment": "motor_nb", "item": "level step 2: x1.05 then divided by the halfway "
                    "scale", "value": MOTOR_NB_LEVEL, "applied": True,
                    "note": "final = (midpoint as read x 1.05) / step 1, as A31 write4.py computed it; the effective "
                            "level is 1.05 of as is to within rounding"})
    refit = comp.small_claim.add(comp.large_claim).to_numpy()
    recombined = np.where(nb, refit * step1 * MOTOR_NB_LEVEL / step1, refit * step1)
    comp["final_from_components"] = recombined
    gap = float(np.abs(recombined / fin_f - 1).max())
    checks += [{"check": "components recombine to the final file: (small + large) x step 1 (x 1.05 / step 1 for "
                "motor new business), max relative difference", "line": "all", "component": "", "term": "",
                "value": gap, "passed": gap < 1e-14},
               {"check": "components recombine to the final file: rows that differ in any bit", "line": "all",
                "component": "", "term": "", "value": int((recombined != fin_f).sum()), "passed": True,
                "type": "information", "note": "the A31 steps used the default parser's reading of the files"}]
    comp.to_parquet(OUT / "rows_components.parquet", index=False)
    office_premium.price_reference(emerging.SUBMISSIONS / FILES["final"], EMG).to_parquet(
        OUT / "reference_office_premium.parquet", index=False)
    terms = pd.DataFrame(terms)
    seen = terms.note.eq(SEEN)
    stale = OUT / "terms_values_seen.parquet"
    if seen.any():
        terms[seen].to_parquet(stale, index=False)
        terms = pd.concat([terms[~seen], terms[seen].drop_duplicates(["line", "component", "term"]).assign(
            value=np.nan, contribution=np.nan,
            note="lookup in terms_values_seen.parquet (the values that occur: competition data, not committed); "
                 "the exact curve is in spline_pieces.csv")]).sort_index(kind="stable")
    elif stale.exists():
        stale.unlink()
    terms.to_csv(OUT / "terms.csv", index=False)
    pd.DataFrame(pieces).to_csv(OUT / "spline_pieces.csv", index=False)
    pd.DataFrame(maps + mappings(ctx, cfg)).to_csv(OUT / "input_mappings.csv", index=False)
    pd.DataFrame(scalars + not_applied(ctx)).to_csv(OUT / "scalars.csv", index=False)
    checks += evaluate_from_export(fin_f, order)
    checks.append({"check": "sha256 of the final file", "line": "all", "component": "", "term": FILES["final"],
                   "value": rec["submissions_sha256"][FILES["final"]], "passed": True, "type": "information"})
    out = pd.DataFrame(checks)
    out["type"] = out["type"].fillna("check")
    out["passed"] = out.passed.astype(object).where(out.type == "check", None)  # information rows are not checks
    out.to_csv(OUT / "reconciliation.csv", index=False)
    rec["checks_failed"] = out[(out.type == "check") & ~out.passed.astype(bool)].check.tolist()
    surrogate = ("buildings_surrogate", "rows_surrogate")  # surrogate_buildings.py's outputs, not this run's
    rec["outputs_sha256"] = {p.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(p)
                             for p in sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.json"))
                             + sorted(OUT.glob("*.parquet")) if not p.name.startswith(surrogate)}
    print(out[["type", "check", "line", "component", "term", "value", "passed"]].to_string(index=False))


def line_scalars(name: str, model) -> list[dict]:
    s = model.small
    out = []
    add = lambda item, value, note="": out.append(  # noqa: E731
        {"line": name, "segment": "", "item": item, "value": value, "applied": True, "note": note})
    if isinstance(s, rgl.GLM):
        add("small-claim model", "Tweedie GLM (p = 1.5, log link), fitted on developed 2021-24")
        add("small-claim GLM L2 penalty", s.alpha)
    else:
        add("small-claim model", "XGBoost with the year trend in the offset (A15), fitted on developed 2021-24",
            f"the trees: {name}_xgboost.json, and as tables in {name}_xgboost_nodes.csv and "
            f"{name}_xgboost_features.csv; or fit a substitute to rows_components.parquet small_claim")
        add("XGBoost base rate (in the offset)", s.base, "small-claim cost per exposure year in the fitting rows")
        add("XGBoost trees", s.n_trees)
        for k, v in s.params.items():
            add(f"XGBoost {k}", v)
    add("small-claim trend slope (per year, log scale)", s.slope)
    add("small-claim trend origin (year)", s.centre)
    add(f"small-claim trend multiplier {YEAR}", float(np.exp(s.slope * (YEAR - s.centre))))
    if model.excess is not None:
        for k, v in model.tail.items():
            add(f"large claims: {k}", v, "generalised Pareto fit to reported cost above the threshold" if k == "shape"
                else "")
        add("large-claim GLM L2 penalty", model.excess.alpha)
        add(f"large-claim trend multiplier {YEAR}", float(np.exp(s.slope * (YEAR - model.excess.centre))),
            "the small-claim slope from the large-claim GLM's origin, which equals the small-claim origin")
    return out


def not_applied(ctx: emerging.Context) -> list[dict]:
    """What the developed production model applies and C0 does not, with the production value where one is
    recorded, so the replica can leave it out."""
    out = []
    add = lambda line, item, value, note: out.append(  # noqa: E731
        {"line": line, "segment": "", "item": item, "value": value, "applied": False, "note": note})
    summary = ctx.summary.set_index("line")
    for name in LINES:
        for g, k in settings(ctx.dev[name], ctx.summary)["ks"].items():
            add(name, f"claim-history credibility k ({g})", k, "developed production value; C0 uses no claim history "
                "in the emerging market (D-34), so the history factor is 1")
        add(name, "level-uncertainty tilt", float(summary.loc[name, "uncertainty_tilt"]),
            "developed production value (ev_g6_summary.csv); part of run_glm_lines' production path, which C0 "
            "does not call")
        add(name, "developed level factor", float(summary.loc[name, "level_factor"]),
            "developed production value (ev_g6_summary.csv); C0's level is the segment steps instead")
    for item, note in (("claim-history rebalance", "none: production computes it per window when it applies the "
                        "history factor, which C0 does not (D-34); no value is stored"),
                       ("rebalancing constants", "none in C0"),
                       ("E5 missing-value fill", "not adopted (D-35): missing values keep their own level"),
                       ("CANN correction", "not adopted on any line (A20)")):
        add("all", item, np.nan, note)
    w = ctx.dev["contents"].weights["final"]
    for m, v in enumerate(w, start=1):
        add("contents", f"seasonal exposure weight, month {m}", v, "developed 2021-24 weights (D-22): they shaped "
            "the fit, but emerging rows are priced on plain exposure, so the replica uses a weight of 1")
    return out


def mappings(ctx: emerging.Context, cfg: dict) -> list[dict]:
    """How emerging data is recoded before the model sees it, besides the categorical level maps."""
    out = []
    add = lambda section, term, source="", mapped="", note="": out.append(  # noqa: E731
        {"section": section, "line": "", "component": "", "term": term, "source_value": source,
         "mapped_value": mapped, "note": note})
    for table, select in emerging.CONFORM["emerging"].items():
        add("conforming view", table, "", select, "emerging table as the shared SQL reads it (emerging.CONFORM)")
    for code, cat in re.findall(r"WHEN '(\w+)' THEN '(\w+)'", emerging.CONSTRUCTION):
        add("construction code", "construction_type", code, cat,
            "buildings only, applied in the conforming view, so construction_type_raw already holds the category")
    add("age band", "policyholder_age_band, driver_age_band", "emerging band", "used as given",
        "the developed rows were cut the same way: " + emerging.AGE_BAND.format(x="age"))
    for f in sorted(emerging.PORTABLE_DROP):
        add("dropped factor", f, "", "not used", "absent from the emerging market or not carried over (PORTABLE_DROP; "
            "occupation: no evidence it adds value, D-31)")
    v = cfg["sql_variables"]
    add("data fix D-01", "exposure", "period dates", "(period_end - period_start + 1) / "
        f"{v['exposure_basis_days']}", "the raw exposure column is not used")
    for name, key in (("contents", "si_threshold"), ("buildings", "bld_si_threshold")):
        si = pd.read_parquet(EMG / f"test_{name}.parquet", columns=["sum_insured"]).sum_insured
        hit = si[si >= v[key]]
        found = (f"{len(hit):,} of the {len(si):,} 2025 rows qualify, from {hit.min():,.0f} to {hit.max():,.0f}"
                 if len(hit) else f"none of the {len(si):,} 2025 rows qualifies")
        add("data fix D-02", f"sum_insured ({name})", f">= {v[key]}", f"divided by {v['si_divisor']}",
            f"applies to the raw value in the test file: {found}")
    n_r11 = ctx.emg_con.execute("SELECT count(*) FROM addresses WHERE region = 'R11'").fetchone()[0]
    add("data fix D-26", "elevation_m", "region R11", "divided by 3.28084, rounded to 2 dp",
        f"{n_r11} emerging addresses are in R11")
    typical = ctx.emg_con.execute("""SELECT energy_rating, round(avg(year_built)) AS y FROM addresses
                                     WHERE year_built <> 1900 GROUP BY energy_rating ORDER BY 1""").fetchall()
    for rating, y in typical:
        add("data fix D-27", "year_built", f"1900 with energy rating {rating if rating is not None else 'missing'}",
            int(y), "emerging addresses' mean year built for the rating, rounded (sql/03 addresses_clean)")
    sql = (common.PACK_ROOT / "sql" / "08_features.sql").read_text()
    used = {f for m in ctx.pcfg["glm_lines"]["models"].values() if m["small"] == "glm"
            for f in rgl.resolve(ctx.pcfg, m["categorical"]) if f.startswith("f_")}
    expr, acc = {}, []
    for raw in sql.splitlines():
        if raw.strip().startswith("--"):
            continue
        acc.append(raw.strip())
        hit = re.search(r"\bAS (f_\w+)\s*,?\s*(--.*)?$", raw)
        if hit:
            expr[hit.group(1)] = " ".join(acc).rstrip(",")
            acc = []
        elif raw.strip().endswith(",") and " AS " in raw:
            acc = []
    for f in sorted(used):
        add("contents feature", f, "", re.sub(r"\s+", " ", expr.get(f, "?")),
            "sql/08: lvl(x) = coalesce(CAST(x AS VARCHAR), 'MISSING')")
    add("missing values", "categorical and banded terms", "missing", "MISSING",
        "its own level where the fit saw one (D-06); otherwise it scores as the base level")
    add("missing values", "spline terms", "missing", "see terms.csv kind 'spline missing'",
        "the curve is skipped and the missing indicator's coefficient used instead")
    return out


# ---- re-evaluation from the exported files alone -----------------------------------------------------------------

def evaluate_from_export(final_file: np.ndarray, order: pd.Series) -> list[dict]:
    """Recompute every GLM component and the final prediction from the files just written, without the model, by
    both the spline lookups and the cubic pieces."""
    terms = pd.read_csv(OUT / "terms.csv", dtype=str, keep_default_na=False)  # levels stay text, as written
    for col in ("value", "band_lower", "band_upper", "contribution"):
        terms[col] = [float(v) if v else np.nan for v in terms[col]]  # Python's float() parses exactly
    pieces = pd.read_csv(OUT / "spline_pieces.csv", float_precision="round_trip")
    maps = pd.read_csv(OUT / "input_mappings.csv", dtype=str, keep_default_na=False)
    maps = maps[maps.section == "categorical level"]
    seen_path = OUT / "terms_values_seen.parquet"
    seen = pd.read_parquet(seen_path) if seen_path.exists() else None
    comp = pd.read_parquet(OUT / "rows_components.parquet").set_index("row_id")
    checks, trees = [], {}
    sc = pd.read_csv(OUT / "scalars.csv", dtype=str, keep_default_na=False)
    get = lambda name, item: float(sc[(sc.line == name) & (sc["item"] == item)].value.iloc[0])  # noqa: E731
    for name in LINES:
        nodes_path = OUT / f"{name}_xgboost_nodes.csv"
        if not nodes_path.exists():
            continue
        nodes, features = read_trees(nodes_path, OUT / f"{name}_xgboost_features.csv")
        x = pd.read_parquet(OUT / f"rows_inputs_{name}.parquet").set_index("row_id")
        offset = tree_offset(x.exposure.to_numpy(dtype=float), get(name, "XGBoost base rate (in the offset)"),
                             get(name, "small-claim trend slope (per year, log scale)"),
                             get(name, "small-claim trend origin (year)"))
        model = comp.loc[x.index, "small_claim"].to_numpy()
        single, double = (score_trees(nodes, features, x, offset, s) for s in (True, False))
        cat = features[features.code.notna()].groupby("feature")["level"].apply(set)
        no_code = sum(int((~x[f].astype("string").fillna("MISSING").isin(lv)).sum()) for f, lv in cat.items())
        checks += [{"check": "XGBoost scored from the exported tree tables in single precision, as XGBoost adds "
                    "them: rows that differ from the model in any bit", "line": name, "component": "small", "term": "",
                    "value": int((single != model).sum()), "passed": bool((single == model).all())},
                   {"check": "XGBoost scored from the exported tree tables in double precision, as a spreadsheet "
                    "would: max relative difference from the model", "line": name, "component": "small", "term": "",
                    "value": float(np.abs(double / model - 1).max()),
                    "passed": bool(np.abs(double / model - 1).max() < 1e-4),
                    "note": "XGBoost adds the leaves in single precision, and that rounding over 193 additions can "
                            "approach 1e-4; a wrong branch moves a row far more"},
                   {"check": "XGBoost categorical inputs with no code in the features table (scored as missing)",
                    "line": name, "component": "small", "term": "", "value": no_code, "passed": True,
                    "type": "information"}]
        trees[name] = pd.Series(single, index=x.index)
    for route in ("lookup", "pieces"):
        totals = {}
        for name in LINES:
            x = pd.read_parquet(OUT / f"rows_inputs_{name}.parquet").set_index("row_id")
            parts = {}
            for label, t in terms[terms.line == name].groupby("component"):
                cut = lambda df: df[(df.line == name) & (df.component == label)]  # noqa: E731
                pred = evaluate_glm(t, cut(pieces), cut(maps), x, route, None if seen is None else cut(seen))
                col = "small_claim" if label == "small" else "large_claim"
                gap = float(np.abs(pred / comp.loc[x.index, col].to_numpy() - 1).max())
                checks.append({"check": f"GLM re-evaluated from the exported tables ({route}) against the model, max "
                               "relative difference", "line": name, "component": label, "term": "", "value": gap,
                               "passed": bool(gap < 1e-12)})
                parts[label] = pd.Series(pred, index=x.index)
            small = parts["small"] if "small" in parts else trees[name]  # buildings: the exported trees
            totals[name] = small + parts.get("large", 0.0)
        c = comp.loc[order]
        total = pd.concat(totals.values()).reindex(order).to_numpy()
        pred = total * c.level_step1_scale.to_numpy() * c.level_step2_times.to_numpy() / c.level_step2_divide.to_numpy()
        gap = float(np.abs(pred / final_file - 1).max())
        checks.append({"check": f"final prediction rebuilt from the exported tables ({route}; buildings small claims "
                       "from the tree tables in single precision), max relative difference", "line": "all",
                       "component": "",
                       "term": "", "value": gap, "passed": bool(gap < 1e-12)})
    return checks


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    try:
        run(cfg, rec)
        rec["status"] = "success" if not rec.get("checks_failed") else "checks failed"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost", "scipy")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
