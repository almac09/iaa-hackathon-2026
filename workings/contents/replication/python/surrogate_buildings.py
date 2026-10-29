"""Surrogate GLM for the buildings small-claim XGBoost (audit A35): a GLM fitted to the trees' own predictions for
the 2025 buildings rows, so those small claims can be priced from rating tables instead of 193 trees.

It approximates the trees for convenience and is not part of any submission: the trees remain the model, and
buildings_xgboost_nodes.csv reproduces them exactly. The GLM takes the features the trees split on:
  - a numeric with at most 20 values in 2025 is a factor with one band per value; any other numeric is a cubic
    spline with knots at quantiles from the 1st to the 99th percentile and flat beyond them, so the few rows at
    either end cannot bend the curve (the trees still split beyond them for a few climate features: see A35). A
    numeric whose quantiles give fewer than three distinct knots stays banded by FactorSpec's deciles, as
    x_clay_dry and x_fz3_rain do in the smaller candidates;
  - each categorical as levels;
  - the ten strongest interactions in the trees: pairs of features split one after the other, ranked by the gain of
    the second split. Each is the cross of two banded factors, each feature cut at its four highest-gain split
    points in the trees among those with 2025 values on both sides.
The intercept is then adjusted so the 2025 total equals the trees'.

The knot count and Tweedie power are settled on an 80/20 split of the 2025 rows: the candidate with the lowest
held-out mean squared log error against the trees (exposure-weighted) is refitted on every row. Other interaction
counts (at the chosen knots and power) and more knots (at the chosen power) are then fitted, for comparison only.

Reads the Excel export (run python/export_excel.py first) and writes:
  export/excel/buildings_surrogate_terms.csv, buildings_surrogate_spline_pieces.csv,
  buildings_surrogate_input_mappings.csv, buildings_surrogate_reconciliation.csv   aggregate, committed
  export/excel/rows_surrogate_buildings.parquet                                       row-level, never committed
  export/excel/buildings_surrogate_terms_values_seen.parquet  row-level, never committed; only when a spline has too
                                                              many values for a lookup grid (none in 9a528268)
  audit/evidence/emerging/ev_a35_surrogate_specification.csv                          the candidates, held out

Usage (from the pack root): python python/surrogate_buildings.py [--run-label "<label>"]
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import export_excel as ex  # noqa: E402
import glm_lines  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import xgb_models  # noqa: E402

RUNNER = "surrogate_buildings"
NAME, LABEL = "buildings", "surrogate"
OUT = ex.OUT
PREFIX = OUT / "buildings_surrogate"
EVIDENCE = common.PACK_ROOT / "audit" / "evidence" / "emerging" / "ev_a35_surrogate_specification.csv"
N_PAIRS = 10  # the owner's choice: the ten strongest interactions in the trees
SPLIT_POINTS = 4  # each interaction feature is cut at this many of its highest-gain split points
MAX_BAND_VALUES = 20  # a numeric with at most this many values is one band per value, not a spline
KNOT_RANGE = (0.01, 0.99)  # the quantiles of the outer spline knots
ALPHA = 1e-4  # light L2: the target is the trees' prediction, not noisy claims
HOLD_OUT, SEED = 0.2, 1
CANDIDATES = [(knots, p) for knots in (4, 6, 8, 10, 12) for p in (1.0, 1.5, 2.0)]
MORE_PAIRS, MORE_KNOTS = (0, 5, 20), (16, 20)  # comparisons at the chosen power (A35)
PD_QUANTILES = (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)


def interaction_ranking(model: dict) -> list[tuple[str, str, float]]:
    """Pairs of different features that the trees split one after the other (a split and its child's), with the
    child splits' share of all such gain, strongest first (ties by name)."""
    names = model["learner"]["feature_names"]
    gain = {}
    for t in model["learner"]["gradient_booster"]["model"]["trees"]:
        left, right = t["left_children"], t["right_children"]
        for n in range(len(left)):
            if left[n] == -1:
                continue
            for child in (left[n], right[n]):
                if left[child] == -1:
                    continue
                pair = tuple(sorted((names[t["split_indices"][n]], names[t["split_indices"][child]])))
                if pair[0] != pair[1]:
                    gain[pair] = gain.get(pair, 0.0) + t["loss_changes"][child]
    total = sum(gain.values())
    return [(a, b, g / total) for (a, b), g in sorted(gain.items(), key=lambda kv: (-kv[1], kv[0]))]


def tree_split_points(model: dict, x: pd.DataFrame, features: list[str],
                      n: int = SPLIT_POINTS) -> dict[str, np.ndarray]:
    """Each feature's `n` split points with the most gain in the trees, as band edges, among those with rows of `x`
    on both sides (a split point beyond every value would only add an empty band). A band holds lower < x <= upper,
    and each edge is the largest double below the split's exact threshold, so x falls below an edge exactly when the
    trees send it left: a value on a split point goes to the band above, as in the trees."""
    names = model["learner"]["feature_names"]
    gain = {f: {} for f in features}
    for t in model["learner"]["gradient_booster"]["model"]["trees"]:
        for k, (left, i, kind) in enumerate(zip(t["left_children"], t["split_indices"], t["split_type"])):
            if left != -1 and kind == 0 and names[i] in gain:
                g = gain[names[i]]
                g[t["split_conditions"][k]] = g.get(t["split_conditions"][k], 0.0) + t["loss_changes"][k]
    edges = {}
    for f, g in gain.items():
        v = x[f].dropna().to_numpy(dtype=float)
        cut = {c: float(np.nextafter(ex.double_cut(c), -np.inf)) for c in g}
        inside = [c for c in sorted(g, key=lambda c: (-g[c], c)) if (v <= cut[c]).any() and (v > cut[c]).any()]
        edges[f] = np.array(sorted(cut[c] for c in inside[:n]))
    return edges


def value_edges(d: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """One band per value: an edge at every value but the largest, so band qNN holds the NN-th smallest value, and a
    value beyond the range joins the end band."""
    return {c: np.unique(d[c].dropna().to_numpy(dtype=float))[:-1] for c in cols}


def interaction_bands(fit: pd.DataFrame, edges: dict[str, np.ndarray]) -> glm_lines.FactorSpec:
    """The banding the interactions use: each feature cut at its given edges."""
    spec = glm_lines.FactorSpec(fit, sorted(edges), [])
    spec.edges = {f: edges[f] for f in sorted(edges)}  # in place of FactorSpec's quantile edges
    return spec


def with_interactions(d: pd.DataFrame, bands: glm_lines.FactorSpec, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """`d` with one column per interaction, named 'a*b': the two features' bands joined as 'qAA|qBB' (MISSING for
    a missing value)."""
    f = bands.factors(d)
    return d.assign(**{f"{a}*{b}": (f[a] + "|" + f[b]).to_numpy() for a, b in pairs})


def fit_surrogate(d: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str],
                  pairs: list[tuple[str, str]], edges: dict[str, np.ndarray], knots: int,
                  p: float) -> tuple[rgl.GLM, glm_lines.FactorSpec]:
    """The GLM fitted to `y`, the trees' expected cost for the rows of `d`, with the bands its interactions use
    (`edges`: the band edges of each interaction feature)."""
    bands = interaction_bands(d, {f: edges[f] for pair in pairs for f in pair})
    d = with_interactions(d, bands, pairs)
    few = [f for f in numeric if d[f].nunique() <= MAX_BAND_VALUES]
    spec = glm_lines.FactorSpec(d, numeric, categorical + [f"{a}*{b}" for a, b in pairs],
                                smooth=[f for f in numeric if f not in few], n_knots=knots, knot_range=KNOT_RANGE)
    spec.edges.update(value_edges(d, few))  # in place of FactorSpec's quantile bands
    return rgl.GLM(spec, [], ALPHA, trend=False).fit(d, y, p), bands


def fidelity(pred: np.ndarray, target: np.ndarray, exposure: np.ndarray) -> dict:
    """How closely the surrogate follows the trees: the log ratio, exposure-weighted, and the relative error per
    row."""
    lr, lt = np.log(pred / target), np.log(target / exposure)
    msle = float(np.average(lr ** 2, weights=exposure))
    spread = float(np.average((lt - np.average(lt, weights=exposure)) ** 2, weights=exposure))
    err = np.abs(pred / target - 1)
    return {"mean squared log error": msle, "log rate variation explained": 1 - msle / spread,
            "median relative error": float(np.median(err)), "95th percentile relative error":
            float(np.quantile(err, 0.95)), "99th percentile relative error": float(np.quantile(err, 0.99)),
            "largest relative error": float(err.max()), "share within 10%": float((err <= 0.1).mean()),
            "share within 20%": float((err <= 0.2).mean()), "total difference": float(pred.sum() / target.sum() - 1)}


def settle(x: pd.DataFrame, y: np.ndarray, numeric: list[str], categorical: list[str],
           ranking: list[tuple[str, str, float]], edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Every candidate fitted on 80% of the rows and scored on the other 20%, then the comparisons at the chosen
    power. `ranking` holds the pairs of numeric features only, strongest first."""
    held = np.random.default_rng(SEED).random(len(x)) < HOLD_OUT
    fit, out = x[~held], x[held]

    def score(n: int, knots: int, p: float, candidate: bool) -> dict:
        pairs = [r[:2] for r in ranking[:n]]
        if len(pairs) < n:
            raise RuntimeError(f"the trees have {len(ranking)} numeric pairs, fewer than the {n} needed")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            glm, bands = fit_surrogate(fit, y[~held], numeric, categorical, pairs, edges, knots, p)
        pred = glm.predict(with_interactions(out, bands, pairs))
        return {"interactions": len(pairs), "knots": knots, "tweedie_power": p, "candidate": candidate,
                "parameters": len(glm.names) + 1, "solver_iterations": int(glm.model.n_iter_), "warnings": len(caught),
                "fitting_rows": len(fit), "held_out_rows": len(out), **fidelity(pred, y[held], out.exposure.to_numpy())}

    rows = [score(N_PAIRS, knots, p, True) for knots, p in CANDIDATES]
    best = min(range(len(rows)), key=lambda i: (rows[i]["mean squared log error"], rows[i]["parameters"]))
    knots, p = rows[best]["knots"], rows[best]["tweedie_power"]
    rows += [score(n, knots, p, False) for n in MORE_PAIRS] + [score(N_PAIRS, k, p, False) for k in MORE_KNOTS]
    t = pd.DataFrame(rows)
    t["chosen"] = t.index == best
    return t


def band_rows(bands: glm_lines.FactorSpec) -> list[dict]:
    """The interaction bands as terms rows: band qNN is lower < x <= upper, as for a banded term."""
    rows = []
    for f, edges in bands.edges.items():
        for i in range(len(edges) + 1):
            rows.append(ex.term_row(NAME, LABEL, f, "interaction band", f"q{i:02d}",
                                    lower=-np.inf if i == 0 else edges[i - 1],
                                    upper=np.inf if i == len(edges) else edges[i],
                                    note="lower < x <= upper; builds the interaction levels"))
        rows.append(ex.term_row(NAME, LABEL, f, "interaction band", "MISSING", note="x missing"))
    return rows


def add_interactions(t: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """The interaction inputs built from the exported bands alone, as a spreadsheet would: each feature's band,
    then 'band of a|band of b' for each interaction term 'a*b'."""
    bands = t[t.kind == "interaction band"]
    label = {}
    for f, g in bands.groupby("term"):
        v = x[f].astype(float).to_numpy()
        s = np.full(len(v), "MISSING", dtype=object)
        for r in g[g["level"] != "MISSING"].itertuples():
            s[(v > r.band_lower) & (v <= r.band_upper)] = r.level
        label[f] = s
    terms = [c for c in t[t.kind == "categorical"].term.unique() if "*" in c]
    return x.assign(**{c: [a + "|" + b for a, b in zip(label[c.split("*")[0]], label[c.split("*")[1]])]
                       for c in terms})


def read_terms(path: Path) -> pd.DataFrame:
    """A terms file with the levels as text and every number exactly as written, as export_excel reads its own."""
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("value", "band_lower", "band_upper", "contribution"):
        t[col] = [float(v) if v else np.nan for v in t[col]]
    return t


def with_derived(d: pd.DataFrame, moved: str) -> pd.DataFrame:
    """`d` with the trees' derived inputs recomputed after `moved` changed (definitions in export_excel.CONTINUOUS)."""
    if moved in ("flood_zone", "rainfall_mm") and "x_fz3_rain" in d:
        d = d.assign(x_fz3_rain=((d.flood_zone.fillna(99) >= 3) * d.rainfall_mm / 1000).fillna(0).astype(float))
    if moved in ("soil_type", "dry_summer_index") and "x_clay_dry" in d:
        d = d.assign(x_clay_dry=((d.soil_type == "clay").astype(float) * d.dry_summer_index).fillna(0).astype(float))
    return d


def tree_scorer(features: pd.DataFrame, sc: pd.DataFrame):
    """The trees' expected cost in YEAR for any rows, from the exported model file and its features table."""
    booster = xgb.Booster(model_file=str(OUT / f"{NAME}_xgboost.json"))
    order = features.sort_values(["position", "code"], kind="stable")
    numeric = order[order.type == "numeric"].feature.tolist()
    cats = {}  # in the model's feature order, each with its levels in code order
    for _, g in order[order.type == "categorical"].groupby("position", sort=True):
        cats[g.feature.iloc[0]] = g[g.code.notna()]["level"].tolist()
    get = lambda item: float(sc[(sc.line == NAME) & (sc["item"] == item)].value.iloc[0])  # noqa: E731
    base, slope, centre = (get("XGBoost base rate (in the offset)"),
                           get("small-claim trend slope (per year, log scale)"), get("small-claim trend origin (year)"))
    return lambda d: xgb_models.predict(booster, xgb_models.matrix(d, numeric, cats), d.exposure.to_numpy(), base,
                                        extra=slope * (ex.YEAR - centre)).astype(float)


def model_terms(x: pd.DataFrame) -> dict:
    """What the surrogate takes from the exported trees for the rows of `x`: the features table, the numeric and
    categorical features the trees split on, the interaction ranking (every pair, then the numeric pairs, which the
    comparisons go further down) and the band edges of each feature in the numeric pairs the comparisons use."""
    features = pd.read_csv(OUT / "buildings_xgboost_features.csv", keep_default_na=False, na_values=[""],
                           dtype={"level": str})
    split_on = set(pd.read_csv(OUT / "buildings_xgboost_nodes.csv", usecols=["feature"]).feature.dropna())
    numeric = [f for f in features[features.type == "numeric"].feature.unique() if f in split_on]
    categorical = [f for f in features[features.type == "categorical"].feature.unique() if f in split_on]
    model = json.loads((OUT / f"{NAME}_xgboost.json").read_text())
    ranking = interaction_ranking(model)
    if any(f not in numeric for a, b, _ in ranking[:N_PAIRS] for f in (a, b)):
        raise RuntimeError("one of the strongest interactions involves a categorical feature, which the banded "
                           "cross does not cover")
    numeric_ranking = [r for r in ranking if r[0] in numeric and r[1] in numeric]
    edges = tree_split_points(model, x, sorted({f for r in numeric_ranking[:max(MORE_PAIRS + (N_PAIRS,))]
                                                for f in r[:2]}))
    return {"features": features, "numeric": numeric, "categorical": categorical, "ranking": ranking,
            "numeric_ranking": numeric_ranking, "pairs": [r[:2] for r in numeric_ranking[:N_PAIRS]], "edges": edges}


def final_fit(x: pd.DataFrame, y: np.ndarray, m: dict, knots: int, p: float):
    """The surrogate fitted on every row, with ln(factor) added to its intercept so the 2025 total equals the
    trees'. Returns the GLM, its interaction bands, the rows with their interaction inputs, the factor and the
    warnings the fit raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        glm, bands = fit_surrogate(x, y, m["numeric"], m["categorical"], m["pairs"], m["edges"], knots, p)
    xi = with_interactions(x, bands, m["pairs"])
    factor = y.sum() / glm.predict(xi).sum()
    glm.model.intercept_ = glm.model.intercept_ + np.log(factor)
    return glm, bands, xi, float(factor), caught


def run(cfg: dict, rec: dict) -> None:
    x = pd.read_parquet(OUT / "rows_inputs_buildings.parquet")
    comp = pd.read_parquet(OUT / "rows_components.parquet").set_index("row_id").loc[x.row_id]
    y = comp.small_claim.to_numpy()
    inputs = [OUT / "buildings_xgboost.json", OUT / "buildings_xgboost_features.csv", OUT / "buildings_xgboost_nodes.csv",
              OUT / "scalars.csv", OUT / "rows_inputs_buildings.parquet", OUT / "rows_components.parquet"]
    rec["inputs_sha256"] = {p.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(p) for p in inputs}
    m = model_terms(x)
    features, numeric, categorical, ranking, pairs = (m[k] for k in ("features", "numeric", "categorical", "ranking",
                                                                       "pairs"))

    spec_table = settle(x, y, numeric, categorical, m["numeric_ranking"], m["edges"])
    spec_table.to_csv(EVIDENCE, index=False)
    chosen = spec_table[spec_table.chosen].iloc[0]
    glm, bands, xi, factor, caught = final_fit(x, y, m, int(chosen.knots), float(chosen.tweedie_power))
    pred = glm.predict(xi)
    sc = pd.read_csv(OUT / "scalars.csv", dtype=str, keep_default_na=False)
    slope = float(sc[(sc.line == NAME) & (sc["item"] == "small-claim trend slope (per year, log scale)")].value.iloc[0])
    r = ex.export_glm(NAME, LABEL, glm, xi, xi, slope, None, fitted_on="2025 buildings")

    terms = pd.DataFrame(r["terms"] + band_rows(bands))
    seen = terms.note.eq(ex.SEEN)
    seen_path = OUT / "buildings_surrogate_terms_values_seen.parquet"
    if seen.any():
        terms[seen].to_parquet(seen_path, index=False)
        terms = pd.concat([terms[~seen], terms[seen].drop_duplicates(["line", "component", "term"]).assign(
            value=np.nan, contribution=np.nan,
            note=f"lookup in {seen_path.name} (the values that occur: competition data, not committed); the exact "
                 f"curve is in {PREFIX.name}_spline_pieces.csv")]).sort_index(kind="stable")
    else:
        seen_path.unlink(missing_ok=True)
    terms.to_csv(f"{PREFIX}_terms.csv", index=False)
    pd.DataFrame(r["pieces"]).to_csv(f"{PREFIX}_spline_pieces.csv", index=False)
    add = lambda term, source, mapped, note: {"section": "interaction", "line": NAME, "component": LABEL,  # noqa: E731
                                              "term": term, "source_value": source, "mapped_value": mapped,
                                              "note": note}
    pd.DataFrame(r["maps"] + [add(f"{a}*{b}", f"band of {a} | band of {b}", "qAA|qBB",
                                  f"bands: kind 'interaction band' in {PREFIX.name}_terms.csv; then map the pair as a "
                                  "categorical level; a pair with no row here is OTHER")
                              for a, b in pairs]).to_csv(f"{PREFIX}_input_mappings.csv", index=False)

    rows = pd.DataFrame({"row_id": x.row_id.to_numpy(), "exposure": x.exposure.to_numpy(),
                         "new_business": comp.new_business.to_numpy(), "reference_policy":
                         comp.reference_policy.to_numpy(), "small_claim": y, "surrogate": pred, "ratio": pred / y})
    rows = rows.merge(r["rows"], on="row_id").merge(r["levels"].rename_axis("row_id").reset_index(), on="row_id")
    rows.to_parquet(OUT / "rows_surrogate_buildings.parquet", index=False)

    checks = list(r["checks"])
    add_check = lambda check, value, passed, kind="check", note="": checks.append(  # noqa: E731
        {"check": check, "line": NAME, "component": LABEL, "term": "", "value": value,
         "passed": passed if kind == "check" else None, "note": note, "type": kind})
    t = read_terms(Path(f"{PREFIX}_terms.csv"))
    pieces = pd.read_csv(f"{PREFIX}_spline_pieces.csv", float_precision="round_trip")
    maps = pd.read_csv(f"{PREFIX}_input_mappings.csv", dtype=str, keep_default_na=False)
    maps = maps[maps.section == "categorical level"]
    seen_rows = pd.read_parquet(seen_path) if seen_path.exists() else None
    xin = add_interactions(t, x)  # from the exported bands, not the fitted ones
    for route in ("lookup", "pieces"):
        again = ex.evaluate_glm(t, pieces, maps, xin, route, seen_rows)
        gap = float(np.abs(again / pred - 1).max())
        add_check(f"surrogate re-evaluated from its exported tables ({route}) against the fitted GLM, max relative "
                  "difference", gap, bool(gap < 1e-12))
    gap = float(abs(pred.sum() / y.sum() - 1))
    add_check("surrogate 2025 total against the trees' total, relative difference", gap, bool(gap < 1e-12))
    add_check("ln of this added to the intercept, so the 2025 totals agree (the exported intercept includes it)",
              float(factor), None, "information")
    add_check("solver iterations of the final fit (limit 1000)", int(glm.model.n_iter_),
              bool(glm.model.n_iter_ < models.SOLVER["max_iter"]))
    add_check("warnings raised by the final fit", len(caught), len(caught) == 0)
    X, _ = models.design_matrix(glm._frame(xi), glm.levels, glm.numeric)
    X = np.column_stack([np.ones(len(X)), X])
    add_check("design columns, intercept included, and their rank: L2 shares an effect between aliased columns",
              f"{X.shape[1]} / {int(np.linalg.matrix_rank(X))}", None, "information")
    for k, v in fidelity(pred, y, x.exposure.to_numpy()).items():
        add_check(f"against the trees, every 2025 row (in sample): {k}", v, None, "information")
    held = chosen.to_dict()
    for k in ("log rate variation explained", "median relative error", "95th percentile relative error",
              "99th percentile relative error", "largest relative error", "total difference"):
        add_check(f"against the trees, the 20% held out when choosing the specification: {k}", held[k], None,
                  "information", "from ev_a35_surrogate_specification.csv")
    nb = rows.new_business.astype(bool)
    for label, m in (("new business", nb), ("renewals", ~nb)):
        e = np.abs(rows.ratio[m] - 1)
        add_check(f"against the trees, {label}: median and 95th percentile relative error",
                  f"{np.median(e):.6f} / {np.quantile(e, 0.95):.6f}", None, "information")
    ref = np.abs(rows.ratio[rows.reference_policy] - 1)
    add_check(f"against the trees, the {len(ref)} buildings reference policies: largest relative error",
              float(ref.max()), None, "information")

    # One feature at a time: every 2025 row moved to one value, surrogate total against the trees' total. This is
    # how far a single term, read as a relativity, departs from the trees.
    trees = tree_scorer(features, sc)
    bad = int((trees(x) != y).sum())
    add_check("trees rescored from buildings_xgboost.json for the one-feature check: rows that differ from "
              "small_claim", bad, bad == 0)
    for derived, source in (("x_fz3_rain", "flood_zone"), ("x_clay_dry", "soil_type")):
        if derived in x:
            bad = int((with_derived(x, source)[derived].to_numpy() != x[derived].to_numpy()).sum())
            add_check(f"{derived} rebuilt from its sources, for moving them: rows that differ from the input", bad,
                      bad == 0)
    for f in sorted({f for pair in pairs for f in pair} - set(ex.CONTINUOUS)):
        v = x[f].dropna().to_numpy(dtype=float)
        values = np.unique(v) if len(np.unique(v)) <= MAX_BAND_VALUES else \
            np.unique(np.quantile(v, PD_QUANTILES, method="inverted_cdf"))
        for value in values:
            d = with_derived(x.assign(**{f: value}), f)
            gap = float(glm.predict(with_interactions(d, bands, pairs)).sum() / trees(d).sum() - 1)
            add_check(f"every 2025 row at {f} = {value:g}: surrogate total against the trees' total", gap, None,
                      "information")
    add_check("specification", f"{len(pairs)} interactions cut at up to {SPLIT_POINTS} tree split points each; "
              f"splines with {int(chosen.knots)} knots from the {KNOT_RANGE[0]:.0%} to the {KNOT_RANGE[1]:.0%} "
              f"quantile, one band per value for numerics with at most {MAX_BAND_VALUES} values; Tweedie power "
              f"{chosen.tweedie_power:g}; L2 {ALPHA:g}", None, "information")
    for a, b, share in ranking[:N_PAIRS]:
        add_check(f"interaction {a}*{b}: share of the trees' parent-and-child split gain", share, None, "information")
    for p, h in rec["inputs_sha256"].items():
        add_check(f"sha256 of {Path(p).name}", h, None, "information")
    out = pd.DataFrame(checks)
    out["type"] = out["type"].fillna("check")
    out["passed"] = out.passed.astype(object).where(out.type == "check", None)
    out.to_csv(f"{PREFIX}_reconciliation.csv", index=False)
    rec["checks_failed"] = out[(out.type == "check") & ~out.passed.astype(bool)].check.tolist()
    rec["outputs_sha256"] = {p.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(p)
                             for p in sorted(OUT.glob("buildings_surrogate*")) + [OUT / "rows_surrogate_buildings.parquet",
                                                                                  EVIDENCE]}
    print(spec_table.to_string(index=False))
    print(out[["type", "check", "value", "passed"]].to_string(index=False))


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
                           **{pkg: metadata.version(pkg) for pkg in ("scikit-learn", "xgboost", "scipy")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
