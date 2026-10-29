"""Scratch (A35): the figures in the surrogate's README section and the A35 entry that python/surrogate_buildings.py
does not write itself.

    python doc_figures.py <pack>/audit/evidence/emerging
Reads the Excel export in <pack>/export/excel (run python/export_excel.py and python/surrogate_buildings.py first)
and writes one aggregate file; no row-level data is written.
  ev_a35_doc_figures.csv  one line per figure: section, figure, value

It refits the surrogate exactly as the run does, checks that the refit reproduces rows_surrogate_buildings.parquet
bit for bit and that the rescored trees reproduce small_claim, and then:
  - choice    every candidate's held-out error on two more random splits (seeds 2 and 3)
  - knots     each spline's tree split gain beyond its outer knots, on splits with 2025 values on both sides
  - tails     every 2025 row moved to a climate feature's 0.5th and 99.5th percentile
  - aliasing  the design's rank, where the aliasing lies, and the aliased terms' contributions
  - renewal   every 2025 row one renewal on, as for its quote a year later
  - zones     flood zones 4 and 5, as they are and with every row moved there, by elevation
  - other     the 2025 rows with an interaction pair pooled as OTHER
  - missing   every 2025 row with a categorical input missing, for each categorical never missing in 2025
  - cells     the parameters of the 80% fit against the final fit"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACK / "python"))
import models  # noqa: E402
import surrogate_buildings as sb  # noqa: E402

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
OUT = Path(sys.argv[1])
CLIMATE = ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index", "wildfire_index", "rainfall_mm",
           "mean_temperature_c"]
# elevation bands: the surrogate's own interaction bands (q00 is under 2.1 m, q04 is 2.8 m or more), q04 split further
ELEVATION = [("under 2.1 m", lambda b, e: b == "q00"),
             ("2.1 to under 2.8 m", lambda b, e: np.isin(b, ["q01", "q02", "q03"])),
             ("2.8 to under 5 m", lambda b, e: (b == "q04") & (e < 5)),
             ("5 to under 10 m", lambda b, e: (e >= 5) & (e < 10)),
             ("10 to under 20 m", lambda b, e: (e >= 10) & (e < 20)),
             ("20 m or more", lambda b, e: e >= 20)]

rows = []


def add(section: str, figure: str, value) -> None:
    rows.append({"section": section, "figure": figure, "value": value})


def gap(s: np.ndarray, t: np.ndarray) -> float:
    return float(s.sum() / t.sum() - 1)


x = pd.read_parquet(sb.OUT / "rows_inputs_buildings.parquet")
y = pd.read_parquet(sb.OUT / "rows_components.parquet").set_index("row_id").loc[x.row_id].small_claim.to_numpy()
spec = pd.read_csv(sb.EVIDENCE, float_precision="round_trip")
chosen = spec[spec.chosen].iloc[0]
m = sb.model_terms(x)
glm, bands, xi, factor, _ = sb.final_fit(x, y, m, int(chosen.knots), float(chosen.tweedie_power))
pred = glm.predict(xi)
if not np.array_equal(pred, pd.read_parquet(sb.OUT / "rows_surrogate_buildings.parquet").surrogate.to_numpy()):
    raise RuntimeError("the refit does not reproduce rows_surrogate_buildings.parquet")
trees = sb.tree_scorer(m["features"], pd.read_csv(sb.OUT / "scalars.csv", dtype=str, keep_default_na=False))
if not np.array_equal(trees(x), y):
    raise RuntimeError("the rescored trees do not reproduce small_claim")
terms = sb.read_terms(sb.OUT / "buildings_surrogate_terms.csv")
cross = [f"{a}*{b}" for a, b in m["pairs"]]
err = np.abs(pred / y - 1)


def moved(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The surrogate and the trees for rows `d`, and whether each row has an interaction pair pooled as OTHER."""
    di = sb.with_interactions(d, bands, m["pairs"])
    return glm.predict(di), trees(d), (glm.spec.factors(di)[cross] == "OTHER").any(axis=1).to_numpy()


# choice: the candidates on two more random splits (seed 1's are in ev_a35_surrogate_specification.csv)
seed = sb.SEED
sb.MORE_PAIRS, sb.MORE_KNOTS = (), ()
for other_seed in (2, 3):
    sb.SEED = other_seed
    t = sb.settle(x, y, m["numeric"], m["categorical"], m["numeric_ranking"], m["edges"])
    for _, r in t.iterrows():
        add("choice", f"seed {other_seed}: held-out mean squared log error, {r.knots} knots, power {r.tweedie_power:g}",
            float(r["mean squared log error"]))
sb.SEED = seed

# knots: tree split gain beyond each spline's outer knots
learner = json.loads((sb.OUT / "buildings_xgboost.json").read_text())["learner"]
names, all_gain, splits = learner["feature_names"], 0.0, {}
for t in learner["gradient_booster"]["model"]["trees"]:
    for c, g, i, left, kind in zip(t["split_conditions"], t["loss_changes"], t["split_indices"], t["left_children"],
                                   t["split_type"]):
        if left != -1:
            all_gain += g
            if kind == 0:
                splits.setdefault(names[i], []).append((np.float32(c), g))
for f, spline in glm.spec.splines.items():
    knots = spline.bsplines_[0].t[3:-3]
    v = x[f].dropna().to_numpy(dtype=float)
    total = sum(g for _, g in splits.get(f, []))
    inside = [(c, g) for c, g in splits.get(f, []) if v.min() < c <= v.max()]  # 2025 values on both sides
    add("knots", f"{f}: share of its split gain, on splits inside the 2025 range, at or below the first knot "
        f"({knots[0]:g})", sum(g for c, g in inside if c <= knots[0]) / total)
    add("knots", f"{f}: share of its split gain, on splits inside the 2025 range, above the last knot ({knots[-1]:g})",
        sum(g for c, g in inside if c > knots[-1]) / total)
    add("knots", f"{f}: share of all the trees' split gain", total / all_gain)

# tails: every row at a climate feature's 0.5th and 99.5th percentile, derived inputs rebuilt
for f in CLIMATE:
    v = x[f].dropna().to_numpy(dtype=float)
    for q in (0.005, 0.995):
        value = float(np.quantile(v, q, method="inverted_cdf"))
        s, t, _ = moved(sb.with_derived(x.assign(**{f: value}), f))
        add("tails", f"every row at {f} = {value:g} (its {q:.1%} quantile): surrogate total against the trees'",
            gap(s, t))

# aliasing
X, cols = models.design_matrix(glm._frame(xi), glm.levels, glm.numeric)
X = np.column_stack([np.ones(len(X)), X])
term = ["(intercept)"] + [c.split("=")[0] if "=" in c else c.rsplit("_s", 1)[0].removesuffix("_missing")
                          for c in cols]
main = np.array(["*" not in f for f in term])


def deficit(keep: np.ndarray) -> int:
    return int(keep.sum() - np.linalg.matrix_rank(X[:, keep]))


add("aliasing", "design columns, intercept included / rank deficit",
    f"{X.shape[1]} / {deficit(np.ones(len(term), bool))}")
add("aliasing", "main-term columns, intercept included / rank deficit", f"{main.sum()} / {deficit(main)}")
keep = main.copy()
for f in ("renewal_count", "customer_since_year", "discount_applied"):
    keep &= np.array([name != f for name in term])
    add("aliasing", f"main-term rank deficit once {f} is also left out", deficit(keep))
add("aliasing", "share of 2025 rows with customer_since_year = 2025 - renewal_count",
    float((x.customer_since_year + x.renewal_count == 2025).mean()))
add("aliasing", "share of 2025 rows on which campaign_code is missing exactly when renewal_count >= 3",
    float((x.campaign_code.isna() == (x.renewal_count >= 3)).mean()))
add("aliasing", "share of 2025 rows on which a 5% discount goes exactly with the direct channel",
    float(((x.discount_applied == 0.05) == (x.channel == "direct")).mean()))
r = pd.read_parquet(sb.OUT / "rows_surrogate_buildings.parquet")
add("aliasing", "largest |renewal_count + customer_since_year contribution| on a row (log scale)",
    float((r.surrogate__renewal_count + r.surrogate__customer_since_year).abs().max()))
for f, level in (("discount_applied", "q01"), ("channel", "direct")):
    add("aliasing", f"contribution of {f} = {level} (log scale)",
        float(terms[(terms.term == f) & (terms.level == level)].contribution.iloc[0]))

# renewal: the same customers a year on, with one more renewal and no campaign code from the third
keep = (x.renewal_count < x.renewal_count.max()).to_numpy()
d = x[keep].assign(renewal_count=x.renewal_count[keep] + 1)
d.loc[d.renewal_count >= 3, "campaign_code"] = pd.NA
s, t, _ = moved(d)
add("renewal", "every row one renewal on: surrogate total against the trees'", gap(s, t))
add("renewal", "every row one renewal on: median relative error, as is / moved",
    f"{np.median(err[keep]):.6f} / {np.median(np.abs(s / t - 1)):.6f}")

# zones
eb, e = bands.factors(x)["elevation_m"].to_numpy(), x.elevation_m.to_numpy()
for zone in (4, 5):
    real = (x.flood_zone == zone).to_numpy()
    add("zones", f"rows really in zone {zone}: rows / surrogate total against the trees' / median relative error",
        f"{real.sum()} / {gap(pred[real], y[real]):.6f} / {np.median(err[real]):.6f}")
    s, t, pooled = moved(sb.with_derived(x.assign(flood_zone=float(zone)), "flood_zone"))
    for label, rule in ELEVATION:
        b = rule(eb, e)
        add("zones", f"every row moved to zone {zone}, elevation {label}: rows / surrogate total against the trees'",
            f"{b.sum()} / {gap(s[b], t[b]):.6f}")
    add("zones", f"every row moved to zone {zone}: share with an OTHER pair / total on them / total on the rest",
        f"{pooled.mean():.6f} / {gap(s[pooled], t[pooled]):.6f} / {gap(s[~pooled], t[~pooled]):.6f}")
    if zone == 5:
        high = eb == "q04"
        ratio = s[high] / t[high] - 1
        add("zones", "every row moved to zone 5, 2.8 m or more: per-row median / 95th percentile / largest overpricing",
            f"{np.median(ratio):.6f} / {np.quantile(ratio, 0.95):.6f} / {ratio.max():.6f}")
        add("zones", "rows really in zone 5 at 2.1 m or more (outside elevation band q00)",
            int((real & ~np.isin(eb, ["q00", "MISSING"])).sum()))
    else:
        high = e >= 5
        add("zones", "every row moved to zone 4, 5 m or more: rows / with no OTHER pair / total on those",
            f"{high.sum()} / {(high & ~pooled).sum()} / {gap(s[high & ~pooled], t[high & ~pooled]):.6f}")
        add("zones", "rows really in zone 4 at 5 m or more: rows / surrogate total against the trees'",
            f"{(real & high).sum()} / {gap(pred[real & high], y[real & high]):.6f}")

# other: the 2025 rows with a pooled pair
pooled = (glm.spec.factors(xi)[cross] == "OTHER").any(axis=1).to_numpy()
add("other", "2025 rows with an interaction pair pooled as OTHER: share", float(pooled.mean()))
add("other", "median / 95th percentile relative error on them; on the rest",
    f"{np.median(err[pooled]):.6f} / {np.quantile(err[pooled], 0.95):.6f}; "
    f"{np.median(err[~pooled]):.6f} / {np.quantile(err[~pooled], 0.95):.6f}")

# missing: a categorical input missing on every row, for those never missing in 2025
for f in m["categorical"]:
    if not x[f].isna().any():
        s, t, _ = moved(sb.with_derived(x.assign(**{f: pd.Series(None, index=x.index, dtype=object)}), f))
        add("missing", f"every row with {f} missing: surrogate total against the trees' / per-row median |gap|",
            f"{gap(s, t):.6f} / {np.median(np.abs(s / t - 1)):.6f}")
main_cats = terms[(terms.kind == "categorical") & ~terms.term.str.contains("*", regex=False)]
add("missing", "categoricals in the terms / with an OTHER level / with a MISSING level",
    f"{main_cats.term.nunique()} / {main_cats[main_cats.level == 'OTHER'].term.nunique()} / "
    f"{main_cats[main_cats.level == 'MISSING'].term.nunique()}")

# cells: the 80% fit's parameters against the final fit's
held = np.random.default_rng(seed).random(len(x)) < sb.HOLD_OUT
part, _ = sb.fit_surrogate(x[~held], y[~held], m["numeric"], m["categorical"], m["pairs"], m["edges"],
                           int(chosen.knots), float(chosen.tweedie_power))
add("cells", "parameters, 80% fit / final fit", f"{len(part.names) + 1} / {len(glm.names) + 1}")
add("cells", "columns only in the final fit", "; ".join(sorted(set(glm.names) - set(part.names))))
add("cells", "columns only in the 80% fit", "; ".join(sorted(set(part.names) - set(glm.names))))

pd.DataFrame(rows).to_csv(OUT / "ev_a35_doc_figures.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
