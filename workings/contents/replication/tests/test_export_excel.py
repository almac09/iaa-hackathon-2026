"""Unit tests for the Excel export (export_excel.py): its exact conversions, and a synthetic GLM and XGBoost model
exported and then re-evaluated from their own tables, as the spreadsheet will."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import export_excel as ex  # noqa: E402
import glm_lines  # noqa: E402
import run_glm_lines as rgl  # noqa: E402


def fitted_spline(x):
    knots = np.unique(np.quantile(x, np.linspace(0, 1, 4)))  # as glm_lines.FactorSpec does
    return SplineTransformer(degree=3, knots=knots.reshape(-1, 1), extrapolation="constant",
                             include_bias=False).fit(x.reshape(-1, 1))


def test_cubic_pieces_reproduce_the_fitted_curve_inside_and_outside_the_knots():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 5.0, 2000)
    st = fitted_spline(x)
    coef = rng.normal(size=st.n_features_out_)
    sp, lo, hi = ex.spline_function(st, coef)
    pieces = pd.DataFrame(ex.spline_pieces(sp, lo, hi))
    probe = np.concatenate([np.linspace(lo - 20, hi + 20, 5001), [lo, hi], np.unique(st.bsplines_[0].t)])
    curve = st.transform(probe.reshape(-1, 1)) @ coef
    assert np.abs(ex.eval_pieces(pieces, probe) - curve).max() < 1e-10
    assert not np.isnan(ex.eval_pieces(pieces, probe)).any()  # every probe falls in a piece
    assert pieces.iloc[0].a1 == pieces.iloc[-1].a1 == 0.0  # constant beyond the knots


def test_exact_scale_reproduces_the_product_bit_for_bit():
    rng = np.random.default_rng(2)
    a = rng.gamma(1.5, 300.0, 5000)
    m = float(np.sqrt(0.916777))
    b = a * m
    found, how, hits = ex.exact_scale(a, b, [m])
    assert np.array_equal(a * found, b) and hits >= 1 and how.startswith("computed")
    found, how, _ = ex.exact_scale(a, b, [])  # no candidate: recovered from the ratios
    assert np.array_equal(a * found, b) and how.startswith("recovered")


def test_lookup_grid_uses_the_recorded_precision_and_never_adds_data_values():
    v = np.array([0.0, 12.3, 59.9, 60.0])
    grid, how = ex.lookup_values(v, v)
    assert len(grid) == 601 and grid[3] == 0.3 and "steps of 0.1" in how
    si = np.array([44_000.0, 45_000.0, 894_000.0])  # sum insured in thousands of pounds
    grid, how = ex.lookup_values(si, si)
    assert len(grid) == 851 and "steps of 1000" in how
    wide = np.array([4_900.0, 1_000_000.0, 98_095_100.0])
    grid, how = ex.lookup_values(wide, wide)
    assert how == ex.SEEN and len(grid) == 3
    with pytest.raises(ValueError):
        ex.lookup_values(v, np.array([12.34]))  # a seen value off the grid


def synthetic(seed: int, n: int = 4000) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"row_id": [f"R{i:05d}" for i in range(n)], "yr": rng.integers(2021, 2025, n),
                      "exposure": rng.uniform(0.2, 1.0, n), "band": rng.integers(0, 6, n).astype(float),
                      "curve": np.round(rng.gamma(2.0, 3.0, n), 1), "z": rng.normal(size=n),
                      "cat": rng.choice(["a", "b", "c", "rare"], n, p=[0.5, 0.3, 0.198, 0.002])})
    d.loc[rng.random(n) < 0.05, "curve"] = np.nan
    d.loc[rng.random(n) < 0.05, "cat"] = None
    mu = 100 * d.exposure * np.exp(0.2 * (d.cat == "b") + 0.05 * d.band + 0.1 * d.z)
    return d, np.where(rng.random(n) < 0.1, rng.gamma(2.0, mu / 0.2), 0.0)


@pytest.mark.parametrize("label", ["small", "large"])
def test_a_glm_rebuilt_from_its_exported_tables_matches_the_model(label):
    d, y = synthetic(3)
    spec = glm_lines.FactorSpec(d, ["band", "curve"], ["cat"], n_bands=5, smooth=["curve"])
    small = label == "small"  # a large-claim GLM has no year term: it takes another model's slope, and a Pareto scale
    glm = rgl.GLM(spec, ["z"], 1e-4, trend=small).fit(d, y, 1.5)
    slope, scale = (glm.slope, None) if small else (0.04, 1.0254)
    r = ex.export_glm("test", label, glm, d, d, slope, scale)
    terms, pieces, maps = pd.DataFrame(r["terms"]), pd.DataFrame(r["pieces"]), pd.DataFrame(r["maps"])
    assert set(terms.kind) >= {"intercept", "categorical", "band", "spline value", "spline missing", "continuous",
                               "trend", "offset"} | (set() if small else {"scale"})
    assert maps[maps.source_value == "rare"].mapped_value.iloc[0] == "OTHER"  # under 0.5% of exposure
    model = glm.predict(d, ex.YEAR if small else None)
    if not small:
        model = model * scale * np.exp(slope * (ex.YEAR - glm.centre))
    for route in ("lookup", "pieces"):
        assert np.abs(ex.evaluate_glm(terms, pieces, maps, d, route) / model - 1).max() < 1e-12
    assert np.abs(np.exp(r["rows"][f"{label}__total_log"]) / model - 1).max() < 1e-12
    # A model level the fit never saw scores as the base level, as '(any other level)' says.
    new = d.head(50).assign(cat="unseen")
    unseen = pd.concat([maps, pd.DataFrame([{"section": "categorical level", "line": "test", "component": label,
                                             "term": "cat", "source_value": "unseen", "mapped_value": "never fitted"}])])
    base = d.head(50).assign(cat=glm.levels["cat"][0])
    assert np.allclose(ex.evaluate_glm(terms, pieces, unseen, new, "pieces"),
                       ex.evaluate_glm(terms, pieces, maps, base, "pieces"), rtol=1e-14, atol=0)


def test_double_cut_takes_xgboosts_branch_for_every_double():
    rng = np.random.default_rng(4)
    cuts = np.concatenate([rng.normal(0, 1, 300), rng.lognormal(4, 3, 300) * rng.choice([-1, 1], 300),
                           [0.0, 1e-30, -2.5, 45_000.0]]).astype(np.float32)
    for t in cuts:
        c = ex.double_cut(float(t))
        below = float(np.nextafter(t, np.float32(-np.inf)))
        for x in (c, np.nextafter(c, -np.inf), np.nextafter(c, np.inf), float(t), np.nextafter(float(t), -np.inf),
                  np.nextafter(float(t), np.inf), below, (below + float(t)) / 2):
            assert (x < c) == (np.float32(x) < t)  # XGBoost: left when float32(x) < t
    # A decimal input on a single-precision threshold: XGBoost sends 12.3 right, a plain double comparison left.
    t = np.float32(12.3)
    assert 12.3 < float(t) and not np.float32(12.3) < t and not 12.3 < ex.double_cut(float(t))


def test_trees_rebuilt_from_their_exported_tables_match_xgboost(tmp_path):
    d, _ = synthetic(5)
    d["cat"] = d.cat.replace({"c": "NA"})  # a level pandas would read as missing by default
    rng = np.random.default_rng(6)
    mu = 100 * d.exposure * np.exp(1.2 * (d.cat == "b") - 0.9 * (d.cat == "NA") + 0.1 * d.band + 0.3 * d.z)
    y = np.where(rng.random(len(d)) < 0.3, rng.gamma(2.0, mu / 0.6), 0.0)
    s = rgl.XGBSmall(["band", "curve", "z"], ["cat"], d, {"max_depth": 3, "eta": 0.3, "min_child_weight": 1.0}, 25,
                     {"seed": 7, "nthread": 1}, 1).fit(d, y, 1.5)
    r = ex.export_trees(tmp_path / "model.json", s, d, s.predict(d, ex.YEAR))
    r["nodes"].to_csv(tmp_path / "nodes.csv", index=False)
    r["features"].to_csv(tmp_path / "features.csv", index=False)
    nodes, features = ex.read_trees(tmp_path / "nodes.csv", tmp_path / "features.csv")
    assert {"split", "category split", "leaf"} <= set(nodes.kind)
    assert d.curve.isna().any() and d.cat.isna().any()  # missing values take the default branches
    for c in ("threshold_float32", "threshold", "leaf_value"):  # read back exactly as written
        assert np.array_equal(nodes[c], r["nodes"][c].astype(float), equal_nan=True)
    held = pd.concat([nodes.threshold_float32, nodes.leaf_value]).dropna().to_numpy()
    assert np.array_equal(held.astype(np.float32).astype(float), held)  # exact single-precision values
    level = {(f, c): v for f, c, v in zip(features.feature, features.code, features["level"]) if pd.notna(c)}
    assert all(n.right_levels.split(";") == [level[(n.feature, float(c))] for c in n.right_codes.split(";")]
               for n in nodes[nodes.kind == "category split"].itertuples())
    split = nodes[nodes.kind == "split"]
    on_cut = [d.head(20).assign(**{f: c}) for f, c in zip(split.feature, split.threshold)]  # x == threshold goes right
    score = pd.concat([d, d.head(200).assign(cat="unseen"), *on_cut], ignore_index=True)  # and a level never seen
    model = s.predict(score, ex.YEAR).astype(float)
    offset = ex.tree_offset(score.exposure.to_numpy(), s.base, s.slope, s.centre)
    assert np.array_equal(ex.score_trees(nodes, features, score, offset, single=True), model)
    assert np.abs(ex.score_trees(nodes, features, score, offset, single=False) / model - 1).max() < 1e-5
