"""Unit tests for the buildings surrogate GLM (surrogate_buildings.py): the interaction ranking and bands, the
accuracy measures, a synthetic surrogate exported and then re-evaluated from its own tables, interactions included,
as the spreadsheet will, and a whole run on synthetic trees."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402
import export_excel as ex  # noqa: E402
import glm_lines  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import surrogate_buildings as sb  # noqa: E402


def tree(features, left, right, gains, cuts=None, kinds=None):
    n = len(features)
    return {"split_indices": features, "left_children": left, "right_children": right, "loss_changes": gains,
            "split_conditions": cuts or [0.0] * n, "split_type": kinds or [0] * n}


def model(*trees, names=("a", "b", "c")):
    return {"learner": {"feature_names": list(names), "gradient_booster": {"model": {"trees": list(trees)}}}}


def test_interaction_ranking_counts_parent_and_child_splits_by_the_childs_gain():
    m = model(
        # node 0 splits on a; its children split on b (gain 3) and a (gain 5, same feature: not a pair)
        tree([0, 1, 0, 0, 0, 0, 0], [1, 3, 5, -1, -1, -1, -1], [2, 4, 6, -1, -1, -1, -1], [9, 3, 5, 0, 0, 0, 0]),
        # node 0 splits on c, its left child on b (gain 1); the right child is a leaf
        tree([2, 1, 0, 0, 0], [1, 3, -1, -1, -1], [2, 4, -1, -1, -1], [7, 1, 0, 0, 0]))
    assert sb.interaction_ranking(m) == [("a", "b", 0.75), ("b", "c", 0.25)]


def test_tree_split_points_keep_the_highest_gain_thresholds_and_send_a_value_on_one_to_the_band_above():
    m = model(tree([0, 0, 0, 0, 0], [1, 3, -1, -1, -1], [2, 4, -1, -1, -1], [5, 9, 0, 0, 0], [4.0, 2.2, 0, 0, 0]),
              tree([0, 0, 0], [1, -1, -1], [2, -1, -1], [3, 0, 0], [4.0, 0, 0]),
              tree([0, 0, 0], [1, -1, -1], [2, -1, -1], [1, 0, 0], [7.5, 0, 0]))
    edges = sb.tree_split_points(m, pd.DataFrame({"a": [0.0, 3.0, 5.0, 9.0]}), ["a"], n=2)["a"]
    # gains: 4.0 -> 8, 2.2 -> 9, 7.5 -> 1 (dropped)
    assert len(edges) == 2 and edges[0] < 2.2 < edges[1] < 4.0
    # XGBoost sends x left when float32(x) < t: so x = t goes right, and so does 3.9999999, which rounds to 4
    bands = glm_lines._band(pd.Series([float(np.float32(2.2)), 2.2, 4.0, 3.9999999, 3.9999, 2.1999]), edges)
    assert bands.tolist() == ["q01", "q01", "q02", "q02", "q01", "q00"]


def test_tree_split_points_sum_gain_over_trees_and_skip_leaves_and_split_points_beyond_the_data():
    stump = lambda gain, cut: tree([0, 0, 0], [1, -1, -1], [2, -1, -1], [gain, 0, 0], [cut, 0.3, -0.3])  # noqa: E731
    x = pd.DataFrame({"a": [-1.0, 0.0, 1.0, 3.0, 5.0]})
    # 4.0 gains 5 in each of two trees, 10 in all, against 2.2's 9 in one
    edges = sb.tree_split_points(model(stump(5, 4.0), stump(5, 4.0), stump(9, 2.2)), x, ["a"], n=1)["a"]
    assert len(edges) == 1 and 3 < edges[0] < 4.0
    # the edge is just below the threshold: the smallest double that float32 rounds up to 4.0 goes right, as in XGBoost
    assert glm_lines._band(pd.Series([ex.double_cut(4.0)]), edges).tolist() == ["q01"]
    # a leaf's value (0.3, -0.3 here) is not a split point
    assert len(sb.tree_split_points(model(stump(5, 4.0), stump(9, 2.2)), x, ["a"], n=4)["a"]) == 2
    # a split point beyond every value of x would add an empty band: the next one is taken instead
    assert sb.tree_split_points(model(stump(9, 10.0), stump(5, 4.0)), x, ["a"], n=1)["a"][0] < 4.0


def test_value_edges_give_each_value_its_own_band():
    edges = sb.value_edges(pd.DataFrame({"v": [3.0, 1.0, 2.0, 2.0, np.nan]}), ["v"])["v"]
    assert glm_lines._band(pd.Series([0.0, 1.0, 2.0, 3.0, 9.0, np.nan]), edges).tolist() == \
        ["q00", "q00", "q01", "q02", "q02", "MISSING"]


def test_fidelity_weighs_by_exposure_and_signs_the_total():
    f = sb.fidelity(np.array([10.0, 33.0, 45.0]), np.array([10.0, 30.0, 50.0]), np.array([0.5, 1.0, 1.0]))
    assert f["mean squared log error"] == pytest.approx((np.log(1.1) ** 2 + np.log(0.9) ** 2) / 2.5, rel=1e-12)
    assert f["median relative error"] == pytest.approx(0.1) and f["largest relative error"] == pytest.approx(0.1)
    assert f["total difference"] == pytest.approx(88 / 90 - 1) and f["share within 20%"] == 1
    perfect = sb.fidelity(np.array([10.0, 30.0]), np.array([10.0, 30.0]), np.array([0.5, 1.0]))
    assert perfect["log rate variation explained"] == 1 and perfect["mean squared log error"] == 0


def synthetic(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"row_id": [f"R{i:05d}" for i in range(n)], "yr": 2025, "exposure": rng.uniform(0.2, 1.0, n),
                      "u": np.round(rng.gamma(2.0, 3.0, n), 1), "v": rng.integers(1, 6, n).astype(float),
                      "w": np.round(rng.normal(50, 10, n), 2), "cat": rng.choice(["a", "b", "c"], n)})
    d.loc[rng.random(n) < 0.05, "u"] = np.nan  # missing values reach a spline and an interaction band
    d.loc[rng.random(n) < 0.05, "v"] = np.nan
    return d


def test_the_surrogate_bands_numerics_with_up_to_20_values_and_keeps_spline_knots_inside_the_tails():
    d = synthetic(8, 3000).assign(k20=lambda f: (np.arange(len(f)) % 20).astype(float),
                                  k21=lambda f: (np.arange(len(f)) % 21).astype(float))
    y = (100 * d.exposure * np.exp(0.02 * d.w + 0.01 * d.k21)).to_numpy()
    glm, _ = sb.fit_surrogate(d, y, ["w", "k20", "k21"], ["cat"], [], {}, 4, 2.0)
    knots = glm.spec.splines["w"].bsplines_[0].t[3:-3]  # the knots between the end padding
    assert np.allclose(knots[[0, -1]], np.quantile(d.w, [0.01, 0.99]))
    assert "k20" not in glm.spec.splines and len(glm.spec.edges["k20"]) == 19 and "k21" in glm.spec.splines


def test_with_derived_rebuilds_a_derived_input_only_when_one_of_its_sources_moved():
    d = pd.DataFrame({"flood_zone": [np.nan, 2, 3, 5], "rainfall_mm": [1000.0, 1000, 1100, 689],
                      "soil_type": [None, "clay", "sand", "clay"], "dry_summer_index": [0.5, 0.4, 0.3, 0.2],
                      "x_fz3_rain": -1.0, "x_clay_dry": -1.0})
    for moved in ("flood_zone", "rainfall_mm"):  # a missing zone counts as 99, as in sql/11
        assert sb.with_derived(d, moved).x_fz3_rain.tolist() == [1.0, 0.0, 1.1, 0.689]
    for moved in ("soil_type", "dry_summer_index"):
        assert sb.with_derived(d, moved).x_clay_dry.tolist() == [0.0, 0.4, 0.0, 0.2]
    assert sb.with_derived(d, "elevation_m").equals(d) and sb.with_derived(d, "flood_zone").x_clay_dry.eq(-1).all()


def test_a_surrogate_rebuilt_from_its_exported_tables_matches_the_fitted_glm(tmp_path):
    d = synthetic(8, 6000)
    trees = 300 * d.exposure * np.exp(0.05 * d.u.fillna(4) + 0.3 * (d.v.fillna(3) > 3) * (d.u.fillna(4) > 6)
                                      + 0.01 * d.w + 0.2 * (d.cat == "b"))
    pairs, edges = [("u", "v"), ("u", "w")], {"u": np.array([2.0, 6.0]), "v": np.array([3.0]), "w": np.array([45.0, 55.0])}
    glm, bands = sb.fit_surrogate(d, trees.to_numpy(), ["u", "v", "w"], ["cat"], pairs, edges, 4, 1.5)
    assert "v" not in glm.spec.splines and glm.spec.edges["v"].tolist() == [1, 2, 3, 4]  # five values: one band each
    x = sb.with_interactions(d, bands, pairs)
    assert {"u*v", "u*w"} <= set(glm.levels) and x["u*v"].str.contains("MISSING").any()
    r = ex.export_glm("buildings", "surrogate", glm, x, x, 0.04, None, fitted_on="2025 buildings")
    rows = sb.band_rows(bands)
    for f, e in edges.items():  # every band and a MISSING row for each interaction feature
        assert [b["level"] for b in rows if b["term"] == f] == [f"q{i:02d}" for i in range(len(e) + 1)] + ["MISSING"]
    pd.DataFrame(r["terms"] + rows).to_csv(tmp_path / "terms.csv", index=False)
    pd.DataFrame(r["pieces"]).to_csv(tmp_path / "pieces.csv", index=False)
    pd.DataFrame(r["maps"]).to_csv(tmp_path / "maps.csv", index=False)
    t = sb.read_terms(tmp_path / "terms.csv")
    assert not t.note.str.contains("developed").any()
    pieces = pd.read_csv(tmp_path / "pieces.csv", float_precision="round_trip")
    maps = pd.read_csv(tmp_path / "maps.csv", dtype=str, keep_default_na=False)
    rebuilt = sb.add_interactions(t, d)  # the interaction inputs from the exported bands alone
    assert (rebuilt["u*v"] == x["u*v"]).all() and (rebuilt["u*w"] == x["u*w"]).all()
    model_pred = glm.predict(x)
    for route in ("lookup", "pieces"):
        assert np.abs(ex.evaluate_glm(t, pieces, maps, rebuilt, route) / model_pred - 1).max() < 1e-12
    on_edge = d.head(30).assign(u=edges["u"][1])  # a value on a band edge belongs to the band below it
    assert (sb.add_interactions(t, on_edge)["u*v"].str.startswith("q01|")).all()


def synthetic_export(tmp_path: Path, monkeypatch) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic trees and the export files the surrogate reads, in `tmp_path`, with the module pointed at them."""
    rng = np.random.default_rng(9)
    d = synthetic(9, 4000).assign(fine=lambda f: np.round(rng.uniform(0, 100, len(f)), 6))  # too fine for a grid
    claims = np.where(rng.random(len(d)) < 0.3, rng.gamma(2.0, 100 * d.exposure * np.exp(
        0.08 * d.u.fillna(4) + 0.5 * (d.v.fillna(3) > 3) * (d.u.fillna(4) > 6) + 0.01 * d.fine) / 0.6), 0.0)
    history = d.assign(yr=rng.integers(2021, 2025, len(d)))  # the trees are fitted on earlier years, as in C0
    s = rgl.XGBSmall(["u", "v", "w", "fine"], ["cat"], history, {"max_depth": 3, "eta": 0.3, "min_child_weight": 1.0},
                     30, {"seed": 3, "nthread": 1}, 1).fit(history, claims, 1.5)
    small = s.predict(d, ex.YEAR)
    r = ex.export_trees(tmp_path / "buildings_xgboost.json", s, d, small)
    r["features"].to_csv(tmp_path / "buildings_xgboost_features.csv", index=False)
    r["nodes"].to_csv(tmp_path / "buildings_xgboost_nodes.csv", index=False)
    pd.DataFrame({"line": "buildings", "segment": "", "applied": True, "note": "",
                  "item": ["XGBoost base rate (in the offset)", "small-claim trend slope (per year, log scale)",
                           "small-claim trend origin (year)"], "value": [s.base, s.slope, s.centre]}).to_csv(
        tmp_path / "scalars.csv", index=False)
    d.to_parquet(tmp_path / "rows_inputs_buildings.parquet", index=False)
    pd.DataFrame({"row_id": d.row_id, "small_claim": small.astype(float), "new_business": rng.random(len(d)) < 0.7,
                  "reference_policy": np.arange(len(d)) < 7}).to_parquet(tmp_path / "rows_components.parquet",
                                                                         index=False)
    for name, value in (("OUT", tmp_path), ("PREFIX", tmp_path / "buildings_surrogate"),
                        ("EVIDENCE", tmp_path / "ev.csv"), ("N_PAIRS", 1)):
        monkeypatch.setattr(sb, name, value)
    monkeypatch.setattr(common, "PACK_ROOT", tmp_path)
    return d, small.astype(float)


def test_a_whole_run_on_synthetic_trees_writes_tables_that_rebuild_the_surrogate(tmp_path, monkeypatch):
    d, small = synthetic_export(tmp_path, monkeypatch)
    for name, value in (("CANDIDATES", [(4, 2.0), (6, 2.0)]), ("MORE_PAIRS", (0, 2)), ("MORE_KNOTS", (8,))):
        monkeypatch.setattr(sb, name, value)
    rec = {}
    sb.run({}, rec)
    assert rec["checks_failed"] == []
    spec = pd.read_csv(tmp_path / "ev.csv", float_precision="round_trip")
    chosen = spec[spec.chosen].iloc[0]
    assert spec.chosen.sum() == 1 and chosen.candidate
    assert chosen["mean squared log error"] == spec[spec.candidate]["mean squared log error"].min()
    # the comparisons at the chosen knots and power, then more knots at the chosen power
    assert spec[~spec.candidate][["interactions", "knots", "tweedie_power"]].values.tolist() == [
        [0, chosen.knots, chosen.tweedie_power], [2, chosen.knots, chosen.tweedie_power], [1, 8, chosen.tweedie_power]]
    t = sb.read_terms(tmp_path / "buildings_surrogate_terms.csv")
    fine = t[(t.term == "fine") & (t.kind == "spline value")]
    assert len(fine) == 1 and fine.value.isna().all()  # a placeholder: the values seen stay in the row-level file
    assert (tmp_path / "buildings_surrogate_terms_values_seen.parquet").exists()
    assert not np.isin(d.fine.dropna(), t.value.dropna()).any()  # no data value in the committed table
    pieces = pd.read_csv(tmp_path / "buildings_surrogate_spline_pieces.csv", float_precision="round_trip")
    assert ((pieces.term == "w") & (pieces.note == "cubic piece")).sum() == chosen.knots - 1  # the final fit's knots
    maps = pd.read_csv(tmp_path / "buildings_surrogate_input_mappings.csv", dtype=str, keep_default_na=False)
    assert maps.note.str.contains("merged into OTHER").any() and not maps.note.str.contains("developed").any()
    out = pd.read_parquet(tmp_path / "rows_surrogate_buildings.parquet")
    assert out.surrogate.sum() == pytest.approx(small.sum(), rel=1e-12)  # the totals agree
    # the exported surrogate is the chosen candidate, at its knots and power, refitted on every row
    glm, _, xi, _, _ = sb.final_fit(d, small, sb.model_terms(d), int(chosen.knots), float(chosen.tweedie_power))
    assert np.array_equal(glm.predict(xi), out.surrogate.to_numpy())
    checks = pd.read_csv(tmp_path / "buildings_surrogate_reconciliation.csv", float_precision="round_trip")
    held = checks[checks.check.str.endswith("held out when choosing the specification: median relative error")]
    assert float(held.value.iloc[0]) == chosen["median relative error"]
    # one feature at a time, recomputed from the exported tables and the rescored trees
    one = checks[checks.check.str.startswith("every 2025 row at")]
    f = one.check.iloc[0].split()[4]
    one = one[one.check.str.startswith(f"every 2025 row at {f} = ")].value.astype(float).to_numpy()
    v = d[f].dropna().to_numpy(dtype=float)
    values = np.unique(v) if len(np.unique(v)) <= sb.MAX_BAND_VALUES else \
        np.unique(np.quantile(v, sb.PD_QUANTILES, method="inverted_cdf"))
    features = pd.read_csv(tmp_path / "buildings_xgboost_features.csv", keep_default_na=False, na_values=[""],
                           dtype={"level": str})
    trees = sb.tree_scorer(features, pd.read_csv(tmp_path / "scalars.csv", dtype=str, keep_default_na=False))
    maps = maps[maps.section == "categorical level"]
    gaps = [ex.evaluate_glm(t, pieces, maps, sb.add_interactions(t, d.assign(**{f: value})), "pieces").sum()
            / trees(d.assign(**{f: value})).sum() - 1 for value in values]
    assert len(one) == len(values) > 1 and np.allclose(one, gaps, rtol=0, atol=1e-9)


def test_a_whole_run_fails_its_check_when_the_rescored_trees_do_not_reproduce_small_claim(tmp_path, monkeypatch):
    synthetic_export(tmp_path, monkeypatch)
    for name, value in (("CANDIDATES", [(4, 2.0)]), ("MORE_PAIRS", ()), ("MORE_KNOTS", ())):
        monkeypatch.setattr(sb, name, value)
    sc = pd.read_csv(tmp_path / "scalars.csv")
    sc.loc[sc["item"] == "XGBoost base rate (in the offset)", "value"] *= 1.01
    sc.to_csv(tmp_path / "scalars.csv", index=False)
    rec = {}
    sb.run({}, rec)
    assert len(rec["checks_failed"]) == 1 and rec["checks_failed"][0].startswith("trees rescored")
