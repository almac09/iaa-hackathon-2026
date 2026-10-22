"""Unit tests for python/glm_lines.py and the no-look-ahead rule of sql/12."""
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402
import glm_lines  # noqa: E402


def test_split_at_threshold_on_reported_cost_then_develop():
    claims = pd.DataFrame({"row_id": ["R1", "R1", "R2"], "net_cost": [80_000.0, 10_000.0, 40_000.0],
                           "dev_factor": [1.0, 2.0, 3.0]})
    t = glm_lines.split_targets(claims, 50_000)
    assert t.loc["R1"].tolist() == [50_000 + 20_000, 30_000, 1]
    # 40k is under the threshold reported; developed x3 it would be 120k, but it stays small.
    assert t.loc["R2"].tolist() == [120_000, 0, 0]


def test_gpd_tail_recovers_known_shape():
    rng = np.random.default_rng(0)
    x = 50_000 + rng.pareto(1 / 0.3, 20000) * 20_000 / (1 / 0.3)  # GPD with shape 0.3 above 50k
    tail = glm_lines.gpd_tail(pd.DataFrame({"net_cost": np.r_[x, [1_000.0] * 100]}), 50_000)
    assert tail["n_large"] == 20000
    assert tail["shape"] == pytest.approx(0.3, abs=0.03)
    assert tail["adjustment"] == pytest.approx(1.0, abs=0.05)


def test_factors_band_numerics_and_merge_rare_levels():
    train = pd.DataFrame({"x": np.arange(100, dtype=float), "c": ["a"] * 99 + ["rare"], "exposure": 1.0})
    spec = glm_lines.FactorSpec(train, ["x"], ["c"], min_level_share=0.02)
    f = spec.factors(pd.DataFrame({"x": [0.0, 99.0, np.nan], "c": ["a", "rare", None]}))
    assert f.x.tolist() == ["q00", "q09", "MISSING"]
    assert f.c.tolist() == ["a", "OTHER", "OTHER"]


def test_smoothed_numeric_is_a_flat_extrapolating_curve_with_missing_flag():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": np.r_[rng.uniform(0, 100, 999), np.nan], "c": "a", "exposure": 1.0})
    spec = glm_lines.FactorSpec(train, ["x"], ["c"], smooth=["x"], n_knots=4)
    assert spec.numeric == []  # smoothed, so not banded
    b = spec.basis(pd.DataFrame({"x": [50.0, 100.0, 1000.0, np.nan]}))
    assert list(b.columns) == ["x_s0", "x_s1", "x_s2", "x_s3", "x_s4", "x_missing"]  # 4 knots, cubic, bias dropped
    assert np.allclose(b.iloc[1, :5], b.iloc[2, :5])   # flat beyond the training range
    assert b.iloc[3].tolist() == [0, 0, 0, 0, 0, 1]
    assert glm_lines.FactorSpec(train.assign(x=np.r_[[0.0, 1.0] * 499, 0.0, np.nan]), ["x"], ["c"],
                                smooth=["x"]).numeric == ["x"]  # two distinct values: banded instead


def test_credibility_factor():
    n, e = np.array([0, 2, 0, np.nan]), np.array([0.0, 0.5, 1.0, np.nan])
    assert glm_lines.credibility_factor(n, e, 1.0).tolist() == [1.0, 3 / 1.5, 1 / 2, 1.0]
    assert glm_lines.credibility_factor(n, e, np.inf).tolist() == [1.0] * 4  # group switched off


def test_history_sees_only_claims_reported_before_the_period():
    cfg = common.load_config()
    rows = lambda line, ids, cust, starts: pd.DataFrame({  # noqa: E731
        "line": line, "src": "train", "row_id": ids, "customer_id": cust, "address_id": "A1",
        "period_start": starts, "period_end": [date(s.year, 12, 31) for s in starts], "exposure": 1.0})
    contents = rows("contents", ["C21", "C22", "C23"], "K1", [date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)])
    buildings = rows("buildings", ["B21"], "K1", [date(2021, 1, 1)])
    motor = rows("motor", ["M22"], "K1", [date(2022, 1, 1)])
    claims = pd.DataFrame({
        "line": ["contents", "contents", "buildings", "contents"],
        "row_id": ["C21", "C21", "B21", "C22"],
        "loss_date": [date(2021, 3, 1), date(2021, 11, 1), date(2021, 6, 1), date(2022, 2, 1)],
        # the November 2021 contents claim is only reported in February 2022, after C22 starts
        "report_date": [date(2021, 3, 5), date(2022, 2, 1), date(2021, 6, 9), date(2022, 2, 3)],
        "net_cost": [900.0, 7000.0, 3000.0, 500.0]})
    con = duckdb.connect()
    for name, df in {"contents_rows": contents, "buildings_rows": buildings, "motor_rows": motor,
                     "line_claims": claims,
                     "line_target": pd.DataFrame({"line": ["contents"], "row_id": ["C21"], "n_claims": [2]})}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    common.set_sql_variables(con, {**cfg, "sql_variables": {**cfg["sql_variables"], "hist_large_claim": 5000}})
    con.execute((common.PACK_ROOT / "sql" / "12_claim_history.sql").read_text())
    h = con.execute("SELECT * FROM claim_history").df().set_index("row_id")
    assert h.loc["C21", ["h_own_n", "h_own_expo", "h_home_n"]].tolist() == [0, 0, 0]  # nothing before 2021
    assert h.loc["C22", "h_own_n"] == 1          # the late-reported claim is not yet known
    assert h.loc["C22", "h_own_nlarge"] == 0
    assert h.loc["C22", "h_home_n"] == 1         # the 2021 buildings claim at the same address
    assert h.loc["C23", ["h_own_n", "h_own_n1y", "h_own_nlarge", "h_own_expo"]].tolist() == [3, 1, 1, 2.0]
    assert h.loc["M22", "h_cross_n"] == 2        # both home claims known by 2022 for the same customer
    assert h.loc["M22", "h_own_n"] == 0

    mu = pd.DataFrame({"line": ["contents"] * 3 + ["buildings", "motor"], "row_id": ["C21", "C22", "C23", "B21", "M22"],
                       "mu": [0.1, 0.2, 0.3, 0.4, 0.5]})
    con.register("freq_pred_df", mu)
    con.execute("CREATE TABLE freq_pred AS SELECT * FROM freq_pred_df")
    con.execute((common.PACK_ROOT / "sql" / "13_history_expected.sql").read_text())
    e = con.execute("SELECT * FROM history_expected").df().set_index("row_id")
    assert e.loc["C21", ["e_own", "e_home", "e_cross"]].tolist() == [0, 0, 0]
    assert e.loc["C23", "e_own"] == pytest.approx(0.3)      # C21 + C22, never the row itself
    assert e.loc["C23", "e_home"] == pytest.approx(0.4)     # B21 at the same address
    assert e.loc["C23", "e_cross"] == pytest.approx(0.5)    # M22 for the same customer
    assert e.loc["M22", "e_cross"] == pytest.approx(0.5)    # C21 + B21
