"""Unit tests for the Week 2 market layer in python/emerging.py (A30): no competition data needed."""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402
import emerging as em  # noqa: E402
import run_glm_lines as rgl  # noqa: E402


@pytest.fixture
def cfg():
    return common.load_config()


def test_age_bands_match_the_emerging_cut_points():
    con = duckdb.connect()
    con.register("ages", pd.DataFrame({"a": [17, 24, 25, 34, 35, 49, 50, 64, 65, 90, None]}))
    got = con.execute(f"SELECT {em.AGE_BAND.format(x='a')} AS b FROM ages").df().b.tolist()
    assert got[:10] == ["17-24", "17-24", "25-34", "25-34", "35-49", "35-49", "50-64", "50-64", "65+", "65+"]
    assert got[10] is None or pd.isna(got[10])


def test_emerging_construction_codes_map_to_the_developed_ones():
    con = duckdb.connect()
    con.register("codes", pd.DataFrame({"construction_type": ["BRK", "stn", "WOD", "MIX", "other"]}))
    got = con.execute(f"SELECT {em.CONSTRUCTION} AS c FROM codes").df().c.tolist()
    assert got == ["brick", "stone", "timber", "non_standard", "other"]


def test_portable_features_drop_what_the_emerging_market_lacks_and_add_the_bands(cfg):
    p = em.portable(cfg)
    motor = p["glm_lines"]["models"]["motor"]
    assert "driver_age_band" in motor["categorical"] and "driver_age" not in motor["numeric"]
    assert not {"region", "credit_band"} & set(motor["categorical"])
    assert motor["excess"]["categorical"] == ["driver_age_band"]
    contents = p["glm_lines"]["models"]["contents"]
    assert contents["continuous"] == ["log_si"] and "policyholder_age_band" in contents["categorical"]
    assert not {"f_region", "f_credit"} & set(contents["categorical"])
    bld = p["lines"]["models"]["buildings"]
    assert "policyholder_age_band" in bld["categorical"] and not {"region", "occupation"} & set(bld["categorical"])
    for spec in (motor, contents, bld):
        assert not em.PORTABLE_DROP & set(rgl.resolve(p, spec.get("numeric", [])))
    assert "region" in cfg["glm_lines"]["models"]["motor"]["categorical"]  # the Week 1 config is untouched


def test_conforming_views_give_both_markets_the_same_columns(cfg, tmp_path, monkeypatch):
    dev, emg = tmp_path / "dev", tmp_path / "emg"
    dev.mkdir(), emg.mkdir()
    pd.DataFrame({"row_id": ["M1"], "driver_age": [30], "annual_mileage": [8000.0], "telematics_opt_in": [True],
                  "telematics_score": [0.5]}).to_parquet(dev / "policies_motor.parquet")
    pd.DataFrame({"row_id": ["M2"], "driver_age_band": ["25-34"]}).to_parquet(emg / "policies_motor.parquet")
    cols = {}
    for market, path in (("developed", dev), ("emerging", emg)):
        con = duckdb.connect()
        monkeypatch.setitem(em.CONFORM, market, {"policies_motor": em.CONFORM[market]["policies_motor"]})
        em.conform(con, {"paths": {"data_dir": str(path)}}, market)
        df = con.execute("SELECT * FROM policies_motor").df()
        cols[market] = set(df.columns)
        if market == "developed":
            assert df.driver_age_band.iloc[0] == "25-34"
    assert cols["developed"] == cols["emerging"]


def test_market_config_points_at_the_market(cfg):
    m = em.market_cfg(cfg, "emerging", external_pattern=True)
    assert m["paths"]["data_dir"] == "data/emerging" and m["sql_files"] == em.SQL_FILES
    assert m["sql_variables"]["use_external_dev_pattern"] is True
    assert cfg["paths"]["data_dir"] == "data"
    assert em.MARKETS["emerging"]["windows"]["final"] == ([2023, 2024], 2025)


def test_lines_default_to_the_developed_windows():
    line = rgl.Line("contents", pd.DataFrame({"src": [], "claim_cost_dev": [], "excess_dev": [], "exposure": []}),
                    pd.DataFrame(), {"glm_lines": {"models": {"contents": {}}}, "model": {"tweedie_power": 1.5}})
    assert line.windows is rgl.WINDOWS


def _sim(n=30_000, seed=4):
    import numpy as np
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"region": rng.choice(["E1", "E2", "E3"], n), "customer_id": rng.integers(0, n // 3, n),
                      "exposure": rng.uniform(0.3, 1.0, n)})
    base = 200 * d.exposure.to_numpy()  # the developed prediction
    true = base * 0.75 * d.region.map({"E1": 1.0, "E2": 1.3, "E3": 1.0}).to_numpy()  # level 0.75, E2 30% higher
    lam = true ** 0.5 / (30 * 0.5)
    k = rng.poisson(lam)
    y = np.where(k > 0, rng.gamma(np.maximum(k, 1e-9), 30 * 0.5 * true ** 0.5), 0.0)
    return d, y, base


def test_corrections_recover_the_level_and_a_regional_effect():
    import numpy as np
    d, y, base = _sim()
    m = em.Correction([], ["region"], 1e-6).fit(d, y, base)
    rows = pd.DataFrame({"region": ["E1", "E2"], "customer_id": [0, 0], "exposure": [1.0, 1.0]})
    pred = m.predict(rows, np.array([200.0, 200.0]))
    assert pred[0] / 200 == pytest.approx(0.75, rel=0.08)
    assert pred[1] / pred[0] == pytest.approx(1.3, rel=0.08)


def test_a_heavier_penalty_shrinks_towards_the_developed_relativities():
    import numpy as np
    d, y, base = _sim()
    rows = pd.DataFrame({"region": ["E1", "E2"], "customer_id": [0, 0], "exposure": [1.0, 1.0]})
    light = em.Correction([], ["region"], 1e-6).fit(d, y, base).predict(rows, np.array([1.0, 1.0]))
    heavy = em.Correction([], ["region"], 1.0).fit(d, y, base).predict(rows, np.array([1.0, 1.0]))
    assert abs(np.log(heavy[1] / heavy[0])) < abs(np.log(light[1] / light[0]))


def test_folds_keep_each_customer_in_one_fold():
    customers = [1, 2, 3, 1, 2, 3, 4, 5, 1]
    f = em.folds_by_customer(customers, k=3)
    assert len({f[i] for i, c in enumerate(customers) if c == 1}) == 1
    assert set(f) <= {0, 1, 2}


def test_choose_alpha_returns_a_grid_value():
    d, y, base = _sim(n=8_000)
    a, cv = em.choose_alpha(d, y, base, [], ["region"])
    assert a in em.ALPHAS and len(cv) == len(em.ALPHAS)


def test_fill_values_keep_each_fields_type():
    assert em.fill_value(pd.Series([True, False, True], dtype="boolean")) is True
    assert em.fill_value(pd.Series([1.0, 3.0, 10.0])) == 3.0
    assert em.fill_value(pd.Series(["a", "b", "b"])) == "b"
