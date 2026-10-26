"""Unit tests for seasonal exposure in python/run_glm_lines.py (D-22, A22)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import run_glm_lines as rgl  # noqa: E402


def make_line():
    # Two fitting-year rows (2021): one January only, one July only; claims twice as frequent in January.
    df = pd.DataFrame({"row_id": ["a", "b", "c"], "src": ["train", "train", "test"], "yr": [2021, 2021, 2025],
                       "exposure": [31 / 365, 31 / 365, 62 / 365], "claim_cost_dev": 0.0, "excess_dev": 0.0})
    cfg = {"glm_lines": {"models": {"contents": {}}}, "model": {"tweedie_power": 1.5}}
    line = rgl.Line("contents", df, pd.DataFrame(), cfg)
    days = np.zeros((3, 12))
    days[0, 0], days[1, 6], days[2, 0], days[2, 6] = 31, 31, 31, 31
    line.month_days = days
    line.claim_months = pd.DataFrame({"yr": [2021, 2021], "m": [1, 7], "n": [20, 10]})
    return line


def test_seasonal_exposure_keeps_fitting_total_and_weights_winter_up():
    line = make_line()
    line.use_window("tune")
    e = line.df.exposure.to_numpy()
    assert e[:2].sum() == pytest.approx(62 / 365)                  # fitting years keep their total
    assert e[0] / e[1] == pytest.approx(2.0)                       # January row carries twice July's exposure
    assert e[2] == pytest.approx(62 / 365)                         # a row covering both months is unchanged here
    assert (line.df.exposure_plain == [31 / 365, 31 / 365, 62 / 365]).all()  # reported rates keep plain exposure
    assert line.train.exposure.sum() == pytest.approx(62 / 365)    # train/test views refreshed


def test_lines_without_seasonal_exposure_are_untouched():
    line = make_line()
    line.month_days = None
    line.use_window("tune")
    assert (line.df.exposure == line.df.exposure_plain).all()


def test_offset_line_fits_reported_cost_and_the_share_weights_quarters():
    import development_tests
    line = make_line()
    off = development_tests.OffsetLine("contents", line.df, pd.DataFrame(
        {"row_id": ["a", "a", "b"], "net_cost": [100.0, 50.0, 70_000.0], "dev_factor": [2.0, 2.0, 1.0]}), line.cfg)
    d = off.df[off.df.src == "train"]
    assert off.reported_small(d, None).tolist() == [150.0, 70_000.0]          # reported, not developed
    assert off.reported_small(d, 50_000).tolist() == [150.0, 50_000.0]       # split on reported cost


def test_shape_adjust_keeps_the_total_and_moves_the_spread():
    mu = np.array([1.0, 2.0, 4.0, 8.0])
    expo = np.array([1.0, 1.0, 1.0, 0.05])
    assert np.allclose(rgl.shape_adjust(mu, expo), mu)                         # identity at power 1, factor 1
    wide = rgl.shape_adjust(mu, np.ones(4), power=1.2)
    assert wide.sum() == pytest.approx(mu.sum()) and wide[3] / wide[0] > mu[3] / mu[0]   # spread stretched
    short = rgl.shape_adjust(mu, expo, short_factor=0.5)
    assert short.sum() == pytest.approx(mu.sum())
    assert short[3] / short[2] == pytest.approx(0.5 * mu[3] / mu[2])          # only the short row is scaled


def test_fit_shape_recovers_a_planted_power():
    rng = np.random.default_rng(1)
    expo = np.ones(200_000)
    true_rate = np.exp(rng.normal(0, 0.6, 200_000))
    mu_flat = true_rate ** (1 / 1.15)                                            # predictions under-spread
    y = rng.gamma(2.0, true_rate / 2.0) * (rng.random(200_000) < 0.5) * 2
    a, f = rgl.fit_shape(y, mu_flat * true_rate.sum() / mu_flat.sum(), expo, 1.5, 0.1)
    assert 1.10 <= a <= 1.20 and f == 1.0                                      # no short rows: factor left at 1
