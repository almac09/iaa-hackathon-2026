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
