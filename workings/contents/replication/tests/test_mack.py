"""Unit tests for python/mack.py against Mack's (1993) published example (Taylor and Ashe's general insurance
triangle, also the GenIns data in R's ChainLadder package)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import mack  # noqa: E402

ROWS = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]


def triangle():
    t = np.full((10, 10), np.nan)
    for i, r in enumerate(ROWS):
        t[i, :len(r)] = r
    return t


def test_reserve_and_standard_error_match_mack_1993():
    cl = mack.chain_ladder(triangle())
    assert cl["reserve"].sum() == pytest.approx(18_680_856, rel=1e-6)
    assert np.sqrt(mack.mack_mse(cl, range(10))) == pytest.approx(2_447_095, rel=2e-3)
