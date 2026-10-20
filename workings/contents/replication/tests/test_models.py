"""Unit tests for python/models.py: the metric, the exposure handling and the design matrix."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import d2_tweedie_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import models  # noqa: E402

# Fixed example shared with tests/testthat/test_models.R, so both languages are checked against
# the same expected values.
Y = np.array([0.0, 0.0, 120.0, 0.0, 3500.0, 0.0, 800.0, 0.0])
PRED = np.array([50.0, 40.0, 90.0, 60.0, 400.0, 30.0, 300.0, 45.0])
EXPECTED_D2 = 0.34644972  # from sklearn.metrics.d2_tweedie_score(Y, PRED, power=1.5), scikit-learn 1.9.1


def test_d2_matches_sklearn_and_shared_constant():
    assert models.d2_tweedie(Y, PRED, 1.5) == pytest.approx(d2_tweedie_score(Y, PRED, power=1.5), abs=1e-12)
    assert models.d2_tweedie(Y, PRED, 1.5) == pytest.approx(EXPECTED_D2, abs=1e-8)


def test_deviance_scaling_justifies_exposure_weights():
    # d(c y, c mu) = c^(2-p) d(y, mu): fitting y/e with weight e^(2-p) equals using a log(e) offset.
    y, mu, c, p = np.array([0.0, 2.0, 7.5]), np.array([1.0, 1.5, 4.0]), 0.37, 1.5
    assert np.allclose(models.tweedie_deviance(c * y, c * mu, p), c ** (2 - p) * models.tweedie_deviance(y, mu, p))


def test_exposure_weighted_fit_recovers_rate():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.integers(0, 2, n).astype(float)
    expo = rng.uniform(0.1, 1.0, n)
    rate = np.exp(1.0 + 0.5 * x)
    y = rng.poisson(rate * expo).astype(float) * 10.0
    m = models.fit_tweedie(x[:, None], y, expo, 1.5, alpha=0.0)
    assert m.intercept_ == pytest.approx(1.0 + np.log(10.0), abs=0.05)
    assert m.coef_[0] == pytest.approx(0.5, abs=0.05)


def test_design_matrix_uses_code_point_order_reference():
    df = pd.DataFrame({"f": ["b", "MISSING", "a", "b"], "z": [0.1, 0.2, 0.3, 0.4]})
    levels = models.factor_levels(df, ["f"])
    assert levels["f"] == ["MISSING", "a", "b"]  # upper case sorts first, as in R's radix sort
    X, names = models.design_matrix(df, levels, ["z"])
    assert names == ["f=a", "f=b", "z"]
    assert X[:, 0].tolist() == [0, 0, 1, 0]
