"""Unit tests for python/xgb_models.py: the exposure offset, the feature guard and the bootstrap."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import models  # noqa: E402
import xgb_models  # noqa: E402

PARAMS = {"max_depth": 1, "eta": 0.3, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1.0,
          "reg_lambda": 0.0}


def test_offset_recovers_rate_and_scales_with_exposure():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.integers(0, 2, n)
    expo = rng.uniform(0.1, 1.0, n)
    y = rng.poisson(np.exp(1.0 + 0.5 * x) * expo).astype(float) * 10.0
    X = pd.DataFrame({"x": x.astype(float)})
    base = y.sum() / expo.sum()
    b = xgb_models.fit(X, y, expo, base, 1.5, PARAMS, n_rounds=200, seed=0, nthread=1)

    grid = pd.DataFrame({"x": [0.0, 1.0]})
    rate = xgb_models.predict(b, grid, np.ones(2), base)
    assert rate == pytest.approx(10 * np.exp([1.0, 1.5]), rel=0.03)
    # The offset is additive on the log scale, so halving exposure halves the prediction exactly.
    assert xgb_models.predict(b, grid, np.full(2, 0.5), base) == pytest.approx(rate / 2, rel=1e-6)


def test_matrix_uses_training_categories_and_keeps_numeric_missing():
    train = pd.DataFrame({"c": ["b", "a", None], "z": [1.0, None, 3.0]})
    score = pd.DataFrame({"c": ["a", "new"], "z": [None, 2.0]})
    cats = xgb_models.category_levels(train, ["c"])
    assert cats["c"] == ["MISSING", "a", "b"]  # missing is a level (D-06), code-point order as in models.py
    X = xgb_models.matrix(score, ["z"], cats)
    assert X.c.cat.categories.tolist() == ["MISSING", "a", "b"]
    assert X.c.isna().tolist() == [False, True]  # an unseen level is treated as missing, not as a new code
    assert X.z.isna().tolist() == [True, False]  # numeric missing is left to XGBoost's default direction


def test_feature_guard_rejects_excluded_and_post_period_columns():
    cfg = {"decisions": {"D04_exclude_contacts": {"apply": True}, "D05_exclude_written_premium": {"apply": True}}}
    xgb_models.check_features(["sum_insured", "region"], cfg)
    for bad in ("customer_service_contacts", "written_premium", "earned_premium", "renewal_decision", "claim_cost"):
        with pytest.raises(ValueError, match=bad):
            xgb_models.check_features(["sum_insured", bad], cfg)
    cfg["decisions"]["D04_exclude_contacts"]["apply"] = False
    xgb_models.check_features(["customer_service_contacts"], cfg)


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.gamma(0.2, 500, 5000)
    good = 0.7 * y + 0.3 * y.mean()
    flat = np.full_like(y, y.mean())
    assert np.all(xgb_models.paired_bootstrap_d2(y, flat, flat, 1.5, n_boot=50, seed=3) == 0)
    diff = xgb_models.paired_bootstrap_d2(y, good, flat, 1.5, n_boot=50, seed=3)
    assert np.all(diff > 0)
    assert np.array_equal(diff, xgb_models.paired_bootstrap_d2(y, good, flat, 1.5, n_boot=50, seed=3))


def test_trend_backfitting_recovers_known_yearly_trend():
    rng = np.random.default_rng(2)
    n = 40000
    x = rng.integers(0, 2, n)
    yr = rng.integers(2021, 2025, n)
    expo = rng.uniform(0.2, 1.0, n)
    y = rng.poisson(np.exp(1.0 + 0.5 * x + 0.06 * (yr - 2022.5)) * expo).astype(float) * 10.0
    X = pd.DataFrame({"x": x.astype(float)})  # no year feature: the trend can only enter through the offset
    b, slope, centre, history = xgb_models.fit_with_trend(X, y, expo, yr, y.sum() / expo.sum(), 1.5, PARAMS,
                                                          n_rounds=100, seed=0, nthread=1, n_iter=3)
    assert slope == pytest.approx(0.06, abs=0.01)
    assert abs(history[-1]["residual_slope"]) < 0.002  # converged: the final fit leaves no trend in its A/E
    # Projecting a year ahead raises the prediction by exp(slope) exactly, whatever the row.
    grid = pd.DataFrame({"x": [0.0, 1.0]})
    base = y.sum() / expo.sum()
    p24 = xgb_models.predict(b, grid, np.ones(2), base, extra=slope * (2024 - centre) * np.ones(2))
    p25 = xgb_models.predict(b, grid, np.ones(2), base, extra=slope * (2025 - centre) * np.ones(2))
    assert p25 / p24 == pytest.approx(np.exp(slope) * np.ones(2), rel=1e-6)


def test_cluster_bootstrap_matches_unweighted_d2_and_detects_a_real_gain():
    import significance  # noqa: E402  (python/ is on the path above)
    rng = np.random.default_rng(4)
    n = 6000
    y = rng.gamma(0.3, 300, n)
    good, flat = 0.6 * y + 0.4 * y.mean(), np.full(n, y.mean())
    clusters = rng.integers(0, 2000, n)
    one = np.ones((1, n))
    assert significance._weighted_d2(y, good, one, 1.5)[0] == pytest.approx(models.d2_tweedie(y, good, 1.5))
    diff = significance.paired_cluster_bootstrap(y, good, flat, clusters, 1.5, n_boot=200, seed=1)
    s = significance.summarise(diff, models.d2_tweedie(y, good, 1.5) - models.d2_tweedie(y, flat, 1.5))
    assert s["significant_95"] and s["share_not_better"] == 0
    same = significance.paired_cluster_bootstrap(y, flat, flat, clusters, 1.5, n_boot=50, seed=1)
    assert np.allclose(same, 0)
