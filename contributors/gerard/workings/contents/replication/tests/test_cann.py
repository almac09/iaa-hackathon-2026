"""Unit tests for python/cann.py and its PyTorch worker process (python/cann_torch.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import cann  # noqa: E402

SPEC = {"hidden": [16, 8], "dropout": 0.0, "learning_rate": 0.01, "batch_size": 2048, "max_epochs": 30, "patience": 5,
        "val_share": 0.1, "n_seeds": 2, "seed": 7}


def test_torch_side_checks():
    # Zero-initialised correction, loss minimised at the mean, unseen category coded as missing, and seeds trained
    # in one batch that cannot influence each other (run in the worker).
    check = cann.self_check()
    assert check["zero_init"] and check["loss_argmin"] == 1.0 and check["unseen_code"] == 0
    assert check["seeds_separable"]


def test_torch_never_loads_in_the_main_process():
    cann.self_check()
    assert "torch" not in sys.modules


def test_worker_stays_free_of_xgboost_when_the_caller_has_loaded_it():
    # A20: a multiprocessing child re-imported the calling script, loaded XGBoost's OpenMP runtime next to torch's,
    # and crashed; the run then waited forever. The worker must not inherit the caller's imports.
    import xgboost  # noqa: F401
    assert cann.self_check()["zero_init"]
    modules = cann.worker_modules()
    assert "torch" in modules and "xgboost" not in modules


def test_ensemble_learns_a_missed_effect_and_keeps_the_total():
    rng = np.random.default_rng(0)
    n = 20000
    d = pd.DataFrame({"x": rng.normal(size=n), "c": rng.choice(["a", "b"], n)})
    true = np.exp(0.5 * (d.c == "b").to_numpy())          # the offset model misses the category effect
    y = rng.poisson(true).astype(float)
    offset = np.full(n, y.mean())
    ens = cann.CANNEnsemble(["x"], ["c"], SPEC, 1.5).fit(d, y, offset)
    fitted = offset * np.exp(ens.correction(d))
    assert fitted.sum() == pytest.approx(offset.sum(), rel=1e-4)  # rebalanced to the offset total
    ratio = fitted[d.c == "b"].mean() / fitted[d.c == "a"].mean()
    assert ratio == pytest.approx(np.exp(0.5), rel=0.1)
