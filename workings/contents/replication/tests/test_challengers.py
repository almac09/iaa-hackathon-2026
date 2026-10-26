"""Unit tests for the A32 leaf refit (python/challengers.py), checked by hand on toy cases."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import challengers as ch  # noqa: E402


def test_gradient_and_hessian_match_the_tweedie_loss_numerically():
    y, f, eps = np.array([0.0, 1.0, 7.5]), np.array([-1.0, 0.2, 1.3]), 1e-5
    loss = lambda f_: -y * np.exp((1 - ch.P) * f_) / (1 - ch.P) + np.exp((2 - ch.P) * f_) / (2 - ch.P)  # noqa: E731
    g, h = ch.tweedie_grad_hess(y, f)
    assert g == pytest.approx((loss(f + eps) - loss(f - eps)) / (2 * eps), rel=1e-6)
    gp, _ = ch.tweedie_grad_hess(y, f + eps)
    gm, _ = ch.tweedie_grad_hess(y, f - eps)
    assert h == pytest.approx((gp - gm) / (2 * eps), rel=1e-6)


def test_one_leaf_newton_step_by_hand():
    # y = (3, 1) at margin 0: gradient (-2, 0), hessian (2, 1), so the Newton step is 2/3; with decay 0.5 from 0: 1/3.
    new = ch.refit_leaves(np.zeros((2, 1), int), [np.array([0.0])], np.zeros(2), np.array([3.0, 1.0]), 0.5, 1.0, 0.0)
    assert new[0][0] == pytest.approx(1 / 3)


def test_empty_leaves_keep_their_value_and_decay_one_changes_nothing():
    leaves = np.array([[0], [0], [0]])  # leaf 1 has no target rows
    values = [np.array([0.2, -0.7])]
    new = ch.refit_leaves(leaves, values, np.zeros(3), np.array([0.0, 2.0, 0.0]), 0.5, 0.05, 1.0)
    assert new[0][1] == -0.7 and new[0][0] != 0.2
    same = ch.refit_leaves(leaves, values, np.zeros(3), np.array([0.0, 2.0, 0.0]), 1.0, 0.05, 1.0)
    assert np.array_equal(same[0], values[0])


def test_trees_are_refitted_in_order_on_the_updated_margin():
    leaves, init, y = np.zeros((2, 2), int), np.zeros(2), np.array([3.0, 1.0])
    new = ch.refit_leaves(leaves, [np.array([0.0]), np.array([0.0])], init, y, 0.0, 1.0, 0.0)
    g, h = ch.tweedie_grad_hess(y, init + new[0][0])
    assert new[1][0] == pytest.approx(-g.sum() / h.sum())  # the second tree sees the first tree's refit
    assert ch.margin(leaves, new, init) == pytest.approx(np.full(2, new[0][0] + new[1][0]))
