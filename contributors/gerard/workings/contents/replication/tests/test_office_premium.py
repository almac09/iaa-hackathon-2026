"""Unit tests for the office premium build-up (A31)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import office_premium as op  # noqa: E402

EXPENSES = pd.DataFrame({
    "item": ["fixed_expense_per_policy", "variable_expense", "commission_direct", "commission_comparison_site",
             "commission_broker", "premium_tax", "cat_reinsurance", "target_profit_margin"],
    "developed": [45.0, 0.12, 0.0, 0.10, 0.15, 0.12, 0.06, 0.05],
    "emerging": [85.0, 0.15, 0.0, np.nan, 0.20, 0.08, 0.12, 0.08],
})


def test_broker_property_policy_by_hand():
    r = op.rates(EXPENSES, "emerging")
    b = op.build_up([500.0], [1.0], np.array(["buildings"]), np.array(["broker"]), r).iloc[0]
    # (500 x 1.12 + 85) / (1 - 0.15 - 0.20 - 0.08 - 0.08) = 645 / 0.49
    assert b.office_premium == pytest.approx(645 / 0.49)
    parts = b.risk_premium_annual + b.cat_reinsurance + b.fixed_expense + b.variable_expense + b.commission \
        + b.premium_tax + b.profit_margin
    assert parts == pytest.approx(b.office_premium)  # the components add back to the premium


def test_motor_has_no_cat_load_and_part_year_cover_is_annualised():
    r = op.rates(EXPENSES, "emerging")
    b = op.build_up([250.0], [0.5], np.array(["motor"]), np.array(["direct"]), r).iloc[0]
    assert b.cat_reinsurance == 0 and b.risk_premium_annual == pytest.approx(500.0)
    assert b.office_premium == pytest.approx((500 + 85) / (1 - 0.15 - 0.0 - 0.08 - 0.08))


def test_missing_comparison_site_commission_falls_back_to_developed():
    assert op.rates(EXPENSES, "emerging")["commission_comparison_site"] == pytest.approx(0.10)
    assert op.rates(EXPENSES, "developed")["commission_comparison_site"] == pytest.approx(0.10)


def test_bad_inputs_are_refused():
    r = op.rates(EXPENSES, "emerging")
    with pytest.raises(ValueError, match="exposure"):
        op.build_up([1.0], [0.0], np.array(["motor"]), np.array(["direct"]), r)
    with pytest.raises(ValueError, match="channel"):
        op.build_up([1.0], [1.0], np.array(["motor"]), np.array(["telephone"]), r)
