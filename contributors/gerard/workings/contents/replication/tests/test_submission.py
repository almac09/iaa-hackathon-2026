"""Unit tests for python/make_submission.py: row matching and the checks on predictions."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import make_submission  # noqa: E402

SAMPLE = pd.DataFrame({"row_id": ["M1", "C2", "B3", "C1"], "prediction": 0.0})


def test_contents_rows_mapped_in_sample_order_and_others_filled():
    out = make_submission.build_submission(SAMPLE, {"C": pd.Series([5.0, 7.0], index=["C1", "C2"])}, 0.1)
    assert out.row_id.tolist() == ["M1", "C2", "B3", "C1"]
    assert out.prediction.tolist() == [0.1, 7.0, 0.1, 5.0]


@pytest.mark.parametrize("contents", [
    pd.Series([5.0], index=["C1"]),                     # a contents row is missing
    pd.Series([5.0, 7.0, 1.0], index=["C1", "C2", "C9"]),  # an extra row
    pd.Series([5.0, 7.0, 1.0], index=["C1", "C2", "C2"]),  # a duplicate row
])
def test_rejects_rows_that_do_not_match_the_sample(contents):
    with pytest.raises(ValueError, match="row_ids"):
        make_submission.build_submission(SAMPLE, {"C": contents}, 0.1)


def test_rejects_zero_predictions():
    with pytest.raises(ValueError, match="strictly positive"):
        make_submission.build_submission(SAMPLE, {"C": pd.Series([0.0, 7.0], index=["C1", "C2"])}, 0.1)


def test_every_line_from_its_own_predictions():
    preds = {"C": pd.Series([5.0, 7.0], index=["C1", "C2"]), "M": pd.Series([3.0], index=["M1"]),
             "B": pd.Series([9.0], index=["B3"])}
    out = make_submission.build_submission(SAMPLE, preds)
    assert out.prediction.tolist() == [3.0, 7.0, 9.0, 5.0]


def test_rows_of_a_line_without_predictions_need_a_fill_value():
    with pytest.raises(ValueError, match="no prediction"):
        make_submission.build_submission(SAMPLE, {"C": pd.Series([5.0, 7.0], index=["C1", "C2"])})


def test_level_variant_rescales_one_line_only():
    import level_variant
    sub = pd.DataFrame({"row_id": ["M1", "M2", "B1"], "claim_cost": [100.0, 300.0, 50.0]})
    out, factor = level_variant.rescale(sub, "M", 100.0)
    assert factor == pytest.approx(0.5)
    assert out.claim_cost.tolist() == [50.0, 150.0, 50.0]
    with pytest.raises(ValueError):
        level_variant.rescale(sub, "C", 1.0)
