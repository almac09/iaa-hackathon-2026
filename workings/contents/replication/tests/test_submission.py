"""Unit tests for python/make_submission.py: row matching and the checks on predictions."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import make_submission  # noqa: E402

SAMPLE = pd.DataFrame({"row_id": ["M1", "C2", "B3", "C1"], "prediction": 0.0})


def test_contents_rows_mapped_in_sample_order_and_others_filled():
    out = make_submission.build_submission(SAMPLE, pd.Series([5.0, 7.0], index=["C1", "C2"]), "C", 0.1)
    assert out.row_id.tolist() == ["M1", "C2", "B3", "C1"]
    assert out.prediction.tolist() == [0.1, 7.0, 0.1, 5.0]


@pytest.mark.parametrize("contents", [
    pd.Series([5.0], index=["C1"]),                     # a contents row is missing
    pd.Series([5.0, 7.0, 1.0], index=["C1", "C2", "C9"]),  # an extra row
    pd.Series([5.0, 7.0, 1.0], index=["C1", "C2", "C2"]),  # a duplicate row
])
def test_rejects_rows_that_do_not_match_the_sample(contents):
    with pytest.raises(ValueError, match="row_ids"):
        make_submission.build_submission(SAMPLE, contents, "C", 0.1)


def test_rejects_zero_predictions():
    with pytest.raises(ValueError, match="strictly positive"):
        make_submission.build_submission(SAMPLE, pd.Series([0.0, 7.0], index=["C1", "C2"]), "C", 0.1)
