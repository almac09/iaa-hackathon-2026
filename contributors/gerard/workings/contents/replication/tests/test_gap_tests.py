"""Unit tests for python/gap_tests.py (A21)."""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import gap_tests  # noqa: E402


def test_month_days_split_each_row_exactly():
    # Rows lie within one calendar year; their days by month must add up to the row's length, with none negative.
    rows = pd.DataFrame({"period_start": pd.to_datetime(["2024-01-01", "2024-02-15", "2025-12-31", "2024-03-01"]),
                         "period_end": pd.to_datetime(["2024-12-30", "2024-03-10", "2025-12-31", "2024-03-31"])})
    rows["yr"] = rows.period_start.dt.year
    con = duckdb.connect()
    con.register("rows_df", rows)
    d = con.execute(f"SELECT {gap_tests.MONTH_DAYS} FROM rows_df").df()
    days = (rows.period_end - rows.period_start).dt.days + 1
    assert (d.sum(axis=1).to_numpy() == days.to_numpy()).all()
    assert (d.to_numpy() >= 0).all()
    assert d.loc[1, "d2"] == 15 and d.loc[1, "d3"] == 10 and d.loc[2, "d12"] == 1
