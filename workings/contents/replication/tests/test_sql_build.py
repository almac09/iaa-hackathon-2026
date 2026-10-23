"""Unit tests for sql/03_build_contents.sql on small hand-built cases (no competition data needed)."""
import copy
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402

SQL = (common.PACK_ROOT / "sql" / "03_build_contents.sql").read_text()


def policy(row_id, policy_id, start, end, exposure, sum_insured=30000.0):
    return dict(row_id=row_id, policy_id=policy_id, customer_id="00000001", address_id="A1",
                period_start=start, period_end=end, exposure=exposure, policyholder_age=40,
                sum_insured=sum_insured, tenure="owner", security_level="locks", high_value_items=1.0,
                accidental_damage_cover=False, works_from_home=False, lodgers=False, bicycle_cover=False,
                voluntary_excess=0, compulsory_excess=350, channel="direct", payment_method="annual",
                missed_payments_12m=0, new_business=False, renewal_count=2, days_quote_to_inception=5,
                discount_applied=0.0, written_premium=120.0, customer_service_contacts=0)


def claim(claim_id, policy_id, loss, incurred, recoveries=0.0, status="closed", line="contents", peril="theft"):
    return dict(claim_id=claim_id, policy_id=policy_id, line=line, peril=peril, loss_date=loss,
                report_date=loss, status=status, paid_to_date=incurred, incurred_to_date=incurred,
                recoveries=recoveries)


@pytest.fixture
def cfg():
    return common.load_config()


def build(cfg, extra_claims=()):
    train = pd.DataFrame([
        # Mid-term change: one policy, two rows. Exposure recorded in days, then on a 360-day basis.
        policy("R1", "P1", date(2023, 1, 1), date(2023, 6, 30), 181.0),
        policy("R2", "P1", date(2023, 7, 1), date(2023, 12, 31), 184 / 360),
        policy("R3", "P2", date(2023, 1, 1), date(2023, 12, 31), 1.0, sum_insured=21_400_000.0),
        policy("R4", "P3", date(2023, 1, 1), date(2023, 12, 31), 1.0, sum_insured=1_000_000.0),
        policy("R5", "P4", date(2023, 1, 1), date(2023, 12, 31), 1.0, sum_insured=999_999.0),
    ]).assign(renewal_decision="renewed")
    test = pd.DataFrame([policy("T1", "P2", date(2025, 1, 1), date(2025, 12, 31), 1.0, sum_insured=5_000_000.0)])
    claims = pd.DataFrame([
        claim("C1", "P1", date(2023, 8, 15), 1000.0, recoveries=200.0),          # -> R2, net 800
        claim("C2", "P1", date(2023, 3, 1), 500.0, status="repudiated"),         # excluded: declined
        claim("C3", "P1", date(2024, 2, 1), 700.0),                              # excluded: no row covers it
        claim("C4", "P1", date(2023, 3, 1), 900.0, line="motor"),                # excluded: other line
        claim("C5", "P2", date(2023, 5, 5), 20000.0, peril="flood"),             # -> R3, size band 4
        *extra_claims,
    ])
    con = duckdb.connect()
    for name, df in {"policies_contents": train, "test_contents": test, "claims": claims,
                     "customers": pd.DataFrame({"customer_id": ["00000001"], "date_of_birth": [date(1983, 1, 1)]}),
                     "addresses": pd.DataFrame({"address_id": ["A1"], "region": ["R01"], "elevation_m": [10.0],
                                                "year_built": [1950], "energy_rating": ["C"]}),
                     "weather": pd.DataFrame({"address_id": ["A1", "A1"], "year": [2023, 2025],
                                              "rainfall_mm": [800.0, 810.0]})}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM {name}_df")
    common.set_sql_variables(con, cfg)
    con.execute(SQL)
    return con.execute("SELECT * FROM contents_model").df().set_index("row_id")


def test_exposure_recomputed_from_dates(cfg):
    m = build(cfg)
    assert m.loc["R1", "exposure"] == pytest.approx(181 / 365)
    assert m.loc["R2", "exposure"] == pytest.approx(184 / 365)
    assert m.loc["R1", "exposure_raw"] == 181.0  # raw value kept for before/after


def test_sum_insured_rescaled_at_or_above_threshold_only(cfg):
    m = build(cfg)
    assert m.loc["R3", "sum_insured"] == 21_400
    assert m.loc["R4", "sum_insured"] == 1_000      # threshold is inclusive
    assert m.loc["R5", "sum_insured"] == 999_999    # just below: unchanged
    assert m.loc["T1", "sum_insured"] == 5_000      # test rows are fixed too
    assert m.loc["R3", "sum_insured_raw"] == 21_400_000


def test_claims_matched_by_loss_date_and_filtered(cfg):
    m = build(cfg)
    assert m.loc["R2", "claim_cost"] == pytest.approx(800.0)   # recoveries deducted, right mid-term row
    assert m.loc["R1", "claim_cost"] == 0                       # repudiated and motor claims excluded
    assert m.loc["R3", "cost_band_4"] == pytest.approx(20000.0)
    assert m.loc["R3", "cost_flood"] == pytest.approx(20000.0)
    assert m["n_claims"].sum() == 2                             # C3 (no covering row) dropped
    assert m.loc["T1", "n_claims"] == 0                         # claims never attach to test rows


def test_decision_switches_off_keep_raw_values(cfg):
    off = copy.deepcopy(cfg)
    off["decisions"]["D01_recompute_exposure"]["apply"] = False
    off["decisions"]["D02_rescale_sum_insured"]["apply"] = False
    m = build(off)
    assert m.loc["R1", "exposure"] == 181.0
    assert m.loc["R3", "sum_insured"] == 21_400_000


def test_duplicate_claims_counted_once(cfg):
    # C6 repeats C1 on every field except claim_id (A18): kept once while D-19 applies, twice when switched off.
    dup = [claim("C6", "P1", date(2023, 8, 15), 1000.0, recoveries=200.0)]
    assert build(cfg, dup).loc["R2", "claim_cost"] == pytest.approx(800.0)
    off = copy.deepcopy(cfg)
    off["decisions"]["D19_dedupe_claims"]["apply"] = False
    assert build(off, dup).loc["R2", "claim_cost"] == pytest.approx(1600.0)


def clean_addresses(cfg, apply=True):
    """addresses_clean (A28) on a hand-built address table."""
    sql = SQL[SQL.index("CREATE OR REPLACE VIEW addresses_clean"):]
    sql = sql[:sql.index(";") + 1]
    con = duckdb.connect()
    con.register("a_df", pd.DataFrame({
        "address_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "region": ["R11", "R01", "R01", "R01", "R01", "R02"],
        "elevation_m": [32.8, 10.0, 12.0, 14.0, 16.0, 20.0],
        "year_built": [1990, 1900, 1950, 1970, 1900, 2000],
        "energy_rating": ["C", "C", "C", "C", None, None]}))
    con.execute("CREATE VIEW addresses AS SELECT * FROM a_df")
    c = copy.deepcopy(cfg)
    c["decisions"]["D26_r11_elevation_feet"]["apply"] = apply
    c["decisions"]["D27_year_built_placeholder"]["apply"] = apply
    common.set_sql_variables(con, c)
    con.execute(sql)
    return con.execute("SELECT * FROM addresses_clean ORDER BY address_id").df().set_index("address_id")


def test_r11_elevation_converted_from_feet_and_raw_kept(cfg):
    a = clean_addresses(cfg)
    assert a.loc["A1", "elevation_m"] == pytest.approx(10.0, abs=0.01)    # 32.8 ft = 10.0 m
    assert a.loc["A1", "elevation_m_raw"] == 32.8
    assert a.loc["A2", "elevation_m"] == 10.0                              # other regions untouched


def test_year_built_placeholder_imputed_by_energy_rating(cfg):
    a = clean_addresses(cfg)
    assert a.loc["A2", "year_built"] == 1970        # typical rating-C year: mean of 1990, 1950, 1970
    assert a.loc["A5", "year_built"] == 2000        # missing rating matched to missing rating
    assert a.loc["A2", "year_built_raw"] == 1900
    assert a.loc["A3", "year_built"] == 1950        # genuine years untouched


def test_address_corrections_switch_off(cfg):
    a = clean_addresses(cfg, apply=False)
    assert a.loc["A1", "elevation_m"] == 32.8 and a.loc["A2", "year_built"] == 1900
