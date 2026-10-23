"""Unit tests for sql/09 and sql/10 on small hand-built cases (no competition data needed)."""
import copy
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402

SQL = [(common.PACK_ROOT / "sql" / f).read_text() for f in ("09_build_lines.sql", "10_development.sql")]


def row(row_id, policy_id, start, end, **extra):
    return dict(row_id=row_id, policy_id=policy_id, period_start=start, period_end=end, exposure=1.0, **extra)


def claim(claim_id, policy_id, line, peril, loss, lag, incurred, recoveries=0.0, status="closed"):
    return dict(claim_id=claim_id, policy_id=policy_id, line=line, peril=peril, loss_date=loss,
                report_date=loss + pd.Timedelta(days=lag), status=status, incurred_to_date=incurred,
                recoveries=recoveries)


@pytest.fixture
def cfg():
    return common.load_config()


def build(cfg, extra_motor=(), extra_claims=()):
    y = lambda yr: (date(yr, 1, 1), date(yr, 12, 31))  # noqa: E731
    buildings = pd.DataFrame([
        row("B1", "PB1", *y(2024), sum_insured=1_061_000.0, construction_type="BRICK "),
        row("B2", "PB2", *y(2024), sum_insured=52_000_000.0, construction_type="Brick"),
    ]).assign(renewal_decision="renewed")
    motor = pd.DataFrame([row("M1", "PM1", *y(2021)), row("M2", "PM1", *y(2024)), row("M3", "PM2", *y(2021)),
                          row("M4", "PM2", *y(2024)), *extra_motor]).assign(renewal_decision="renewed",
                                                                            customer_id="4521")
    claims = pd.DataFrame([
        # Own damage: two old claims at lags 1 and 10, and a 2024 claim aged 5 days, reported at lag 1.
        claim("C1", "PM1", "motor", "own_damage", pd.Timestamp(2021, 1, 1), 1, 100.0),
        claim("C2", "PM1", "motor", "own_damage", pd.Timestamp(2021, 1, 1), 10, 330.0, recoveries=30.0),
        claim("C3", "PM1", "motor", "own_damage", pd.Timestamp(2024, 12, 26), 1, 50.0),
        claim("C4", "PM1", "motor", "own_damage", pd.Timestamp(2024, 6, 1), 1, 999.0, status="repudiated"),
        # Bodily injury reported within 90 days: 2021 cohort 2,000, 2024 cohort 1,000.
        claim("C5", "PM2", "motor", "bodily_injury", pd.Timestamp(2021, 3, 1), 5, 2000.0),
        claim("C6", "PM2", "motor", "bodily_injury", pd.Timestamp(2024, 3, 1), 5, 1000.0, status="open"),
        claim("C7", "PB1", "buildings", "fire", pd.Timestamp(2024, 3, 1), 5, 700.0),
        claim("C8", "PB1", "buildings", "fire", pd.Timestamp(2023, 3, 1), 5, 700.0),  # no row covers 2023
        *extra_claims,
    ])
    empty_rows = pd.DataFrame({"src": pd.Series(dtype=str), "row_id": pd.Series(dtype=str),
                               "policy_id": pd.Series(dtype=str), "period_start": pd.Series(dtype="datetime64[us]"),
                               "period_end": pd.Series(dtype="datetime64[us]"), "yr": pd.Series(dtype=int),
                               "exposure": pd.Series(dtype=float)})
    empty_claims = pd.DataFrame({"claim_id": pd.Series(dtype=str), "row_id": pd.Series(dtype=str),
                                 "net_cost": pd.Series(dtype=float)})
    con = duckdb.connect()
    no_test_rows = lambda d: d.head(0).drop(columns="renewal_decision")  # noqa: E731
    for name, df in {"policies_buildings": buildings, "test_buildings": no_test_rows(buildings),
                     "policies_motor": motor, "test_motor": no_test_rows(motor),
                     "claims": claims, "claims_clean": claims, "contents_rows": empty_rows,
                     "contents_claims": empty_claims,
                     "customers": pd.DataFrame({"customer_id": ["00004521"]})}.items():
        con.register(f"{name}_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    common.set_sql_variables(con, cfg)
    for s in SQL:
        con.execute(s)
    return con


def test_buildings_sum_insured_threshold_and_casing(cfg):
    b = build(cfg).execute("SELECT * FROM buildings_rows").df().set_index("row_id")
    assert b.loc["B1", "sum_insured"] == 1_061_000      # genuine 1.06m risk below the 10m threshold: kept
    assert b.loc["B2", "sum_insured"] == 52_000         # x1000 error divided back
    assert b.loc["B2", "sum_insured_raw"] == 52_000_000
    assert b.construction_type.tolist() == ["brick", "brick"]
    assert b.loc["B1", "construction_type_raw"] == "BRICK "


def test_timber_codes_t_and_w_merged_and_switchable(cfg):
    # D-24: T (2021-22) and W (2023 onwards) are the same timber category recoded.
    con = duckdb.connect()
    con.register("rows_df", pd.DataFrame({"construction_type": ["T", " w", "Brick"]}))
    sql = SQL[0][SQL[0].index("CASE WHEN getvariable('d24_merge_timber')"):SQL[0].index("AS construction_type\nFROM")]
    for apply, expected in ((True, ["timber", "timber", "brick"]), (False, ["t", "w", "brick"])):
        on = copy.deepcopy(cfg)
        on["decisions"]["D24_merge_timber"]["apply"] = apply
        common.set_sql_variables(con, on)
        assert con.execute(f"SELECT {sql} FROM rows_df").df().iloc[:, 0].tolist() == expected


def test_motor_customer_id_padded_to_customers_format(cfg):
    m = build(cfg).execute("SELECT * FROM motor_rows").df()
    assert set(m.customer_id) == {"00004521"} and set(m.customer_id_raw) == {"4521"}
    pad = build(cfg).execute("SELECT * FROM ev_09_motor_customer_padding").df()
    assert pad.n_without_customer_after.tolist() == [0]


def test_claims_matched_net_of_recoveries_and_filtered(cfg):
    c = build(cfg).execute("SELECT * FROM line_claims").df().set_index("claim_id")
    assert set(c.index) == {"C1", "C2", "C3", "C5", "C6", "C7"}  # repudiated C4 and uncovered C8 dropped
    assert c.loc["C2", "net_cost"] == pytest.approx(300.0)
    assert c.loc["C3", "row_id"] == "M2"


def test_truncated_report_lag_pattern_by_hand(cfg):
    # Lag 10 is only observable for the 2021 claims: cost at risk 100 + 300, of which 300 reported at lag 10,
    # so F(5) = 1 - 300/400 = 0.25 and the 2024 claim aged 5 days is grossed up by 4.
    c = build(cfg).execute("SELECT * FROM line_claims_dev").df().set_index("claim_id")
    assert c.loc["C3", "dev_factor"] == pytest.approx(4.0)
    assert c.loc["C3", "net_cost_dev"] == pytest.approx(200.0)
    assert c.loc["C1", "dev_factor"] == pytest.approx(1.0)
    assert c.loc["C7", "dev_factor"] == pytest.approx(1.0)  # fully reported pattern: no uplift


def test_ibner_applies_to_motor_bodily_injury_only_and_can_be_switched_off(cfg):
    on = copy.deepcopy(cfg)
    on["decisions"]["D11_ibner_motor_bi"]["apply"] = True  # off by default since A28 (D-28)
    c = build(on).execute("SELECT * FROM line_claims_dev").df().set_index("claim_id")
    assert c.loc["C6", "dev_factor"] == pytest.approx(2.0)  # 2021 cohort 2,000 / 2024 cohort 1,000
    assert c.loc["C5", "dev_factor"] == pytest.approx(1.0)
    off = copy.deepcopy(cfg)
    off["decisions"]["D11_ibner_motor_bi"]["apply"] = False
    off["decisions"]["D10_develop_ibnr"]["apply"] = False
    c = build(off).execute("SELECT * FROM line_claims_dev").df().set_index("claim_id")
    assert c.dev_factor.tolist() == [1.0] * len(c)


def test_bornhuetter_ferguson_for_an_immature_year_with_a_mature_prior(cfg):
    # A mature 2023 own-damage year (600, fully reported) makes F(5) = 1 - 300/1000 = 0.7, so 2024 is only 70%
    # reported: below bf_max_reported_share, so BF fills the missing 30% from 2023's 600 per year of exposure.
    extra_row = [row("M5", "PM3", date(2023, 1, 1), date(2023, 12, 31))]
    extra_claim = [claim("C9", "PM3", "motor", "own_damage", pd.Timestamp(2023, 1, 1), 1, 600.0)]
    con = build(cfg, extra_row, extra_claim)
    c = con.execute("SELECT * FROM line_claims_dev").df().set_index("claim_id")
    expo_2024 = 2 * 366 / 365  # rows M2 and M4, exposure recomputed from the dates of a leap year
    expo_2023 = 1.0
    assert c.loc["C3", "dev_factor"] == pytest.approx((50 + 0.3 * 600 / expo_2023 * expo_2024) / 50)
    assert c.loc["C9", "dev_factor"] == pytest.approx(1.0)  # the mature year keeps its chain-ladder factor
    off = copy.deepcopy(cfg)
    off["decisions"]["D17_bf_slow_perils"]["apply"] = False
    c = build(off, extra_row, extra_claim).execute("SELECT * FROM line_claims_dev").df().set_index("claim_id")
    assert c.loc["C3", "dev_factor"] == pytest.approx(1 / 0.7)
