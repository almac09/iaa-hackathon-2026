"""Backtest of the development method (sql/10) (audit A17; decisions D-10, D-17; recent weighting A25).

Rebuilds the position a year earlier from report dates: only claims reported by the backtest date, with the
valuation date moved back. That loss year is then developed by the chain ladder and by the method actually used
(BF where it applies), and both are compared with the year as now known: current incurred of every claim reported
since, plus the small IBNR still estimated today. IBNER is off on both sides, because the reported data cannot
test it.

ev_d1: as at 2023-12-31 with the configured pattern (A17). ev_d2 (A25): as at 2022-12-31 and 2023-12-31, for each
pattern_decay in DECAYS (1 = unweighted; below 1 weights recent loss years more), by peril and for each line.

Usage (from the pack root):  python python/backtest_development.py
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "development_backtest"
AS_AT = "2023-12-31"
AS_ATS = ("2022-12-31", "2023-12-31")
DECAYS = (1.0, 0.5, 0.25)


def backtest(con, sql10: str, valuation: str, as_at: str, decay: float):
    """Loss year `as_at`'s year developed from claims reported by `as_at`, against the same year known now."""
    ay = int(as_at[:4])
    con.execute("SET VARIABLE d11_ibner_motor_bi = false")
    con.execute("SET VARIABLE pattern_decay = 1.0")
    con.execute(f"SET VARIABLE valuation_date = DATE '{valuation}'")
    con.execute("CREATE OR REPLACE TABLE line_claims AS SELECT * FROM line_claims_all")
    con.execute(sql10)
    con.execute(f"""CREATE OR REPLACE TABLE known_now AS SELECT line, peril, sum(net_cost_dev) AS cost_now
                    FROM line_claims_dev WHERE year(loss_date) = {ay} GROUP BY ALL""")
    con.execute(f"CREATE OR REPLACE TABLE line_claims AS SELECT * FROM line_claims_all WHERE report_date <= DATE '{as_at}'")
    con.execute(f"SET VARIABLE valuation_date = DATE '{as_at}'")
    con.execute(f"SET VARIABLE pattern_decay = {float(decay)}")
    con.execute(sql10)
    return con.execute(f"""
        WITH d AS (SELECT line, peril, sum(net_cost_dev) AS cost_used FROM line_claims_dev
                   WHERE year(loss_date) = {ay} GROUP BY ALL),
        j AS (SELECT b.line, b.peril, b.reported_share, b.cl_ultimate, d.cost_used, k.cost_now, b.bf_factor IS NOT NULL AS bf
              FROM bf_year b JOIN known_now k USING (line, peril) JOIN d USING (line, peril) WHERE b.ay = {ay})
        SELECT '{as_at}' AS as_at, {float(decay)} AS pattern_decay, line, peril, round(reported_share, 4) AS reported_share,
               round(cl_ultimate / cost_now, 4) AS chain_ladder_over_actual, round(cost_used / cost_now, 4) AS method_used_over_actual,
               bf AS bf_applied
        FROM j
        UNION ALL
        SELECT '{as_at}', {float(decay)}, line, 'ALL', NULL, round(sum(cl_ultimate) / sum(cost_now), 4),
               round(sum(cost_used) / sum(cost_now), 4), NULL
        FROM j GROUP BY line
        ORDER BY 3, 4""").df()


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "lines"
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        rec["sql_sha256"] = common.run_sql_files(con, cfg)
        con.execute((common.PACK_ROOT / "sql/09_build_lines.sql").read_text())
        sql10 = (common.PACK_ROOT / "sql/10_development.sql").read_text()
        valuation = cfg["sql_variables"]["valuation_date"]
        con.execute("ALTER TABLE line_claims RENAME TO line_claims_all")
        grid = [backtest(con, sql10, valuation, a, d) for a in AS_ATS for d in DECAYS]
        backtest(con, sql10, valuation, AS_AT, cfg["sql_variables"]["pattern_decay"])
        df = con.execute("""
            SELECT b.line, b.peril, round(b.reported_share, 4) AS reported_share_at_2023,
                   round(b.cl_ultimate / k.cost_now, 4) AS chain_ladder_over_actual,
                   round(d.cost_used / k.cost_now, 4) AS method_used_over_actual,
                   b.bf_factor IS NOT NULL AS bf_applied
            FROM bf_year b
            JOIN known_now k USING (line, peril)
            JOIN (SELECT line, peril, sum(net_cost_dev) AS cost_used FROM line_claims_dev
                  WHERE year(loss_date) = 2023 GROUP BY ALL) d USING (line, peril)
            WHERE b.ay = 2023
            ORDER BY 1, 2""").df()
        con.close()
        grid = pd.concat(grid)
        outs = [write_csv(df, out_dir / "ev_d1_development_backtest_2023.csv"),
                write_csv(grid, out_dir / "ev_d2_pattern_decay_backtest.csv")]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outs}
        rec["metrics"] = {"development_backtest": df.to_dict("records"),
                          "pattern_decay_lines": grid[grid.peril == "ALL"].to_dict("records")}
        rec["status"] = "success"
        print(df.to_string(index=False))
        print(grid[grid.peril.isin(["ALL", "bodily_injury", "subsidence"])].to_string(index=False))
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
