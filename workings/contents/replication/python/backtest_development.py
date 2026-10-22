"""Backtest of the development method (sql/10) as at 2023-12-31 (audit A17; decisions D-10, D-17).

Rebuilds the position a year earlier from report dates: only claims reported by 2023-12-31, with the valuation
date moved back. Loss year 2023 is then developed by the chain ladder and by the method actually used (BF where it
applies), and both are compared with 2023 as now known: current incurred of every claim reported since, plus the
small IBNR still estimated today. IBNER is off on both sides, because the reported data cannot test it.

Usage (from the pack root):  python python/backtest_development.py
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "development_backtest"
AS_AT = "2023-12-31"


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
        con.execute("SET VARIABLE d11_ibner_motor_bi = false")
        con.execute(sql10)
        con.execute("""CREATE TABLE known_now AS SELECT line, peril, sum(net_cost_dev) AS cost_now
                       FROM line_claims_dev WHERE year(loss_date) = 2023 GROUP BY ALL""")
        con.execute("ALTER TABLE line_claims RENAME TO line_claims_all")
        con.execute(f"CREATE TABLE line_claims AS SELECT * FROM line_claims_all WHERE report_date <= DATE '{AS_AT}'")
        con.execute(f"SET VARIABLE valuation_date = DATE '{AS_AT}'")
        con.execute(sql10)
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
        out = write_csv(df, out_dir / "ev_d1_development_backtest_2023.csv")
        rec["outputs_sha256"] = {out.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(out)}
        rec["metrics"] = {"development_backtest": df.to_dict("records")}
        rec["status"] = "success"
        print(df.to_string(index=False))
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
