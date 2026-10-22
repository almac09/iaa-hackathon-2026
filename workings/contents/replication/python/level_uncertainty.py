"""Uncertainty in the development to ultimate, by bootstrap (audit A20; feeds the level step of run_glm_lines.py).

Resamples training policy rows with replacement (their claims and exposure with them), re-runs the development
SQL (sql/10: reporting pattern, Bornhuetter-Ferguson, IBNER) on each resample, and records the development factor
(developed / reported cost) of loss years 2023 and 2024 per line. Runs the resamples in parallel processes, each
with its own DuckDB connection. Output: audit/evidence/glm_lines/ev_u1_development_bootstrap.csv.

Mack (A25): the bootstrap covers sampling noise in the pattern only. Mack's standard error of the chain-ladder
reserve on quarterly reporting triangles (reported cost by loss quarter and quarters to report) adds the process
error of the claims still to be reported. Output: ev_u2_mack_development.csv, coefficient of variation of the
ultimate for loss years 2023 and 2024 per line.

Usage (from the pack root):  python python/level_uncertainty.py
"""
from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import mack  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "level_uncertainty"
FACTORS_SQL = """SELECT line, year(loss_date) AS ay, sum(net_cost_dev) / sum(net_cost) AS factor
                 FROM line_claims_dev WHERE year(loss_date) IN (2023, 2024) GROUP BY ALL"""
_con, _ids, _sql10 = None, None, None


def _init(cfg: dict) -> None:
    global _con, _ids, _sql10
    _con = common.open_duckdb(cfg)
    common.run_sql_files(_con, cfg)
    _con.execute((common.PACK_ROOT / "sql/09_build_lines.sql").read_text())
    _con.execute("ALTER TABLE line_rows RENAME TO line_rows_full")
    _con.execute("ALTER TABLE line_claims RENAME TO line_claims_full")
    _ids = _con.execute("SELECT row_id FROM line_rows_full WHERE src = 'train' ORDER BY row_id").df().row_id.to_numpy()
    _sql10 = (common.PACK_ROOT / "sql/10_development.sql").read_text()


def _one(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = np.bincount(rng.integers(0, len(_ids), len(_ids)), minlength=len(_ids))
    keep = counts > 0
    _con.register("samp_df", pd.DataFrame({"row_id": _ids[keep], "k": counts[keep]}))
    _con.execute("""CREATE OR REPLACE TABLE line_rows AS
                    SELECT r.* EXCLUDE (rep) FROM (SELECT r.*, unnest(range(s.k)) AS rep
                    FROM line_rows_full r JOIN samp_df s USING (row_id)) r""")
    _con.execute("""CREATE OR REPLACE TABLE line_claims AS
                    SELECT c.* EXCLUDE (rep) FROM (SELECT c.*, unnest(range(s.k)) AS rep
                    FROM line_claims_full c JOIN samp_df s USING (row_id)) c""")
    _con.execute(_sql10)
    return _con.execute(FACTORS_SQL).df().assign(seed=seed)


def mack_by_year(con, years=(2023, 2024)) -> pd.DataFrame:
    """Mack chain ladder per line on quarterly triangles of reported cost (IBNR only, as the data is a single
    snapshot), summarised for each loss year in `years`."""
    tri = con.execute("""
        SELECT line, (year(loss_date) - 2021) * 4 + quarter(loss_date) - 1 AS i,
               (year(report_date) * 4 + quarter(report_date)) - (year(loss_date) * 4 + quarter(loss_date)) AS k,
               sum(net_cost) AS c
        FROM line_claims WHERE year(loss_date) >= 2021 GROUP BY ALL""").df()
    rows = []
    for line, g in tri.groupby("line"):
        n = int(g.i.max()) + 1
        inc = np.zeros((n, n))
        for r in g.itertuples():
            inc[int(r.i), min(int(r.k), n - 1)] += r.c
        cum = np.cumsum(inc, axis=1)
        t = np.full((n, n), np.nan)
        for i in range(n):
            t[i, :n - i] = cum[i, :n - i]
        cl = mack.chain_ladder(t)
        for y in years:
            idx = [i for i in range(n) if 2021 + i // 4 == y]
            se = np.sqrt(mack.mack_mse(cl, idx))
            ult, res = cl["ultimate"][idx].sum(), cl["reserve"][idx].sum()
            rows.append({"line": line, "ay": y, "reported": cl["latest"][idx].sum(), "cl_ultimate": ult,
                         "cl_reserve": res, "mack_se": se, "mack_cv_of_ultimate": se / ult,
                         "mack_cv_of_reserve": se / res})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    lv = cfg["glm_lines"]["level"]
    rec = common.new_run_record(RUNNER, cfg)
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        common.run_sql_files(con, cfg)
        for rel in ("sql/09_build_lines.sql", "sql/10_development.sql"):
            con.execute((common.PACK_ROOT / rel).read_text())
        point = con.execute(FACTORS_SQL).df()
        mk = mack_by_year(con)
        con.close()
        with mp.get_context("spawn").Pool(lv["workers"], initializer=_init, initargs=(cfg,)) as pool:
            boot = pd.concat(pool.map(_one, [lv["seed"] + b for b in range(lv["n_boot"])]))
        g = boot.groupby(["line", "ay"]).factor
        out = point.rename(columns={"factor": "factor_point"}).merge(
            pd.DataFrame({"boot_mean": g.mean(), "boot_sd": g.std(), "p05": g.quantile(0.05), "p95": g.quantile(0.95),
                          "n_boot": g.size()}).reset_index(), on=["line", "ay"])
        out["cv"] = out.boot_sd / out.boot_mean
        path = write_csv(out.sort_values(["line", "ay"]), common.PACK_ROOT / lv["uncertainty_evidence"])
        mpath = write_csv(mk.sort_values(["line", "ay"]), common.PACK_ROOT / lv["mack_evidence"])
        rec["outputs_sha256"] = {q.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(q) for q in (path, mpath)}
        rec["metrics"] = {"development_bootstrap": out.round(6).to_dict("records"),
                          "mack": mk.round(6).to_dict("records")}
        rec["status"] = "success"
        print(out.round(4).to_string(index=False))
        print(mk.round(4).to_string(index=False))
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
