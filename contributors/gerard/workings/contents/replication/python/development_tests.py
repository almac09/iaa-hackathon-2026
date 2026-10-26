"""Fitting on reported claims with development as an offset, tested under the two-year rule (audit A25).

The production model grosses each claim up to ultimate before fitting (up to about x4.5 for late-2024 motor bodily
injury), so a single recent claim enters the fit inflated. The textbook alternative fits the small-claim model to
reported cost, with the expected reported share F as part of the offset: E[reported] = mu x F. The model then
learns ultimate cost without amplifying individual claims, and predicts ultimate cost with plain exposure.

F per row: the reported / developed cost of its line and loss quarter (all perils, so it carries IBNR, BF and the
bodily-injury IBNER), weighted by the row's days in each quarter; 1 where a quarter has no claims. The large-claim
part, the tree counts and every other setting stay as in the production model (without CANN), so the comparison
isolates the fitting target. In the tuning window (fit 2021-22) F is close to 1, so the test mostly shows in the
backtest window, where 2023 is still developing; the final fit (2021-24) is where it matters most.

Usage (from the pack root, after python/run_glm_lines.py):
  python python/development_tests.py --run-label "A25 development methods: offset fit"
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
from combined_tests import baseline_build, window_preds  # noqa: E402
from gap_tests import compare, settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "development_tests"


def reported_share(con, line: str) -> pd.Series:
    """Expected reported share of each row's ultimate cost, by row_id."""
    cells = con.execute(f"""SELECT year(loss_date) AS yr, quarter(loss_date) AS q, sum(net_cost) / sum(net_cost_dev) AS f
                            FROM line_claims_dev WHERE line = '{line}' GROUP BY ALL""").df()
    f = {(int(r.yr), int(r.q)): float(r.f) for r in cells.itertuples()}
    days = con.execute(f"SELECT row_id, yr, {rgl.MONTH_DAYS} FROM line_rows WHERE line = '{line}'").df()
    q_days = np.stack([days[[f"d{m}" for m in range(3 * q - 2, 3 * q + 1)]].sum(axis=1).to_numpy() for q in (1, 2, 3, 4)], 1)
    q_f = np.array([[f.get((int(y), q), 1.0) for q in (1, 2, 3, 4)] for y in days.yr.to_numpy()])
    share = (q_days * q_f).sum(1) / q_days.sum(1)
    return pd.Series(share, index=days.row_id.to_numpy())


class OffsetLine(rgl.Line):
    """A line whose small-claim model is fitted to reported cost with exposure x F as the offset."""

    f_row: pd.Series

    def reported_small(self, d: pd.DataFrame, threshold) -> np.ndarray:
        c = self.claims[self.claims.row_id.isin(d.row_id)]
        part = c.net_cost if threshold is None else np.minimum(c.net_cost, threshold)
        return part.groupby(c.row_id).sum().reindex(d.row_id.to_numpy(), fill_value=0).to_numpy()

    def build(self, years, threshold, alpha, trees=None, trend=True, a16=False):
        model = super().build(years, threshold, alpha, trees, trend, a16)
        d = self.rows(years)
        y = self.reported_small(d, threshold)
        d_fit = d.assign(exposure=d.exposure.to_numpy() * self.f_row.reindex(d.row_id.to_numpy()).to_numpy())
        if self.spec["small"] == "glm":
            spec, cont = self.glm_spec(self.spec, d)
            model.small = rgl.GLM(spec, cont, alpha, trend).fit(d_fit, y, self.p)
        else:
            lm = self.cfg["lines"]["models"][self.name]
            model.small = rgl.XGBSmall(lm["numeric"], lm["categorical"], self.train, trees[1], trees[0],
                                       self.cfg["xgboost"], self.cfg["lines"]["trend_iterations"]).fit(d_fit, y, self.p)
        return model


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "glm_lines"
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        summary = pd.read_csv(out_dir / "ev_g6_summary.csv")
        con = common.open_duckdb(cfg)
        lines = rgl.prepare(con, cfg, rec)[0]
        shares = {name: reported_share(con, name) for name in lines}
        con.close()
        results, share_rows = [], []
        for name, line in lines.items():
            s = settings(line, summary)
            trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
            base = window_preds(line, s, baseline_build(s, trees))
            # From plain exposure: the original line may already hold a window's seasonal exposure (D-22).
            off = OffsetLine(line.name, line.df.assign(exposure=line.df.exposure_plain), line.claims, line.cfg)
            off.expected, off.month_days, off.claim_months = line.expected, line.month_days, line.claim_months
            off.f_row = shares[name]
            results += compare(line, base, window_preds(off, s, baseline_build(s, trees)), "D1 offset fit",
                               "reported cost, offset exposure x F", cfg)
            yr = line.df.set_index("row_id").yr
            sh = shares[name]
            share_rows += [{"line": name, "yr": int(y), "mean_reported_share": float(sh[yr[yr == y].index].mean()),
                            "min_reported_share": float(sh[yr[yr == y].index].min())} for y in sorted(yr.unique())]
        res = pd.DataFrame(results)
        outputs = [write_csv(res, out_dir / "ev_dv1_offset_fit.csv"),
                   write_csv(pd.DataFrame(share_rows), out_dir / "ev_dv2_reported_share_by_year.csv")]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {"offset_fit": res.round(6).to_dict("records"), "reported_share": share_rows}
        rec["status"] = "success"
        print(pd.DataFrame(share_rows).round(4).to_string(index=False))
        print(res[["line", "score_year", "d2_baseline", "d2_variant", "d2_gain", "ci_low_95", "ci_high_95",
                   "passes_two_year_rule", "passes_significance_rule"]].round(5).to_string(index=False))
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
