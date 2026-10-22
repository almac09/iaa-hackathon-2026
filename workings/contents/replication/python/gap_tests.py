"""Four gaps found by inverting the model (audit A21), each tested under the two-year rule.

  G1 missed payments: the row's own value may partly record the period's outcome (its gap between claimants and
     others is 1.3-2.2 times larger for the same year's claims than for next year's). Variants: the previous row's
     value, which is known before the period starts (the row's own value where there is no previous row), and none.
  G2 adoption by significance: each adopted component re-judged by its two-year mean bootstrap gain (ev_g8), rather
     than by the sign of its gain in each year.
  G3 dispersion: frequency and severity move in opposite directions for several factors, so a Tweedie GLM with one
     dispersion weights rows wrongly. Variant: a double GLM (the mean refitted with weights from a gamma GLM of its
     unit deviances), for the lines with a GLM small-claim model (contents, motor).
  G4 seasonality: buildings claims run about 10% above average in Q1 and Q4. Variant: exposure weighted by the
     line's claim frequency by month of year, estimated on the fitting years only.

Baseline: the chosen model without CANN, whose +0.00005 would only add noise here, with the threshold, penalty,
trees and credibility k of the last full run (ev_g6_summary.csv). Each variant changes one thing; XGBoost's tree
count is re-tuned on the variant's own data, as in the full run. The backtest rows carry any leak too, so for G1
the two-year rule can favour a leaking feature; the result is read with that in mind.

Usage (from the pack root, after python/level_uncertainty.py):  python python/gap_tests.py
"""
from __future__ import annotations

import copy
import datetime as dt
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, TweedieRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import significance  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "gap_tests"
EVAL = rgl.EVAL_WINDOWS
MONTH_DAYS = ", ".join(
    f"greatest(0, date_diff('day', greatest(period_start, make_date(yr, {m}, 1)), "
    f"least(period_end, last_day(make_date(yr, {m}, 1)))) + 1) AS d{m}" for m in range(1, 13))
_TREES: dict[int, tuple] = {}


class DoubleGLM(rgl.GLM):
    """Tweedie GLM whose rows are reweighted by a fitted dispersion (Smyth and Jorgensen's double GLM): a gamma GLM of
    the weighted unit deviances on the mean model's own design gives phi per row, normalised to mean 1, and the mean
    is refitted with weights divided by phi."""

    def fit(self, d: pd.DataFrame, y: np.ndarray, p: float) -> "DoubleGLM":
        super().fit(d, y, p)
        X, _ = models.design_matrix(self._frame(d), self.levels, self.numeric)
        e = d.exposure.to_numpy()
        w = e ** (2 - p)
        dev = np.maximum(w * models.tweedie_deviance(y / e, self.model.predict(X), p), 1e-12)
        phi = GammaRegressor(alpha=self.alpha, **models.SOLVER).fit(X, dev).predict(X)
        self.phi_range = (float(np.percentile(phi / phi.mean(), 1)), float(np.percentile(phi / phi.mean(), 99)))
        self.model = TweedieRegressor(power=p, link="log", alpha=self.alpha, **models.SOLVER).fit(
            X, y / e, sample_weight=w / (phi / phi.mean()))
        return self


class DoubleGLMLine(rgl.Line):
    """A line whose small-claim GLM is a double GLM (G3); everything else as in the full run."""

    def build(self, years, threshold, alpha, trees=None, trend=True, a16=False):
        model = super().build(years, threshold, alpha, trees, trend, a16)
        d = self.rows(years)
        spec, cont = self.glm_spec(self.spec, d)
        model.small = DoubleGLM(spec, cont, alpha, trend).fit(d, self.targets(d, threshold)[0], self.p)
        return model


def clone(line: rgl.Line, df: pd.DataFrame | None = None, cfg: dict | None = None, cls=rgl.Line) -> rgl.Line:
    new = cls(line.name, line.df if df is None else df, line.claims, line.cfg if cfg is None else cfg)
    new.expected = line.expected
    return new


def settings(line: rgl.Line, summary: pd.DataFrame) -> dict:
    s = summary[summary.line == line.name].iloc[0]
    ks = {g: float(s[f"k_{g}"]) for g in rgl.GROUPS[line.name]}
    threshold = None if pd.isna(s.threshold) else float(s.threshold)
    alpha = None if pd.isna(s.alpha) else float(s.alpha)
    return {"threshold": threshold, "alpha": alpha, "ks": ks}


def window_preds(lines_by_window: dict, s: dict, fixed_trees: tuple | None = None) -> dict:
    """Predictions on each backtest year from a line (or one line per window), as the full run makes them.
    `fixed_trees` reuses the baseline's XGBoost trees instead of re-tuning them on the variant."""
    out = {}
    for w in EVAL:
        line = lines_by_window[w]
        yrs, yv = rgl.WINDOWS[w]
        trees = fixed_trees
        if line.spec["small"] == "xgb" and fixed_trees is None:  # tuned once per line object, on the tuning split as in the full run
            if id(line) not in _TREES:
                _TREES[id(line)] = line.tune_trees(s["threshold"])
                print(f"  {line.name}: {_TREES[id(line)][0]} trees", flush=True)
            trees = _TREES[id(line)]
        model = line.build(yrs, s["threshold"], s["alpha"], trees)
        out[w] = line.predict(model, w, s["ks"], line.rows([yv]), year=yv)
    return out


def compare(base_line: rgl.Line, base: dict, alt: dict, test: str, variant: str, cfg: dict) -> list[dict]:
    bs, p, rows = cfg["glm_lines"]["bootstrap"], base_line.p, []
    for w in EVAL:
        v = base_line.rows([rgl.WINDOWS[w][1]])
        y = v.y_full.to_numpy()
        diff = significance.paired_cluster_bootstrap(y, alt[w], base[w], v[bs["cluster"]].to_numpy(), p, bs["n"],
                                                     bs["seed"])
        obs = models.d2_tweedie(y, alt[w], p) - models.d2_tweedie(y, base[w], p)
        rows.append({"line": base_line.name, "test": test, "variant": variant, "score_year": rgl.WINDOWS[w][1],
                     "d2_baseline": models.d2_tweedie(y, base[w], p), "d2_variant": models.d2_tweedie(y, alt[w], p),
                     "pred_over_developed": alt[w].sum() / y.sum(), **significance.summarise(diff, obs)})
    passed = all(r["d2_gain"] > 0 for r in rows)
    mean = np.mean([r["d2_gain"] for r in rows])  # two-year mean with the G2 approximation (years independent)
    se = np.sqrt(sum(((r["ci_high_95"] - r["ci_low_95"]) / 3.92) ** 2 for r in rows)) / len(rows)
    for r in rows:
        r["passes_two_year_rule"] = passed
        r["two_year_mean"], r["two_year_ci_low_95"], r["two_year_ci_high_95"] = mean, mean - 1.96 * se, mean + 1.96 * se
        r["passes_significance_rule"] = passed and mean - 1.96 * se > 0
    print(f"  {base_line.name} {test} {variant}: " + ", ".join(
        f"{r['score_year']} {r['d2_gain']:+.5f} ({r['ci_low_95']:+.5f}, {r['ci_high_95']:+.5f})" for r in rows)
        + f" -> {'passes' if passed else 'fails'}", flush=True)
    return rows


def lagged_missed_payments(con, line: rgl.Line) -> pd.Series:
    """The policy's previous row's missed payments (known before this row starts), by row_id; the row's own value
    where the policy has no earlier row."""
    sql = f"""WITH r AS (SELECT row_id, policy_id, period_start, missed_payments_12m FROM policies_{line.name}
                         UNION ALL SELECT row_id, policy_id, period_start, missed_payments_12m FROM test_{line.name})
              SELECT row_id, coalesce(lag(missed_payments_12m) OVER (PARTITION BY policy_id ORDER BY period_start),
                                      missed_payments_12m) AS mp_lag,
                     lag(missed_payments_12m) OVER (PARTITION BY policy_id ORDER BY period_start) IS NULL AS no_prior
              FROM r"""
    lag = con.execute(sql).df().set_index("row_id")
    line.cfg.setdefault("_lag_coverage", {})[line.name] = 1 - float(lag.no_prior.mean())
    return lag.mp_lag.reindex(line.df.row_id.astype(lag.index.dtype).to_numpy())


def g1_frames(con, line: rgl.Line) -> dict:
    """Variant lines for G1: missed payments lagged, and dropped."""
    lag = lagged_missed_payments(con, line).to_numpy()
    df = line.df.copy()
    df["missed_payments_12m"] = lag
    if "f_missed_pay" in df:  # contents: GLM A's band, rebuilt from the lagged value with the same mapping
        mapping = line.df.groupby("missed_payments_12m").f_missed_pay.first()
        df["f_missed_pay"] = pd.Series(lag).map(mapping).fillna("MISSING").to_numpy()
    cfg_drop = copy.deepcopy(line.cfg)
    m = cfg_drop["glm_lines"]["models"][line.name]
    if line.name == "buildings":
        lm = cfg_drop["lines"]["models"]["buildings"]
        lm["numeric"] = [c for c in lm["numeric"] if c != "missed_payments_12m"]
    else:
        m["numeric"] = [c for c in m["numeric"] if c != "missed_payments_12m"]
        cats = rgl.resolve(cfg_drop, m["categorical"])
        m["categorical"] = [c for c in cats if c != "f_missed_pay"]
    return {"lagged": clone(line, df=df), "dropped": clone(line, cfg=cfg_drop)}


def seasonal_weights(con, line: rgl.Line, years: list[int]) -> pd.DataFrame:
    """Claim frequency by month of loss relative to the year, on the fitting years' training rows."""
    yrs = ", ".join(str(y) for y in years)
    days = con.execute(f"""SELECT {", ".join(f"sum(d{m}) AS d{m}" for m in range(1, 13))}
                           FROM (SELECT {MONTH_DAYS} FROM line_rows
                                 WHERE line = '{line.name}' AND src = 'train' AND yr IN ({yrs}))""").df().iloc[0]
    n = con.execute(f"""SELECT month(loss_date) AS m, count(*) AS n FROM line_claims
                        WHERE line = '{line.name}' AND yr IN ({yrs}) GROUP BY ALL""").df().set_index("m").n
    rate = pd.Series({m: n.get(m, 0) / days[f"d{m}"] for m in range(1, 13)})
    overall = n.sum() / days.sum()
    return pd.DataFrame({"line": line.name, "fit_years": f"{years[0]}-{years[-1]}", "month": range(1, 13),
                         "relative_frequency": (rate / overall).to_numpy(), "claims": [n.get(m, 0) for m in range(1, 13)]})


def g4_line(con, line: rgl.Line, w: pd.DataFrame) -> rgl.Line:
    """The line with exposure replaced by days weighted by the month's relative frequency."""
    days = con.execute(f"SELECT row_id, {MONTH_DAYS} FROM line_rows WHERE line = '{line.name}'").df()
    weights = w.relative_frequency.to_numpy()
    days["exposure_seasonal"] = days[[f"d{m}" for m in range(1, 13)]].to_numpy() @ weights / 365
    s = days.set_index("row_id").exposure_seasonal
    df = line.df.copy()
    df["exposure"] = s.reindex(df.row_id.astype(s.index.dtype).to_numpy()).to_numpy()
    ratio = df.exposure / line.df.exposure
    if ratio.isna().any() or not ((ratio > 0.5) & (ratio < 2)).all():
        raise ValueError(f"{line.name}: seasonal exposure misaligned with the rows")
    first, last = (int(y) for y in w.fit_years.iloc[0].split("-"))
    fit = df.src.eq("train") & df.yr.between(first, last)
    w["exposure_ratio_p01"], w["exposure_ratio_p99"] = ratio.quantile(0.01), ratio.quantile(0.99)
    w["fit_exposure_ratio"] = df.exposure[fit].sum() / line.df.exposure[fit].sum()
    return clone(line, df=df)


def significance_rule(cfg: dict) -> pd.DataFrame:
    """G2: each component's two-year mean gain and an approximate 95% interval from the per-year bootstrap intervals
    (standard error = interval width / 3.92). The years are treated as independent; they share customers, so if
    anything the combined interval is too narrow."""
    ev = pd.read_csv(common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "glm_lines" / "ev_g8_bootstrap_significance.csv")
    ev["se"] = (ev.ci_high_95 - ev.ci_low_95) / 3.92
    rows = []
    for (line, comp), g in ev.groupby(["line", "chosen_vs"], sort=False):
        mean, se = g.d2_gain.mean(), np.sqrt((g.se ** 2).sum()) / len(g)
        kept = not comp.startswith("with ")
        rows.append({"line": line, "component": comp.replace("no ", "").replace("with ", "").replace(" (not adopted)", ""),
                     "currently_adopted": kept, "gain_2023": g.d2_gain.iloc[0], "gain_2024": g.d2_gain.iloc[1],
                     "two_year_mean": mean, "ci_low_95": mean - 1.96 * se, "ci_high_95": mean + 1.96 * se,
                     "significant_two_year": bool(mean - 1.96 * se > 0)})
    out = pd.DataFrame(rows)
    # Sign convention of ev_g8: chosen minus variant. For a component not adopted, the variant contains it.
    flip = ~out.currently_adopted
    for c in ("gain_2023", "gain_2024", "two_year_mean"):
        out.loc[flip, c] = -out.loc[flip, c]
    lo, hi = out.ci_low_95.copy(), out.ci_high_95.copy()
    out.loc[flip, "ci_low_95"], out.loc[flip, "ci_high_95"] = -hi[flip], -lo[flip]
    out["significant_two_year"] = out.ci_low_95 > 0
    out["kept_under_two_year_rule"] = (out.gain_2023 > 0) & (out.gain_2024 > 0)
    out["kept_under_significance_rule"] = out.kept_under_two_year_rule & out.significant_two_year
    return out


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "glm_lines"
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        lines, _ = rgl.prepare(con, cfg, rec)
        summary = pd.read_csv(out_dir / "ev_g6_summary.csv")
        results, weights = [], []

        rule = significance_rule(cfg)
        print("G2 significance rule:\n" + rule.round(5).to_string(index=False), flush=True)

        for name, line in lines.items():
            print(f"{name}: baseline", flush=True)
            s = settings(line, summary)
            base = window_preds({w: line for w in EVAL}, s)

            fixed = _TREES.get(id(line))  # buildings: the baseline's trees, to separate a variant from re-tuning
            for variant, vline in g1_frames(con, line).items():
                results += compare(line, base, window_preds({w: vline for w in EVAL}, s), "G1 missed payments",
                                   variant, cfg)
                if fixed:
                    results += compare(line, base, window_preds({w: vline for w in EVAL}, s, fixed),
                                       "G1 missed payments", f"{variant} (baseline trees)", cfg)

            if line.spec["small"] == "glm":
                dline = clone(line, cls=DoubleGLMLine)
                results += compare(line, base, window_preds({w: dline for w in EVAL}, s), "G3 dispersion",
                                   "double GLM", cfg)

            wl = {}
            for w in EVAL:
                sw = seasonal_weights(con, line, rgl.WINDOWS[w][0])
                weights.append(sw)
                wl[w] = g4_line(con, line, sw)
            results += compare(line, base, window_preds(wl, s), "G4 seasonality", "seasonal exposure", cfg)
            if fixed:
                results += compare(line, base, window_preds(wl, s, fixed), "G4 seasonality",
                                   "seasonal exposure (baseline trees)", cfg)
        con.close()

        res = pd.DataFrame(results)
        res["lag_coverage"] = res.line.map(cfg.get("_lag_coverage", {}))
        outputs = [write_csv(res, out_dir / "ev_g11_gap_tests.csv"),
                   write_csv(rule, out_dir / "ev_g12_significance_rule.csv"),
                   write_csv(pd.concat(weights), out_dir / "ev_g13_seasonal_weights.csv")]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {"gap_tests": res.round(6).to_dict("records"), "significance_rule": rule.round(6).to_dict("records")}
        rec["status"] = "success"
        print(res[["line", "test", "variant", "score_year", "d2_baseline", "d2_variant", "d2_gain", "ci_low_95",
                   "ci_high_95", "passes_two_year_rule"]].round(5).to_string(index=False))
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
