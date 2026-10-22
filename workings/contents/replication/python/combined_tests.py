"""The best ideas from the team's other branches, tested on this pack's model under the two-year rule (audit A24).

  C1 timber merge (Cian, R4): buildings construction_type T (2021-22) and W (2023 on) are one category recoded
     (D-24). Measured with the switch off against on; applied as a data correction whatever the result.
  C2 GLM + XGBoost blend (Alan, Bridget): each line's small-claim model blended linearly with the other family.
     The GLM lines get an XGBoost rate model with the settings that won Bridget's tuning harness (depth 4,
     min_child_weight 10, colsample 0.7, lambda 3, her fixed 500 trees; the year trend backfitted into the offset
     as in this pack);
     buildings gets a smoothed Tweedie GLM on its frequency-model factors. The blend weight is chosen on one
     backtest year and scored on the other, so no year is scored with a weight tuned on it.
  C3 frequency x severity (Bridget's queue, A21): a Poisson GLM for claim counts times a gamma GLM for the mean
     small-claim cost, each with the line's factors and its own year trend, in place of the Tweedie GLM (contents,
     motor). Counts are as reported, so the severity part carries the development.

Baseline: the production model without CANN (its +0.00005 would only add noise), with the settings of the last
full run (ev_g6_summary.csv) and contents' seasonal exposure (D-22). Optional additions are adopted only if they
pass the two-year rule and their two-year mean gain is significant (D-23).

Usage (from the pack root, after python/run_glm_lines.py):  python python/combined_tests.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import xgb_models  # noqa: E402
from gap_tests import compare, settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "combined_tests"
EVAL = rgl.EVAL_WINDOWS
BRIDGET_XGB = {"max_depth": 4, "min_child_weight": 10, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
               "reg_lambda": 3.0}
BRIDGET_TREES = 500  # her fixed tree count; the early-stopped count is logged beside it
WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]


def window_preds(line: rgl.Line, s: dict, build) -> dict:
    """Predictions on each backtest year, as the full run makes them: the window's exposure (seasonal for contents),
    a model from `build(line, years)`, the credibility factor and rebalancing."""
    out = {}
    for w in EVAL:
        yrs, yv = rgl.WINDOWS[w]
        line.use_window(w)
        out[w] = line.predict(build(line, yrs), w, s["ks"], line.rows([yv]), year=yv)
    return out


def baseline_build(s: dict, trees=None):
    return lambda line, yrs: line.build(yrs, s["threshold"], s["alpha"], trees)


def features(line: rgl.Line) -> tuple[list[str], list[str]]:
    node = line.cfg
    for key in line.cfg["glm_lines"]["cann"]["features"][line.name].split("."):
        node = node[key]
    return rgl.resolve(line.cfg, node["numeric"]), rgl.resolve(line.cfg, node["categorical"])


def early_stopped_trees(line: rgl.Line, threshold, numeric, categorical) -> int:
    """Early-stopped tree count for Bridget's settings on the tuning split (fit 2021-22, stop on 2023), for the
    record: she used a fixed 500, which is what the test uses."""
    xc = line.cfg["xgboost"]
    line.use_window("tune")
    (years, val_year), cats = rgl.WINDOWS["tune"], xgb_models.category_levels(line.train, categorical)
    d, v = line.rows(years), line.rows([val_year])
    y, yv = line.targets(d, threshold)[0], line.targets(v, threshold)[0]
    b = xgb_models.fit(xgb_models.matrix(d, numeric, cats), y, d.exposure.to_numpy(), y.sum() / d.exposure.sum(),
                       line.p, BRIDGET_XGB, xc["max_rounds"], xc["seed"], xc["nthread"],
                       val=(xgb_models.matrix(v, numeric, cats), yv, v.exposure.to_numpy()),
                       early_stopping=xc["early_stopping_rounds"])
    return b.best_iteration + 1


def alternative_build(line: rgl.Line, s: dict, trees=None):
    """The other model family for the small claims, with everything else (large claims, tail) unchanged."""
    if line.spec["small"] == "glm":
        numeric, categorical = features(line)
        stopped = early_stopped_trees(line, s["threshold"], numeric, categorical)
        n_trees, params = BRIDGET_TREES, BRIDGET_XGB
        print(f"  {line.name}: blend XGBoost {n_trees} trees (early stopping on 2023 would give {stopped})", flush=True)

        def build(ln, yrs):
            model = ln.build(yrs, s["threshold"], s["alpha"])
            d = ln.rows(yrs)
            model.small = rgl.XGBSmall(numeric, categorical, ln.train, params, n_trees, ln.cfg["xgboost"],
                                       ln.cfg["lines"]["trend_iterations"]).fit(d, ln.targets(d, s["threshold"])[0], ln.p)
            return model
        return build
    part = {**line.spec["freq"], "smooth": "all", "n_knots": 4}

    def build(ln, yrs):
        model = ln.build(yrs, s["threshold"], s["alpha"], trees)
        d = ln.rows(yrs)
        spec, cont = ln.glm_spec(part, d)
        model.small = rgl.GLM(spec, cont, 1e-4, trend=True).fit(d, ln.targets(d, s["threshold"])[0], ln.p)
        return model
    return build


def blend(line: rgl.Line, base: dict, alt: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Weight on the alternative chosen on each year and scored on the other (C2)."""
    grid = []
    for w in EVAL:
        y = line.rows([rgl.WINDOWS[w][1]]).y_full.to_numpy()
        for wt in WEIGHTS:
            grid.append({"line": line.name, "score_year": rgl.WINDOWS[w][1], "weight_on_alternative": wt,
                         "d2": models.d2_tweedie(y, (1 - wt) * base[w] + wt * alt[w], line.p)})
    g = pd.DataFrame(grid)
    best = {w: float(g[g.score_year == rgl.WINDOWS[w][1]].sort_values("d2").weight_on_alternative.iloc[-1])
            for w in EVAL}
    other = {"tune": "backtest", "backtest": "tune"}
    crossed = {w: (1 - best[other[w]]) * base[w] + best[other[w]] * alt[w] for w in EVAL}
    rows = compare(line, base, crossed, "C2 blend",
                   f"weight {best['tune']:.1f} (chosen on 2023) / {best['backtest']:.1f} (on 2024), scored crosswise",
                   cfg)
    for r in rows:
        r["weight_chosen_2023"], r["weight_chosen_2024"] = best["tune"], best["backtest"]
        r["weight_if_adopted"] = (best["tune"] + best["backtest"]) / 2
    return rows, grid


class SevGLM(rgl.GLM):
    """Gamma GLM for the mean cost per claim on the rows with a positive cost; factor levels from all fitting
    rows."""

    def fit(self, d: pd.DataFrame, y: np.ndarray, n: np.ndarray, p: float) -> "SevGLM":
        self.centre = float(np.average(d.yr, weights=d.exposure))
        f = self._frame(d)
        self.numeric = self.basis_columns + self.continuous + (["t"] if self.trend else [])
        self.levels = models.factor_levels(f, [c for c in f if c not in self.numeric])
        X, self.names = models.design_matrix(f, self.levels, self.numeric)
        pos = (n > 0) & (y > 0)
        self.model = models.fit_gamma(X[pos], y[pos] / n[pos], n[pos], self.alpha)
        self.slope = float(self.model.coef_[-1]) if self.trend else 0.0
        return self

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        X, _ = models.design_matrix(self._frame(d, year), self.levels, self.numeric)
        return self.model.predict(X)


class FreqSev:
    """Small-claim cost = Poisson frequency x gamma severity (C3). `slope` is the combined trend, which the
    large-claim part also takes."""

    def __init__(self, spec, cont, alpha):
        self.freq = rgl.GLM(spec, cont, alpha, trend=True, family="poisson")
        self.sev = SevGLM(spec, cont, alpha, trend=True)

    def fit(self, d: pd.DataFrame, y: np.ndarray, n: np.ndarray, p: float) -> "FreqSev":
        n = np.where(y > 0, n, 0.0)  # claims that cost nothing (within the excess, fully recovered) are not counted
        self.freq.fit(d, n, p)
        self.sev.fit(d, y, n, p)
        self.slope = self.freq.slope + self.sev.slope
        return self

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        return self.freq.predict(d, year) * self.sev.predict(d, year)


def freqsev_build(s: dict):
    def build(ln, yrs):
        model = ln.build(yrs, s["threshold"], s["alpha"])
        d = ln.rows(yrs)
        spec, cont = ln.glm_spec(ln.spec, d)
        model.small = FreqSev(spec, cont, s["alpha"]).fit(d, ln.targets(d, s["threshold"])[0],
                                                          d.n_claims.to_numpy(dtype=float), ln.p)
        return model
    return build


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "glm_lines"
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        summary = pd.read_csv(out_dir / "ev_g6_summary.csv")
        results, grid = [], []

        # C1: buildings with the timber codes apart (as before) against merged (D-24).
        cfg_off = copy.deepcopy(cfg)
        cfg_off["decisions"]["D24_merge_timber"]["apply"] = False
        con = common.open_duckdb(cfg_off)
        apart = rgl.prepare(con, cfg_off, {})[0]["buildings"]
        con.close()
        con = common.open_duckdb(cfg)
        lines = rgl.prepare(con, cfg, rec)[0]
        con.close()
        s = settings(lines["buildings"], summary)
        trees_apart, trees_merged = apart.tune_trees(s["threshold"]), lines["buildings"].tune_trees(s["threshold"])
        print(f"buildings trees: apart {trees_apart[0]}, merged {trees_merged[0]}", flush=True)
        base_apart = window_preds(apart, s, baseline_build(s, trees_apart))
        merged = window_preds(lines["buildings"], s, baseline_build(s, trees_merged))
        results += compare(apart, base_apart, merged, "C1 timber merge", "T and W merged", cfg)

        for name, line in lines.items():
            s = settings(line, summary)
            trees = trees_merged if line.spec["small"] == "xgb" else None
            base = merged if name == "buildings" else window_preds(line, s, baseline_build(s, trees))
            alt = window_preds(line, s, alternative_build(line, s, trees))
            rows, g = blend(line, base, alt, cfg)
            results += rows
            grid += g
            results += compare(line, base, alt, "C2 alternative alone",
                               "XGBoost (Bridget's settings)" if line.spec["small"] == "glm" else "smoothed GLM", cfg)
            if line.spec["small"] == "glm":
                results += compare(line, base, window_preds(line, s, freqsev_build(s)), "C3 frequency x severity",
                                   "Poisson x gamma GLM", cfg)

        res = pd.DataFrame(results)
        outputs = [write_csv(res, out_dir / "ev_c1_combined_tests.csv"),
                   write_csv(pd.DataFrame(grid), out_dir / "ev_c2_blend_weight_grid.csv")]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {"combined_tests": res.round(6).to_dict("records")}
        rec["status"] = "success"
        print(res[["line", "test", "variant", "score_year", "d2_baseline", "d2_variant", "d2_gain", "ci_low_95",
                   "ci_high_95", "passes_two_year_rule", "passes_significance_rule"]].round(5).to_string(index=False))
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
