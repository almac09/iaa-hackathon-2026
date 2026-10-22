"""Split three-line model with claim-history credibility, and the combined submission (audit A16).

Owner's design: contents one Tweedie GLM; buildings small claims by XGBoost at depth 3 plus a large-claim GLM;
motor small-claim GLM plus a large-claim GLM, each with a calendar-year trend. Large claims are the developed cost
above a threshold, modelled by a compact Tweedie GLM, scaled to a generalised Pareto mean and given the
small-claim trend.

Claim history enters every line as a credibility factor (k + n) / (k + E): n prior claims of the customer (sql/12)
against E expected over the same prior rows by a frequency GLM without history (sql/13), for the own line, the
other home line and motor/home. Factors are rebalanced so the fitted rows keep their total. History as plain
model factors was tried first and rejected (A16): its length grows with the calendar year, so it confounds the
trend and the 2025 rows fall outside the range seen in training.

  tune   fit 2021-22, score 2023: GLM penalty, large-claim threshold, XGBoost trees, then k per group;
  test   fit 2021-23, score 2024, with variants that drop the history, the split or the trend;
  final  fit 2021-24, predict the 2025 rows; one submission with every line.

Usage (from the pack root):  python python/run_glm_lines.py
"""
from __future__ import annotations

import datetime as dt
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import glm_lines  # noqa: E402
import make_submission  # noqa: E402
import models  # noqa: E402
import xgb_models  # noqa: E402
from run_lines import median_hessian  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "glm_lines"
WINDOWS = {"tune": ([2021, 2022], 2023), "backtest": ([2021, 2022, 2023], 2024),
           "final": ([2021, 2022, 2023, 2024], 2025)}
GROUPS = {"buildings": ["own", "home", "cross"], "contents": ["own", "home", "cross"], "motor": ["own", "cross"]}


def resolve(cfg: dict, v) -> list[str]:
    return cfg["model"][v.split(".")[1]] if isinstance(v, str) else list(v)


class GLM:
    """Tweedie (or Poisson) GLM on banded factors with a log-exposure offset and, optionally, a linear year term."""

    def __init__(self, spec: glm_lines.FactorSpec, continuous: list[str], alpha: float, trend: bool,
                 family: str = "tweedie"):
        self.spec, self.continuous, self.alpha, self.trend, self.family = spec, continuous, alpha, trend, family

    def _frame(self, d: pd.DataFrame, year=None) -> pd.DataFrame:
        f = self.spec.factors(d)
        for c in self.continuous:
            f[c] = d[c].to_numpy()
        if self.trend:
            f["t"] = (d.yr.to_numpy() if year is None else np.full(len(d), year)) - self.centre
        return f

    def fit(self, d: pd.DataFrame, y: np.ndarray, p: float) -> "GLM":
        self.centre = float(np.average(d.yr, weights=d.exposure))
        f = self._frame(d)
        self.levels = models.factor_levels(f, [c for c in f if c not in self.continuous + ["t"]])
        self.numeric = self.continuous + (["t"] if self.trend else [])
        X, self.names = models.design_matrix(f, self.levels, self.numeric)
        e = d.exposure.to_numpy()
        self.model = (models.fit_poisson(X, y, e, self.alpha) if self.family == "poisson"
                      else models.fit_tweedie(X, y, e, p, self.alpha))
        self.slope = float(self.model.coef_[-1]) if self.trend else 0.0
        return self

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        X, _ = models.design_matrix(self._frame(d, year), self.levels, self.numeric)
        return self.model.predict(X) * d.exposure.to_numpy()


class XGBSmall:
    """XGBoost on raw features with the backfitted year trend of A15."""

    def __init__(self, numeric, categorical, train, params, n_trees, xc, n_iter):
        self.numeric, self.params, self.n_trees, self.xc, self.n_iter = numeric, params, n_trees, xc, n_iter
        self.cats = xgb_models.category_levels(train, categorical)

    def fit(self, d: pd.DataFrame, y: np.ndarray, p: float) -> "XGBSmall":
        self.base = y.sum() / d.exposure.sum()
        self.booster, self.slope, self.centre, self.history = xgb_models.fit_with_trend(
            xgb_models.matrix(d, self.numeric, self.cats), y, d.exposure.to_numpy(), d.yr.to_numpy(), self.base, p,
            self.params, self.n_trees, self.xc["seed"], self.xc["nthread"], n_iter=self.n_iter)
        return self

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        yr = d.yr.to_numpy() if year is None else np.full(len(d), year)
        return xgb_models.predict(self.booster, xgb_models.matrix(d, self.numeric, self.cats), d.exposure.to_numpy(),
                                  self.base, extra=self.slope * (yr - self.centre))


class LineModel:
    """Small-claim model plus, for split lines, the large-claim GLM scaled to the Pareto mean."""

    def __init__(self, small, excess=None, tail=None):
        self.small, self.excess, self.tail = small, excess, tail

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        pred = self.small.predict(d, year)
        if self.excess is not None:
            yr = d.yr.to_numpy() if year is None else np.full(len(d), year)
            trend = np.exp(self.small.slope * (yr - self.excess.centre))
            pred = pred + self.excess.predict(d) * self.tail["adjustment"] * trend
        return pred


class Line:
    def __init__(self, name: str, df: pd.DataFrame, claims: pd.DataFrame, cfg: dict):
        self.name, self.cfg, self.spec = name, cfg, cfg["glm_lines"]["models"][name]
        self.df = df.assign(y_full=df.claim_cost_dev + df.excess_dev)
        self.claims = claims
        self.train, self.test = self.df[self.df.src == "train"], self.df[self.df.src == "test"]
        self.p = cfg["model"]["tweedie_power"]
        self.expected = {}  # window -> E columns by row_id (sql/13)

    def rows(self, years) -> pd.DataFrame:
        return self.train[self.train.yr.isin(years)]

    def targets(self, d: pd.DataFrame, threshold) -> tuple[np.ndarray, np.ndarray]:
        if threshold is None:
            return d.y_full.to_numpy(), np.zeros(len(d))
        t = glm_lines.split_targets(self.claims[self.claims.row_id.isin(d.row_id)], threshold)
        t = t.reindex(d.row_id.to_numpy(), fill_value=0)
        return t.small_dev.to_numpy(), t.excess_dev.to_numpy()

    def glm_spec(self, part: dict, d: pd.DataFrame) -> tuple[glm_lines.FactorSpec, list[str]]:
        spec = glm_lines.FactorSpec(d, part["numeric"], resolve(self.cfg, part["categorical"]),
                                    n_bands=part["n_bands"])
        return spec, resolve(self.cfg, part.get("continuous", []))

    def freq_pred(self, years) -> pd.DataFrame:
        """Expected claim counts for every row from a Poisson GLM without history, fitted on `years`."""
        part = self.spec["freq"] if isinstance(self.spec["freq"], dict) else self.spec
        d = self.rows(years)
        spec, cont = self.glm_spec(part, d)
        m = GLM(spec, cont, self.cfg["glm_lines"]["freq_alpha"], trend=False, family="poisson")
        m.fit(d, d.n_claims.to_numpy(dtype=float), self.p)
        return pd.DataFrame({"line": self.name, "row_id": self.df.row_id.to_numpy(), "mu": m.predict(self.df)})

    def tune_trees(self, threshold) -> tuple[int, dict]:
        """Early-stopped tree count for the XGBoost small-claim model on the tuning split."""
        xc, sp = self.cfg["xgboost"], self.spec
        years, val_year = WINDOWS["tune"]
        d, v = self.rows(years), self.rows([val_year])
        y, yv = self.targets(d, threshold)[0], self.targets(v, threshold)[0]
        lm = self.cfg["lines"]["models"][self.name]
        cats = xgb_models.category_levels(self.train, lm["categorical"])
        base = y.sum() / d.exposure.sum()
        params = {**xc["fixed_params"], "max_depth": sp["max_depth"],
                  "min_child_weight": sp["min_child_rows"] * median_hessian(y, d.exposure.to_numpy(), base, self.p)}
        b = xgb_models.fit(xgb_models.matrix(d, lm["numeric"], cats), y, d.exposure.to_numpy(), base, self.p, params,
                           xc["max_rounds"], xc["seed"], xc["nthread"],
                           val=(xgb_models.matrix(v, lm["numeric"], cats), yv, v.exposure.to_numpy()),
                           early_stopping=xc["early_stopping_rounds"])
        return b.best_iteration + 1, params

    def build(self, years, threshold, alpha, trees=None, trend=True) -> LineModel:
        sp, gl = self.spec, self.cfg["glm_lines"]
        d = self.rows(years)
        y_small, y_excess = self.targets(d, threshold)
        if sp["small"] == "glm":
            spec, cont = self.glm_spec(sp, d)
            small = GLM(spec, cont, alpha, trend).fit(d, y_small, self.p)
        else:
            lm = self.cfg["lines"]["models"][self.name]
            n_iter = self.cfg["lines"]["trend_iterations"] if trend else 0
            small = XGBSmall(lm["numeric"], lm["categorical"], self.train, trees[1], trees[0], self.cfg["xgboost"],
                             n_iter).fit(d, y_small, self.p)
        if threshold is None:
            return LineModel(small)
        spec, _ = self.glm_spec(sp["excess"], d)
        excess = GLM(spec, [], gl["excess_alpha"], trend=False).fit(d, y_excess, self.p)
        tail = glm_lines.gpd_tail(self.claims[self.claims.row_id.isin(d.row_id)], threshold)
        return LineModel(small, excess, tail)

    def credibility(self, d: pd.DataFrame, window: str, ks: dict) -> np.ndarray:
        e = self.expected[window].reindex(d.row_id.to_numpy())
        cf = np.ones(len(d))
        for grp, k in ks.items():
            n_col, e_col = glm_lines.HISTORY_GROUPS[grp]
            cf *= glm_lines.credibility_factor(d[n_col].to_numpy(dtype=float), e[e_col].to_numpy(), k)
        return cf

    def predict(self, model: LineModel, window: str, ks: dict, d: pd.DataFrame, year=None) -> np.ndarray:
        """Base prediction times the credibility factor, rebalanced so the fitted rows keep their total."""
        fit = self.rows(WINDOWS[window][0])
        base_fit = model.predict(fit)
        rebalance = base_fit.sum() / (base_fit * self.credibility(fit, window, ks)).sum()
        return model.predict(d, year) * self.credibility(d, window, ks) * rebalance

    def score(self, pred: np.ndarray, d: pd.DataFrame) -> dict:
        y_full, y_rep = d.y_full.to_numpy(), d.claim_cost.to_numpy()
        return {"d2_vs_developed": models.d2_tweedie(y_full, pred, self.p),
                "d2_vs_reported": models.d2_tweedie(y_rep, pred, self.p),
                "pred_over_developed": pred.sum() / y_full.sum()}


def expected_history(con, lines: dict, window: str, sql: str) -> None:
    """E for every row of every line from the window's frequency models (sql/13), stored on each Line."""
    years = WINDOWS[window][0]
    mu = pd.concat([line.freq_pred(years) for line in lines.values()])
    con.register("freq_pred_df", mu)
    con.execute("CREATE OR REPLACE TABLE freq_pred AS SELECT * FROM freq_pred_df")
    con.execute(sql)
    e = con.execute("SELECT * FROM history_expected").df()
    for name, line in lines.items():
        line.expected[window] = e[e.line == name].set_index("row_id")[["e_own", "e_home", "e_cross"]]


def run_line(line: Line) -> dict:
    gl, sp = line.cfg["glm_lines"], line.spec
    out = {"tuning": [], "credibility": [], "backtest": []}
    years, val_year = WINDOWS["tune"]
    val = line.rows([val_year])
    thresholds = gl["thresholds"] if sp["split"] else [None]
    alphas = gl["alpha_grid"] if sp["small"] == "glm" else [None]
    trees = {}
    for threshold, alpha in itertools.product(thresholds, alphas):
        if sp["small"] == "xgb" and threshold not in trees:
            trees[threshold] = line.tune_trees(threshold)
        s = line.score(line.build(years, threshold, alpha, trees.get(threshold), trend=False).predict(val), val)
        out["tuning"].append({"line": line.name, "threshold": threshold, "alpha": alpha,
                              "n_trees": trees.get(threshold, (None,))[0], "d2_2023_developed": s["d2_vs_developed"],
                              "pred_over_developed": s["pred_over_developed"]})
        print(f"  {line.name} tune threshold {threshold} alpha {alpha}: D2 {s['d2_vs_developed']:.4f}", flush=True)
    best = max(out["tuning"], key=lambda r: r["d2_2023_developed"])
    threshold, alpha = best["threshold"], best["alpha"]
    tree_spec = trees.get(threshold)

    # Credibility k per group, one group at a time, on the same tuning split.
    base = line.build(years, threshold, alpha, tree_spec, trend=False)
    ks = {g: np.inf for g in GROUPS[line.name]}
    for grp in GROUPS[line.name]:
        scores = {}
        for k in gl["credibility_k_grid"] + [np.inf]:
            scores[k] = line.score(line.predict(base, "tune", {**ks, grp: k}, val), val)["d2_vs_developed"]
            out["credibility"].append({"line": line.name, "group": grp, "k": k, "d2_2023_developed": scores[k]})
        ks[grp] = max(scores, key=scores.get)
        print(f"  {line.name} credibility {grp}: k {ks[grp]} (D2 {scores[ks[grp]]:.4f}, "
              f"no history {scores[np.inf]:.4f})", flush=True)

    b_years, b_year = WINDOWS["backtest"]
    b_val = line.rows([b_year])
    variants = {"chosen": {}, "no history": {"ks": {g: np.inf for g in ks}}, "no trend": {"trend": False}}
    if sp["split"]:
        variants["no split"] = {"threshold": None}
    for name, change in variants.items():
        th = change.get("threshold", threshold)
        ts = line.tune_trees(th) if sp["small"] == "xgb" and th != threshold else tree_spec
        trend = change.get("trend", True)
        m = line.build(b_years, th, alpha, ts, trend=trend)
        pred = line.predict(m, "backtest", change.get("ks", ks), b_val, year=b_year if trend else None)
        out["backtest"].append({"line": line.name, "variant": name, **line.score(pred, b_val)})
        if name == "chosen":
            out["backtest_pred"] = b_val.assign(mu=pred)
        print(f"  {line.name} backtest {name}: D2 {out['backtest'][-1]['d2_vs_developed']:.4f}, "
              f"p/a {out['backtest'][-1]['pred_over_developed']:.3f}", flush=True)

    f_years, f_year = WINDOWS["final"]
    final = line.build(f_years, threshold, alpha, tree_spec)
    pred = line.predict(final, "final", ks, line.test, year=f_year)
    cf = line.credibility(line.test, "final", ks)
    last = line.rows([2024])
    rate_2024 = last.y_full.sum() / last.exposure.sum()
    out["summary"] = {"line": line.name, "small_model": sp["small"], "threshold": threshold, "alpha": alpha,
                      "n_trees": tree_spec[0] if tree_spec else None, "trend_pa": np.expm1(final.small.slope),
                      **{f"k_{g}": k for g, k in ks.items()},
                      "credibility_min": cf.min(), "credibility_max": cf.max(),
                      "tail_shape": final.tail["shape"] if final.tail else None,
                      "tail_adjustment": final.tail["adjustment"] if final.tail else None,
                      "rate_2025": pred.sum() / line.test.exposure.sum(), "developed_rate_2024": rate_2024,
                      "rate_2025_over_2024": pred.sum() / line.test.exposure.sum() / rate_2024,
                      "pred_min": pred.min(), "pred_max": pred.max()}
    if final.tail:
        out["tail"] = {"line": line.name, **final.tail}
    out["pred"] = pd.Series(pred, index=line.test.row_id.to_numpy())
    return out


def residual_correlation(results: dict, p: float) -> pd.DataFrame:
    """Same-year correlation of buildings and contents Pearson residuals (2024 backtest), by customer and address."""
    agg = {}
    for name in ("buildings", "contents"):
        d = results[name]["backtest_pred"]
        agg[name] = d.groupby(["customer_id", "address_id"]).agg(y=("y_full", "sum"), mu=("mu", "sum"))
    j = agg["buildings"].join(agg["contents"], lsuffix="_b", rsuffix="_c", how="inner")
    r_b = (j.y_b - j.mu_b) / np.sqrt(j.mu_b ** p)
    r_c = (j.y_c - j.mu_c) / np.sqrt(j.mu_c ** p)
    return pd.DataFrame([{"n_customer_addresses": len(j), "pearson_residual_corr": r_b.corr(r_c),
                          "claim_indicator_corr": (j.y_b > 0).astype(float).corr((j.y_c > 0).astype(float))}])


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    gl, sc = cfg["glm_lines"], cfg["submission"]
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / RUNNER
    try:
        sample_path = common.PACK_ROOT / sc["sample_file"]
        rec["sample_sha256"] = common.sha256_file(sample_path)
        sample = pd.read_csv(sample_path, dtype={"row_id": str})
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        rec["sql_sha256"] = common.run_sql_files(con, cfg)
        for rel in cfg["lines"]["sql_files"] + gl["sql_files"]:
            rec["sql_sha256"][rel] = common.sha256_file(common.PACK_ROOT / rel)
            con.execute((common.PACK_ROOT / rel).read_text())
        rec["sql_sha256"][gl["expected_sql"]] = common.sha256_file(common.PACK_ROOT / gl["expected_sql"])
        expected_sql = (common.PACK_ROOT / gl["expected_sql"]).read_text()
        outputs = common.export_sql_evidence(con, out_dir, prefixes=("ev_12",))

        lines = {}
        for name in gl["models"]:
            table = cfg["lines"]["models"][name]["table"]
            df = con.execute(f"""SELECT m.*, h.* EXCLUDE (line, row_id)
                                 FROM {table} m JOIN claim_history h ON h.line = '{name}' AND h.row_id = m.row_id
                                 ORDER BY m.row_id""").df()
            if name == "contents":  # contents_model_dev has no customer or address; the correlation check needs them
                keys = con.execute("SELECT row_id, customer_id, address_id FROM contents_rows").df()
                df = df.merge(keys, on="row_id")
            claims = con.execute(f"SELECT row_id, net_cost, dev_factor FROM line_claims_dev WHERE line = '{name}'").df()
            lines[name] = Line(name, df, claims, cfg)
        for window in WINDOWS:
            expected_history(con, lines, window, expected_sql)
        con.close()

        results = {}
        for name, line in lines.items():
            print(f"{name}: {len(line.df)} rows", flush=True)
            results[name] = run_line(line)

        def cat(key):
            parts = [r[key] for r in results.values() if key in r]
            return pd.concat([pd.DataFrame(x if isinstance(x, list) else [x]) for x in parts])

        summary = cat("summary")
        corr = residual_correlation(results, cfg["model"]["tweedie_power"])
        outputs += [write_csv(cat("tuning"), out_dir / "ev_g1_tuning.csv"),
                    write_csv(cat("credibility"), out_dir / "ev_g2_credibility_tuning.csv"),
                    write_csv(cat("backtest"), out_dir / "ev_g3_backtest_2024.csv"),
                    write_csv(cat("tail"), out_dir / "ev_g4_large_claim_tail.csv"),
                    write_csv(corr, out_dir / "ev_g5_buildings_contents_residual_corr.csv"),
                    write_csv(summary, out_dir / "ev_g6_summary.csv")]

        prefixes = {name: cfg["lines"]["models"][name]["prefix"] for name in results}
        sub = make_submission.build_submission(sample, {prefixes[k]: r["pred"] for k, r in results.items()})
        path = common.PACK_ROOT / gl["out_file"].format(date=dt.date.today().strftime("%Y%m%d"))
        path.parent.mkdir(parents=True, exist_ok=True)
        sub.round(6).to_csv(path, index=False)

        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o)
                                 for o in outputs + [path]}
        rec["metrics"] = {"summary": summary.round(6).to_dict("records"),
                          "backtest": cat("backtest").round(6).to_dict("records"),
                          "residual_corr": corr.round(6).to_dict("records")}
        rec["status"] = "success"
        print(cat("backtest").round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))
        print(corr.round(4).to_string(index=False))
        print(f"Submission: {path.relative_to(common.PACK_ROOT)} ({len(sub)} rows)")
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        import duckdb, scipy, sklearn, xgboost  # noqa: E401  versions recorded for the audit trail
        rec["versions"] = {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "pandas": pd.__version__,
                           "numpy": np.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                           "xgboost": xgboost.__version__}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
