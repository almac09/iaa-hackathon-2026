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
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cann  # noqa: E402
import common  # noqa: E402
import glm_lines  # noqa: E402
import make_submission  # noqa: E402
import models  # noqa: E402
import significance  # noqa: E402
import xgb_models  # noqa: E402
from run_lines import median_hessian  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "glm_lines"
WINDOWS = {"tune": ([2021, 2022], 2023), "backtest": ([2021, 2022, 2023], 2024),
           "final": ([2021, 2022, 2023, 2024], 2025)}
GROUPS = {"buildings": ["own", "home", "cross"], "contents": ["own", "home", "cross"], "motor": ["own", "cross"]}
EVAL_WINDOWS = ("tune", "backtest")
# Days of each row in each calendar month (rows never cross a year end), for seasonal exposure (D-22).
MONTH_DAYS = ", ".join(
    f"greatest(0, date_diff('day', greatest(period_start, make_date(yr, {m}, 1)), "
    f"least(period_end, last_day(make_date(yr, {m}, 1)))) + 1) AS d{m}" for m in range(1, 13))


def resolve(cfg: dict, v) -> list[str]:
    return cfg["model"][v.split(".")[1]] if isinstance(v, str) else list(v)


class GLM:
    """Tweedie (or Poisson) GLM on banded factors with a log-exposure offset and, optionally, a linear year term."""

    def __init__(self, spec: glm_lines.FactorSpec, continuous: list[str], alpha: float, trend: bool,
                 family: str = "tweedie"):
        self.spec, self.continuous, self.alpha, self.trend, self.family = spec, continuous, alpha, trend, family

    def _frame(self, d: pd.DataFrame, year=None) -> pd.DataFrame:
        f = self.spec.factors(d)
        basis = self.spec.basis(d)
        for c in basis:
            f[c] = basis[c].to_numpy()
        self.basis_columns = list(basis.columns)
        for c in self.continuous:
            f[c] = d[c].to_numpy()
        if self.trend:
            f["t"] = (d.yr.to_numpy() if year is None else np.full(len(d), year)) - self.centre
        return f

    def fit(self, d: pd.DataFrame, y: np.ndarray, p: float) -> "GLM":
        self.centre = float(np.average(d.yr, weights=d.exposure))
        f = self._frame(d)
        self.numeric = self.basis_columns + self.continuous + (["t"] if self.trend else [])
        self.levels = models.factor_levels(f, [c for c in f if c not in self.numeric])
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
    """Small-claim model (with its CANN correction, if any) plus, for split lines, the large-claim GLM scaled to the
    Pareto mean."""

    def __init__(self, small, excess=None, tail=None, cann=None):
        self.small, self.excess, self.tail, self.cann = small, excess, tail, cann

    def predict(self, d: pd.DataFrame, year=None) -> np.ndarray:
        pred = self.small.predict(d, year)
        if self.cann is not None:
            pred = pred * np.exp(self.cann.correction(d))
        if self.excess is not None:
            yr = d.yr.to_numpy() if year is None else np.full(len(d), year)
            trend = np.exp(self.small.slope * (yr - self.excess.centre))
            pred = pred + self.excess.predict(d) * self.tail["adjustment"] * trend
        return pred


class Line:
    def __init__(self, name: str, df: pd.DataFrame, claims: pd.DataFrame, cfg: dict):
        self.name, self.cfg, self.spec = name, cfg, cfg["glm_lines"]["models"][name]
        self.df = df.assign(y_full=df.claim_cost_dev + df.excess_dev,
                            exposure_plain=df.exposure)  # reported rates always use plain exposure
        self.claims = claims
        self.train, self.test = self.df[self.df.src == "train"], self.df[self.df.src == "test"]
        self.p = cfg["model"]["tweedie_power"]
        self.expected = {}  # window -> E columns by row_id (sql/13)
        self.month_days, self.claim_months, self.weights = None, None, {}  # seasonal exposure (D-22), if used

    def use_window(self, window: str) -> None:
        """For a line with seasonal exposure (D-22): set every row's exposure to its days weighted by the month's
        claim frequency relative to the year, estimated on the window's fitting years only. A no-op otherwise.
        The fitting years keep their total exposure."""
        if self.month_days is None:
            return
        years = WINDOWS[window][0]
        fit = (self.df.src.eq("train") & self.df.yr.isin(years)).to_numpy()
        days = self.month_days[fit].sum(0)
        n = (self.claim_months[self.claim_months.yr.isin(years)].groupby("m").n.sum()
             .reindex(range(1, 13), fill_value=0).to_numpy())
        rate = np.divide(n, days, out=np.full(12, n.sum() / days.sum()), where=days > 0)  # empty month: no change
        w = rate / (n.sum() / days.sum())
        self.df["exposure"] = self.month_days @ w / 365
        self.train, self.test = self.df[self.df.src == "train"], self.df[self.df.src == "test"]
        assert abs(self.df.exposure[fit].sum() / self.df.exposure_plain[fit].sum() - 1) < 1e-9
        self.weights[window] = w

    def rows(self, years) -> pd.DataFrame:
        return self.train[self.train.yr.isin(years)]

    def targets(self, d: pd.DataFrame, threshold) -> tuple[np.ndarray, np.ndarray]:
        if threshold is None:
            return d.y_full.to_numpy(), np.zeros(len(d))
        t = glm_lines.split_targets(self.claims[self.claims.row_id.isin(d.row_id)], threshold)
        t = t.reindex(d.row_id.to_numpy(), fill_value=0)
        return t.small_dev.to_numpy(), t.excess_dev.to_numpy()

    def glm_spec(self, part: dict, d: pd.DataFrame) -> tuple[glm_lines.FactorSpec, list[str]]:
        smooth = part["numeric"] if part.get("smooth") == "all" else part.get("smooth", [])
        spec = glm_lines.FactorSpec(d, part["numeric"], resolve(self.cfg, part["categorical"]),
                                    n_bands=part["n_bands"], smooth=smooth, n_knots=part.get("n_knots", 4))
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

    def build(self, years, threshold, alpha, trees=None, trend=True, a16=False) -> LineModel:
        """`a16` rebuilds with the settings before A17 (no smoothing, no added motor factors), for significance."""
        sp, gl = ({**self.spec, **self.spec["a16_override"]} if a16 else self.spec), self.cfg["glm_lines"]
        if a16 and "excess" in self.spec.get("a16_override", {}):
            # Keep post-A17 additions (A28 interactions) so the variant reverts only the A17 settings.
            sp["excess"] = {**sp["excess"], "continuous": self.spec["excess"].get("continuous", [])}
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
        spec, cont = self.glm_spec(sp["excess"], d)  # continuous terms only where the excess spec lists them (A28)
        excess = GLM(spec, cont, gl["excess_alpha"], trend=False).fit(d, y_excess, self.p)
        tail = glm_lines.gpd_tail(self.claims[self.claims.row_id.isin(d.row_id)], threshold)
        return LineModel(small, excess, tail)

    def add_cann(self, model: LineModel, years, threshold) -> LineModel:
        """The model with a CANN ensemble correcting its small-claim part, trained on the window's small claims."""
        d = self.rows(years)
        ref = self.cfg["glm_lines"]["cann"]["features"][self.name]
        node = self.cfg
        for key in ref.split("."):
            node = node[key]
        ens = cann.CANNEnsemble(resolve(self.cfg, node["numeric"]), resolve(self.cfg, node["categorical"]),
                                self.cfg["glm_lines"]["cann"], self.p)
        ens.fit(d, self.targets(d, threshold)[0], model.small.predict(d))
        return LineModel(model.small, model.excess, model.tail, cann=ens)

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
    line.use_window("tune")
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

    # CANN (A20): adopted only if it improves both out-of-time years on the full pipeline and, under D-23 (A22), its
    # two-year mean gain is significant.
    trees_by_threshold = {threshold: tree_spec}

    def trees_for(th):
        if sp["small"] != "xgb":
            return None
        if th not in trees_by_threshold:
            trees_by_threshold[th] = line.tune_trees(th)
        return trees_by_threshold[th]

    base, with_cann = {}, {}
    out["cann"] = []
    bs = gl["bootstrap"]
    for w in EVAL_WINDOWS:
        yrs, yv = WINDOWS[w]
        line.use_window(w)
        v = line.rows([yv])
        base[w] = line.build(yrs, threshold, alpha, tree_spec)
        p_without = line.predict(base[w], w, ks, v, year=yv)
        row = {"line": line.name, "score_year": yv, "d2_without_cann": line.score(p_without, v)["d2_vs_developed"]}
        if gl["cann"]["enabled"]:
            with_cann[w] = line.add_cann(base[w], yrs, threshold)
            p_with = line.predict(with_cann[w], w, ks, v, year=yv)
            row["d2_with_cann"] = line.score(p_with, v)["d2_vs_developed"]
            diff = significance.paired_cluster_bootstrap(v.y_full.to_numpy(), p_with, p_without,
                                                         v[bs["cluster"]].to_numpy(), line.p, bs["n"], bs["seed"])
            summ = significance.summarise(diff, row["d2_with_cann"] - row["d2_without_cann"])
            row.update({"ci_low_95": summ["ci_low_95"], "ci_high_95": summ["ci_high_95"]})
        out["cann"].append(row)
        print(f"  {line.name} CANN {yv}: without {row['d2_without_cann']:.5f}, "
              f"with {row.get('d2_with_cann', np.nan):.5f}", flush=True)
    two_year = bool(with_cann) and all(r["d2_with_cann"] > r["d2_without_cann"] for r in out["cann"])
    adopt_cann = two_year
    if with_cann:
        mean, lo, hi = significance.two_year_interval([r["d2_with_cann"] - r["d2_without_cann"] for r in out["cann"]],
                                                      [r["ci_low_95"] for r in out["cann"]],
                                                      [r["ci_high_95"] for r in out["cann"]])
        adopt_cann = two_year and (lo > 0 or not gl["cann"]["require_significance"])
        print(f"  {line.name} CANN two-year mean {mean:+.5f} ({lo:+.5f}, {hi:+.5f}): "
              f"{'adopted' if adopt_cann else 'not adopted'}", flush=True)
    for r in out["cann"]:
        r["passes_two_year_rule"] = two_year
        if with_cann:
            r["two_year_mean"], r["two_year_ci_low_95"], r["two_year_ci_high_95"] = mean, lo, hi
        r["adopted"] = adopt_cann

    def variant(w, **change):
        """The chosen model rebuilt with one component removed (CANN refitted on top when adopted)."""
        yrs = WINDOWS[w][0]
        th = change.get("threshold", threshold)
        m = line.build(yrs, th, alpha, trees_for(th), trend=change.get("trend", True), a16=change.get("a16", False))
        return line.add_cann(m, yrs, th) if adopt_cann else m

    # Per-row predictions on both years for the chosen model and each variant (significance, A20).
    out["preds"] = {}
    for w in EVAL_WINDOWS:
        yrs, yv = WINDOWS[w]
        line.use_window(w)
        v = line.rows([yv])
        chosen = with_cann[w] if adopt_cann else base[w]
        cols = {"chosen": line.predict(chosen, w, ks, v, year=yv)}
        if with_cann:
            cols["no CANN" if adopt_cann else "with CANN (not adopted)"] = line.predict(
                base[w] if adopt_cann else with_cann[w], w, ks, v, year=yv)
        cols["no history"] = line.predict(chosen, w, {g: np.inf for g in ks}, v, year=yv)
        cols["no trend"] = line.predict(variant(w, trend=False), w, ks, v)
        if sp["split"]:
            cols["no split"] = line.predict(variant(w, threshold=None), w, ks, v, year=yv)
        cols["no A17 changes"] = line.predict(variant(w, a16=True), w, ks, v, year=yv)
        out["preds"][w] = pd.DataFrame({"row_id": v.row_id.to_numpy(), "customer_id": v.customer_id.to_numpy(),
                                        "address_id": v.address_id.to_numpy(), "y_full": v.y_full.to_numpy(),
                                        "claim_cost": v.claim_cost.to_numpy(), **cols})
        for name, pred in cols.items():
            if w == "backtest":
                out["backtest"].append({"line": line.name, "variant": name, **line.score(pred, v)})
        print(f"  {line.name} variants scored for {yv}", flush=True)
    out["backtest_pred"] = out["preds"]["backtest"].assign(mu=out["preds"]["backtest"]["chosen"])

    # Level (A20): deviance-optimal scale on each year; adopted only if each year's scale improves the other year.
    p, ys = line.p, {w: out["preds"][w].y_full.to_numpy() for w in EVAL_WINDOWS}
    mus = {w: out["preds"][w]["chosen"].to_numpy() for w in EVAL_WINDOWS}
    k = {w: deviance_scale(ys[w], mus[w], p) for w in EVAL_WINDOWS}
    other = {"tune": "backtest", "backtest": "tune"}
    level = {"line": line.name, **{f"k_{WINDOWS[w][1]}": k[w] for w in EVAL_WINDOWS}}
    for w in EVAL_WINDOWS:
        yv = WINDOWS[w][1]
        level[f"d2_{yv}"] = models.d2_tweedie(ys[w], mus[w], p)
        level[f"d2_{yv}_with_k_from_{WINDOWS[other[w]][1]}"] = models.d2_tweedie(ys[w], mus[w] * k[other[w]], p)
    level["adopted"] = all(level[f"d2_{WINDOWS[w][1]}_with_k_from_{WINDOWS[other[w]][1]}"]
                           > level[f"d2_{WINDOWS[w][1]}"] for w in EVAL_WINDOWS)
    level["level_factor"] = float(np.exp(np.mean(np.log(list(k.values()))))) if level["adopted"] else 1.0

    f_years, f_year = WINDOWS["final"]
    line.use_window("final")
    final = line.build(f_years, threshold, alpha, tree_spec)
    if adopt_cann:
        final = line.add_cann(final, f_years, threshold)
    pred = line.predict(final, "final", ks, line.test, year=f_year) * level["level_factor"]

    # Level uncertainty (A20): development (bootstrap, ev_u1), trend and year-to-year noise, propagated to 2025.
    fitted = line.predict(final, "final", ks, line.train)
    by_year = pd.DataFrame({"yr": line.train.yr.to_numpy(), "y": line.train.y_full.to_numpy(),
                            "mu": fitted}).groupby("yr").sum()
    resid = np.log(by_year.y / by_year.mu)
    sigma_year = float(np.sqrt((resid ** 2).sum() / (len(resid) - 2)))
    dev = line.cfg["_dev_uncertainty"]
    cv_boot = {ay: float(dev[(dev.line == line.name) & (dev.ay == ay)].cv.iloc[0]) for ay in (2023, 2024)}
    cv = dict(cv_boot)
    if line.cfg["glm_lines"]["level"].get("include_mack"):  # A25: the larger of the bootstrap and Mack's error
        mk = line.cfg["_mack"]
        cv_mack = {ay: float(mk[(mk.line == line.name) & (mk.ay == ay)].mack_cv_of_ultimate.iloc[0]) for ay in (2023, 2024)}
        cv = {ay: max(cv_boot[ay], cv_mack[ay]) for ay in cv}
    # The OLS projection from 2021-24 to 2025 weights 2024 by 1.0 and 2023 by 0.5; the projection's own variance plus
    # a new year's noise is 2.5 sigma^2.
    sigma2 = cv[2024] ** 2 + 0.25 * cv[2023] ** 2 + 2.5 * sigma_year ** 2
    tilt_raw = float(np.exp(sigma2 / 2))
    # The tilt is a change of level like any other, so it faces the two-year rule: kept only if it improves D2 on
    # both backtest years (on top of any level factor).
    unc = {"line": line.name, "dev_cv_2024": cv[2024], "dev_cv_2023": cv[2023],
           "bootstrap_cv_2024": cv_boot[2024], "bootstrap_cv_2023": cv_boot[2023],
           "year_residuals": "; ".join(f"{y}: {r:+.4f}" for y, r in resid.items()),
           "sigma_year": sigma_year, "sigma_level_2025": float(np.sqrt(sigma2)), "tilt_computed": tilt_raw}
    for w in EVAL_WINDOWS:
        yv, base_mu = WINDOWS[w][1], mus[w] * level["level_factor"]
        unc[f"d2_{yv}"] = models.d2_tweedie(ys[w], base_mu, p)
        unc[f"d2_{yv}_with_tilt"] = models.d2_tweedie(ys[w], base_mu * tilt_raw, p)
    unc["tilt_adopted"] = all(unc[f"d2_{WINDOWS[w][1]}_with_tilt"] > unc[f"d2_{WINDOWS[w][1]}"] for w in EVAL_WINDOWS)
    tilt = tilt_raw if unc["tilt_adopted"] else 1.0
    pred = pred * tilt
    z = line.cfg["glm_lines"]["level"]["z90"]
    rate = pred.sum() / line.test.exposure_plain.sum()
    out["uncertainty"] = {**unc, "uncertainty_tilt": tilt,
                          "rate_2025": rate, "rate_2025_p05": rate * np.exp(-z * np.sqrt(sigma2)),
                          "rate_2025_p95": rate * np.exp(z * np.sqrt(sigma2))}
    out["level"] = level

    cf = line.credibility(line.test, "final", ks)
    last = line.rows([2024])
    rate_2024 = last.y_full.sum() / last.exposure_plain.sum()
    out["summary"] = {"line": line.name, "small_model": sp["small"], "cann": adopt_cann, "threshold": threshold,
                      "alpha": alpha, "n_trees": tree_spec[0] if tree_spec else None,
                      "trend_pa": np.expm1(final.small.slope), **{f"k_{g}": kv for g, kv in ks.items()},
                      "credibility_min": cf.min(), "credibility_max": cf.max(),
                      "tail_shape": final.tail["shape"] if final.tail else None,
                      "tail_adjustment": final.tail["adjustment"] if final.tail else None,
                      "level_factor": level["level_factor"], "uncertainty_tilt": tilt,
                      "rate_2025": rate, "developed_rate_2024": rate_2024, "rate_2025_over_2024": rate / rate_2024,
                      "pred_min": pred.min(), "pred_max": pred.max()}
    if final.tail:
        out["tail"] = {"line": line.name, **final.tail}
    out["pred"] = pd.Series(pred, index=line.test.row_id.to_numpy())
    if line.weights:
        out["seasonal"] = [{"line": line.name, "window": w, "fit_years": f"{WINDOWS[w][0][0]}-{WINDOWS[w][0][-1]}",
                            "month": m + 1, "relative_frequency": float(wt[m])}
                           for w, wt in line.weights.items() for m in range(12)]
    return out


def deviance_scale(y: np.ndarray, mu: np.ndarray, p: float) -> float:
    """The scale factor that minimises Tweedie deviance, sum(y mu^(1-p)) / sum(mu^(2-p)). For p = 1.5 this is not the
    ratio of totals, which is the deviance-optimal scale only under the canonical link."""
    return float((y * mu ** (1 - p)).sum() / (mu ** (2 - p)).sum())


def bootstrap_table(results: dict, cfg: dict) -> pd.DataFrame:
    """Paired customer-cluster bootstrap of D2(chosen) - D2(variant) for every variant, line and year."""
    bs, p, rows = cfg["glm_lines"]["bootstrap"], cfg["model"]["tweedie_power"], []
    for name, r in results.items():
        for w, d in r["preds"].items():
            y, mu = d.y_full.to_numpy(), d["chosen"].to_numpy()
            for col in [c for c in d.columns if c not in ("row_id", "customer_id", "address_id", "y_full", "claim_cost",
                                                           "chosen")]:
                alt = d[col].to_numpy()
                diff = significance.paired_cluster_bootstrap(y, mu, alt, d[bs["cluster"]].to_numpy(), p, bs["n"],
                                                             bs["seed"])
                obs = models.d2_tweedie(y, mu, p) - models.d2_tweedie(y, alt, p)
                rows.append({"line": name, "score_year": WINDOWS[w][1], "chosen_vs": col,
                             **significance.summarise(diff, obs)})
    return pd.DataFrame(rows)


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


def prepare(con, cfg: dict, rec: dict, out_dir: Path | None = None) -> tuple[dict, list[Path]]:
    """Run the SQL and build every Line with its claim history and expected claims for each window. Writes the SQL
    evidence tables when `out_dir` is given. Shared with python/gap_tests.py so both use the same data."""
    gl = cfg["glm_lines"]
    rec["sql_sha256"] = common.run_sql_files(con, cfg)
    for rel in cfg["lines"]["sql_files"] + gl["sql_files"]:
        rec["sql_sha256"][rel] = common.sha256_file(common.PACK_ROOT / rel)
        con.execute((common.PACK_ROOT / rel).read_text())
    rec["sql_sha256"][gl["expected_sql"]] = common.sha256_file(common.PACK_ROOT / gl["expected_sql"])
    expected_sql = (common.PACK_ROOT / gl["expected_sql"]).read_text()
    outputs = (common.export_sql_evidence(con, out_dir, prefixes=("ev_09", "ev_10", "ev_11", "ev_12"))
               if out_dir is not None else [])

    dev_path = common.PACK_ROOT / gl["level"]["uncertainty_evidence"]
    if not dev_path.exists():
        raise FileNotFoundError(f"{dev_path.relative_to(common.PACK_ROOT)} is missing: "
                                "run python/level_uncertainty.py first.")
    rec["dev_uncertainty_sha256"] = common.sha256_file(dev_path)
    cfg["_dev_uncertainty"] = pd.read_csv(dev_path)
    if gl["level"].get("include_mack"):
        mack_path = common.PACK_ROOT / gl["level"]["mack_evidence"]
        rec["mack_sha256"] = common.sha256_file(mack_path)
        cfg["_mack"] = pd.read_csv(mack_path)
    lines = {}
    for name in gl["models"]:
        table = cfg["lines"]["models"][name]["table"]
        df = con.execute(f"""SELECT m.*, h.* EXCLUDE (line, row_id)
                             FROM {table} m JOIN claim_history h ON h.line = '{name}' AND h.row_id = m.row_id
                             ORDER BY m.row_id""").df()
        if name == "contents":  # contents_model_dev has no customer, address or raw columns
            extra = ", ".join(["customer_id", "address_id"] + gl["models"][name].get("extra_columns", []))
            df = df.merge(con.execute(f"SELECT row_id, {extra} FROM contents_model").df(), on="row_id")
        claims = con.execute(f"SELECT row_id, net_cost, dev_factor FROM line_claims_dev WHERE line = '{name}'").df()
        lines[name] = Line(name, df, claims, cfg)
        if gl["models"][name].get("seasonal_exposure"):
            md = con.execute(f"SELECT row_id, {MONTH_DAYS} FROM line_rows WHERE line = '{name}'").df()
            md = md.set_index("row_id").reindex(df.row_id.astype(md.row_id.dtype).to_numpy())
            if md.isna().any().any():
                raise ValueError(f"{name}: rows missing from line_rows")
            line = lines[name]
            line.month_days = md.to_numpy(dtype=float)
            line.claim_months = con.execute(f"""SELECT yr, month(loss_date) AS m, count(*) AS n FROM line_claims
                                                WHERE line = '{name}' GROUP BY ALL""").df()
    for window in WINDOWS:
        expected_history(con, lines, window, expected_sql)
    return lines, outputs


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
        lines, outputs = prepare(con, cfg, rec, out_dir)
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
        boot = bootstrap_table(results, cfg)
        outputs += [write_csv(cat("tuning"), out_dir / "ev_g1_tuning.csv"),
                    write_csv(cat("credibility"), out_dir / "ev_g2_credibility_tuning.csv"),
                    write_csv(cat("backtest"), out_dir / "ev_g3_backtest_2024.csv"),
                    write_csv(cat("tail"), out_dir / "ev_g4_large_claim_tail.csv"),
                    write_csv(corr, out_dir / "ev_g5_buildings_contents_residual_corr.csv"),
                    write_csv(summary, out_dir / "ev_g6_summary.csv"),
                    write_csv(cat("cann"), out_dir / "ev_g7_cann_two_year.csv"),
                    write_csv(boot, out_dir / "ev_g8_bootstrap_significance.csv"),
                    write_csv(cat("level"), out_dir / "ev_g9_level_factor.csv"),
                    write_csv(cat("uncertainty"), out_dir / "ev_g10_level_uncertainty.csv"),
                    write_csv(cat("seasonal"), out_dir / "ev_g14_seasonal_exposure_weights.csv")]

        prefixes = {name: cfg["lines"]["models"][name]["prefix"] for name in results}
        sub = make_submission.build_submission(sample, {prefixes[k]: r["pred"] for k, r in results.items()})
        path = common.PACK_ROOT / gl["out_file"].format(date=dt.date.today().strftime("%Y%m%d"))
        path.parent.mkdir(parents=True, exist_ok=True)
        sub.round(6).to_csv(path, index=False)

        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o)
                                 for o in outputs + [path]}
        rec["metrics"] = {"summary": summary.round(6).to_dict("records"),
                          "backtest": cat("backtest").round(6).to_dict("records"),
                          "residual_corr": corr.round(6).to_dict("records"),
                          "significance": boot.round(6).to_dict("records"),
                          "level": cat("level").round(6).to_dict("records")}
        rec["status"] = "success"
        print(cat("backtest").round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))
        print(corr.round(4).to_string(index=False))
        print(boot.round(5).to_string(index=False))
        print(cat("level").round(5).to_string(index=False))
        print(cat("uncertainty").round(4).to_string(index=False))
        print(f"Submission: {path.relative_to(common.PACK_ROOT)} ({len(sub)} rows)")
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Versions from package metadata: importing torch here would load its OpenMP runtime next to XGBoost's (A20).
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "scipy", "torch",
                                                                     "xgboost")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
