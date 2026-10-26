"""Statistical tests of the final model's GLMs, asked for after A28 (audit A29).

  S1  Documentation of the final model, fitted on 2021-24 as for the 2025 prediction.
      - Type III tests for every term of every GLM component. Each term is dropped in turn and the model refitted.
        The quasi-likelihood F test is F = (dD / df) / phi, with phi from the full model's Pearson statistic,
        because the Tweedie dispersion is estimated, and df the number of the term's columns that are estimable
        given the other terms (some factors are exactly aliased with others; the production fits are held only
        by their small ridge penalty in those directions). Beside it is a score test of the term at the reduced
        fit, which is valid under aliasing and where a level has no claims. Its statistic is the smaller of the
        cluster-robust one (sandwich, by customer) and the model-based one with the effective dispersion
        trace(G+ B) / rank: a sandwich alone collapses where the term's rows have few claims.
      - Chi-square tests of actual against expected by level of every model factor on both backtest years. The
        production predictions are rescaled to the year's actual total, so only the shape by level is tested,
        with the models' own Tweedie variance phi * sum(mu^p) and phi from the scored year's Pearson statistic.
  S2  Simplification, tested out of time.
      - Level merging. In each fitting window the levels of each categorical factor of each GLM component are
        merged greedily while the Wald statistic for equal coefficients is below 2: Takeuchi's criterion (AIC with
        the effective dispersion) for one parameter. Each pass re-expresses the model in the merges made so far,
        under the quadratic approximation; a merge that is not estimable there (a level aliased with another
        factor's) leaves the fit unchanged and is made freely. Adjacent levels only where the levels are numbers
        or bands, any pair otherwise. The GLMs are refitted with the merged levels and scored on the backtest
        years.
      - Dropping terms. Each term of each GLM component is dropped in turn (a factor, a spline, a column or the
        trend) and scored the same way.
  S3  Pairwise interaction screen. On 2021-22 only, so that both backtest years stay unseen, every pair of terms
      in each GLM component (the year trend included) is screened by the score test of S1 at the production fit.
      Buildings' small-claim XGBoost is screened on its own residuals, with each pair's main effects as nuisance
      terms. Pairs whose interaction is already a model feature (the A28 peril interactions) are recorded but not
      screened. Benjamini-Hochberg q-values at 5% are reported within each model. The three strongest pairs per
      model (smallest p on 2021-22) are added as fixed features, defined on 2021-22, and scored out of time,
      whether or not they pass Benjamini-Hochberg: a positive control (buildings' XGBoost refitted without the
      A28 interaction features, screening the two pairs they came from) showed in the smoke run that a gate at
      that level cannot pass even interactions known to gain about 0.001 out of time. Holm at the scoring step
      controls the error across the three.
Adoption (S2, S3), fixed before the runs: both backtest years improve and the two-year mean gain is significant
(D-23); for dropped terms and interactions also Holm across the candidates scored for a model. Buildings XGBoost
comparisons average four seeds at a fixed tree count; where only a GLM component changes, the other components
are shared, so the comparison is exactly paired.

Usage (from the pack root, after python/run_glm_lines.py):
  python python/stat_tests.py --tests S1 --run-label "A29 statistical tests: S1"
"""
from __future__ import annotations

import copy
import datetime as dt
import itertools
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import final_tests as ft  # noqa: E402
import glm_lines  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
from combined_tests import window_preds  # noqa: E402
from gap_tests import compare, settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "stat_tests"
OUT = common.PACK_ROOT / "audit" / "evidence" / "glm_lines"
EVAL = rgl.EVAL_WINDOWS
SPECIAL = ("MISSING", "OTHER")
MERGE_WALD = 2.0  # S2: merge while the Wald statistic for equal coefficients is below 2 per parameter
FDR = 0.05  # S3: Benjamini-Hochberg level within each model
TOP = 3  # S3: survivors per model scored out of time
SCREEN_YEARS = [2021, 2022]
# Interaction features already in the models (A28, D-29) and the terms they are built from (sql/11).
MODELLED = {"x_clay_dry": {"soil_type", "dry_summer_index"}, "x_fz3_rain": {"flood_zone", "rainfall_mm"}}
CHUNK = 200_000
EMPTY = {"df": 0, "score_robust": 0.0, "score_model": 0.0, "score": 0.0, "p": 1.0}  # nothing left to test


# ---- Linear algebra on large designs -------------------------------------------------------------------------

def gram(X: np.ndarray, w: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """X' diag(w) Y, in row chunks (Y = X by default)."""
    Y = X if Y is None else Y
    out = np.zeros((X.shape[1], Y.shape[1]))
    for a in range(0, len(X), CHUNK):
        out += (X[a:a + CHUNK] * w[a:a + CHUNK, None]).T @ Y[a:a + CHUNK]
    return out


def cluster_order(clusters) -> tuple[np.ndarray, np.ndarray]:
    """Rows sorted by cluster, and where each cluster starts in that order."""
    codes = pd.factorize(pd.Series(clusters))[0]
    order = np.argsort(codes, kind="stable")
    return order, np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]


def meat(Z: np.ndarray, r: np.ndarray, order: np.ndarray, starts: np.ndarray, step: int = 100_000) -> np.ndarray:
    """Sum over clusters of s s', where s sums z_i r_i over the cluster's rows: the cluster-robust 'meat'."""
    edges = np.r_[starts, len(order)]
    B = np.zeros((Z.shape[1], Z.shape[1]))
    for a in range(0, len(starts), step):
        b = min(a + step, len(starts))
        idx = order[edges[a]:edges[b]]
        S = np.add.reduceat(Z[idx] * r[idx, None], starts[a:b] - edges[a], axis=0)
        B += S.T @ S
    return B


def score_test(X, Gp, w, r, Z, order, starts, phi) -> dict:
    """Score test for adding the columns Z to a fitted Tweedie log-link model with design X (Gp the pseudo-inverse
    of X' W X, weights mu^(2-p), score residuals (y - mu) mu^(1-p)). Z is made orthogonal to X in the unpenalised
    information metric and reduced to its estimable directions, so columns in X's span, or aliased among
    themselves, carry no degrees of freedom. The statistic is the smaller of the cluster-robust one and the
    model-based one with dispersion phi."""
    Zt = Z - X @ (Gp @ gram(X, w, Z))
    vals, vecs = np.linalg.eigh(gram(Zt, w))
    scale = np.diag(gram(Z, w)).max()
    keep = vals > 1e-8 * scale
    if not keep.any():
        return EMPTY
    Zq = Zt @ (vecs[:, keep] / np.sqrt(vals[keep]))  # orthonormal in the W metric: model information is phi * I
    U = Zq.T @ r
    robust = float(U @ np.linalg.pinv(meat(Zq, r, order, starts)) @ U)
    model = float(U @ U / phi)
    df = int(keep.sum())
    stat = min(robust, model)
    return {"df": df, "score_robust": robust, "score_model": model, "score": stat, "p": float(stats.chi2.sf(stat, df))}


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n, o = len(p), np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ---- GLM components ------------------------------------------------------------------------------------------

class GLMPlus(rgl.GLM):
    """The production GLM, optionally with merged factor levels (S2): `merges` maps factor -> {level: group}."""

    def __init__(self, spec, continuous, alpha, trend, merges=None):
        super().__init__(spec, continuous, alpha, trend)
        self.merges = merges or {}

    def _frame(self, d: pd.DataFrame, year=None) -> pd.DataFrame:
        f = super()._frame(d, year)
        for c, m in self.merges.items():
            f[c] = f[c].map(lambda v, m=m: m.get(v, v))
        return f


def components(line: rgl.Line, s: dict) -> list[dict]:
    """The GLM components of a line as the production build fits them."""
    out = []
    if line.spec["small"] == "glm":
        out.append({"component": "small" if s["threshold"] else "whole", "part": line.spec, "target": 0,
                    "alpha": s["alpha"], "trend": True, "threshold": s["threshold"]})
    if s["threshold"] is not None:
        out.append({"component": "large", "part": line.spec["excess"], "target": 1,
                    "alpha": line.cfg["glm_lines"]["excess_alpha"], "trend": False, "threshold": s["threshold"]})
    return out


def fit_component(line: rgl.Line, comp: dict, d: pd.DataFrame, merges=None) -> tuple[GLMPlus, np.ndarray]:
    y = line.targets(d, comp["threshold"])[comp["target"]]
    spec, cont = line.glm_spec(comp["part"], d)
    return GLMPlus(spec, cont, comp["alpha"], comp["trend"], merges).fit(d, y, line.p), y


def term_of(name: str, glm: rgl.GLM) -> str | None:
    """The model term a design column belongs to: a factor, a spline (its basis and missing flag), a continuous
    column or the trend."""
    if name == "(intercept)":
        return None
    if "=" in name:
        return name.split("=", 1)[0]
    for c in glm.spec.splines:
        if name == f"{c}_missing" or re.fullmatch(rf"{re.escape(c)}_s\d+", name):
            return c
    return name


def kind_of(term: str, glm: rgl.GLM) -> str:
    if term == "t":
        return "trend"
    if term in glm.levels:
        return "factor"
    return "spline" if term in glm.spec.splines else "continuous"


class Fit:
    """What the tests need from one fitted GLM component: the design with intercept, coefficients, fitted values,
    deviance, Pearson dispersion, the unpenalised information G = X' W X (in units of half the deviance) with its
    pseudo-inverse and rank, the cluster-robust meat B, and the effective dispersion trace(G+ B) / rank, which is
    what the sandwich implies for the model's estimable directions on average."""

    def __init__(self, glm: rgl.GLM, d: pd.DataFrame, y: np.ndarray, p: float, clusters):
        X, names = models.design_matrix(glm._frame(d), glm.levels, glm.numeric)
        self.X, self.names = np.column_stack([np.ones(len(d)), X]), ["(intercept)"] + names
        self.beta = np.r_[glm.model.intercept_, glm.model.coef_]
        self.terms = [term_of(nm, glm) for nm in self.names]
        self.y, self.p, self.e, self.alpha = y, p, d.exposure.to_numpy(), glm.alpha
        self.mu = glm.predict(d)
        self.n, self.k = self.X.shape
        self.dev = float(models.tweedie_deviance(y, self.mu, p).sum())
        self.w, self.r = self.mu ** (2 - p), (y - self.mu) * self.mu ** (1 - p)
        self.G = gram(self.X, self.w)
        self.Gp = np.linalg.pinv(self.G, rcond=1e-10, hermitian=True)
        self.rank = int(np.linalg.matrix_rank(self.G, tol=1e-10 * np.linalg.eigvalsh(self.G).max(), hermitian=True))
        self.phi = float(((y - self.mu) ** 2 / self.mu ** p).sum() / (self.n - self.rank))
        self.phi_deviance = self.dev / (self.n - self.rank)
        self.order, self.starts = cluster_order(clusters)
        self.B = meat(self.X, self.r, self.order, self.starts)
        self.phi_effective = float(np.trace(self.Gp @ self.B)) / self.rank
        self.lam = glm.alpha * float((self.e ** (2 - p)).sum())
        self.penalty_share = self.lam * float((self.beta[1:] ** 2).sum()) / self.dev


# ---- S1: Type III tests and actual against expected by level ---------------------------------------------------

def type3(line: rgl.Line, comp: dict, glm: rgl.GLM, fit: Fit) -> list[dict]:
    rows = []
    for term in dict.fromkeys(t for t in fit.terms if t is not None):
        cols = [j for j, t in enumerate(fit.terms) if t == term]
        keep = [0] + [j for j in range(1, fit.k) if fit.terms[j] != term]
        m = models.fit_tweedie(fit.X[:, keep[1:]], fit.y, fit.e, fit.p, fit.alpha)
        mu = m.predict(fit.X[:, keep[1:]]) * fit.e
        change = float(models.tweedie_deviance(fit.y, mu, fit.p).sum()) - fit.dev
        Xr, wr, rr = fit.X[:, keep], mu ** (2 - fit.p), (fit.y - mu) * mu ** (1 - fit.p)
        Gp = np.linalg.pinv(gram(Xr, wr), rcond=1e-10, hermitian=True)
        sc = score_test(Xr, Gp, wr, rr, fit.X[:, cols], fit.order, fit.starts, fit.phi_effective)
        df = sc["df"]
        f_stat = max(change, 0.0) / df / fit.phi if df else 0.0
        rows.append({"line": line.name, "component": comp["component"], "term": term, "kind": kind_of(term, glm),
                     "columns": len(cols), "df": df, "deviance_change": change, "f_stat": f_stat,
                     "p_f": float(stats.f.sf(f_stat, df, fit.n - fit.rank)) if df else 1.0,
                     "score_robust": sc["score_robust"], "score_model": sc["score_model"], "score": sc["score"],
                     "p_score": sc["p"]})
    return rows


def factor_labels(line: rgl.Line, fit_rows: pd.DataFrame, v: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each model factor's level for the rows `v`, as defined on the fitting rows: GLM factors at their model
    levels, numerics (splines, continuous terms, XGBoost features) in deciles, XGBoost categoricals with levels
    under 0.5% of exposure pooled."""
    out, cats = {}, []
    if line.spec["small"] == "glm":
        spec, cont = line.glm_spec(line.spec, fit_rows)
        f = spec.factors(v)
        out = {c: f[c].to_numpy() for c in f.columns}
        numeric = list(spec.splines) + cont
    else:
        lm = line.cfg["lines"]["models"][line.name]
        numeric, cats = lm["numeric"], lm["categorical"]
    edges = glm_lines.fit_bands(fit_rows, numeric, 10)
    for c in numeric:
        out[c] = glm_lines._band(v[c], edges[c]).to_numpy()
    for c in cats:
        lv = fit_rows[c].astype("string").fillna("MISSING")
        share = fit_rows.exposure.groupby(lv).sum() / fit_rows.exposure.sum()
        s = v[c].astype("string").fillna("MISSING")
        out[c] = s.where(s.isin(set(share.index[share >= 0.005])), "OTHER").to_numpy()
    return out


def ae_tests(line: rgl.Line, preds: dict) -> tuple[list[dict], list[dict]]:
    levels, summary = [], []
    for w in EVAL:
        yrs, yv = rgl.WINDOWS[w]
        fit_rows, v = line.rows(yrs), line.rows([yv])
        y = v.y_full.to_numpy()
        m = preds[w] * y.sum() / preds[w].sum()
        phi = float(((y - m) ** 2 / m ** line.p).sum() / len(y))
        rows = []
        for factor, lab in factor_labels(line, fit_rows, v).items():
            g = pd.DataFrame({"level": lab, "actual": y, "expected": m, "var": phi * m ** line.p,
                              "claims": (y > 0).astype(int), "exposure": v.exposure_plain.to_numpy()})
            g = g.groupby("level").sum().reset_index()
            g["ae"], g["z"] = g.actual / g.expected, (g.actual - g.expected) / np.sqrt(g["var"])
            chi2, df = float((g.z ** 2).sum()), len(g) - 1
            worst = g.loc[g.z.abs().idxmax()]
            rows.append({"line": line.name, "score_year": yv, "factor": factor, "levels": len(g), "df": df,
                         "chi2": chi2, "p": float(stats.chi2.sf(chi2, df)) if df else 1.0, "phi": phi,
                         "worst_level": worst.level, "worst_ae": worst.ae, "worst_z": worst.z,
                         "worst_claims": int(worst.claims)})
            levels += g.assign(line=line.name, score_year=yv, factor=factor).to_dict("records")
        q = bh([r["p"] for r in rows])
        for r, qi in zip(rows, q):
            r["q_bh"] = qi
        summary += rows
    return levels, summary


def s1(cfg: dict, summary_ev: pd.DataFrame) -> dict:
    lines = ft.lines_for(cfg)
    t3, fits, levels, ae = [], [], [], []
    for name, line in lines.items():
        s = settings(line, summary_ev)
        line.use_window("final")
        d = line.rows(rgl.WINDOWS["final"][0])
        for comp in components(line, s):
            glm, y = fit_component(line, comp, d)
            fit = Fit(glm, d, y, line.p, d.customer_id.to_numpy())
            fits.append({"line": name, "component": comp["component"], "rows": fit.n, "columns": fit.k,
                         "rank": fit.rank, "deviance": fit.dev, "phi_pearson": fit.phi,
                         "phi_effective": fit.phi_effective, "phi_deviance": fit.phi_deviance, "alpha": fit.alpha,
                         "penalty_share_of_deviance": fit.penalty_share, "rows_with_claims": int((y > 0).sum())})
            t3 += type3(line, comp, glm, fit)
            print(f"  {name} {comp['component']}: {fit.n} rows, {fit.k} columns (rank {fit.rank}), phi Pearson "
                  f"{fit.phi:.1f}, effective {fit.phi_effective:.1f}", flush=True)
        trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
        lv, sm = ae_tests(line, window_preds(line, s, lambda ln, yrs: ln.build(yrs, s["threshold"], s["alpha"], trees)))
        levels += lv
        ae += sm
    return {"ev_s1_type3.csv": pd.DataFrame(t3), "ev_s1_component_fits.csv": pd.DataFrame(fits),
            "ev_s1_ae_by_level.csv": pd.DataFrame(levels), "ev_s1_ae_chisq.csv": pd.DataFrame(ae)}


# ---- S2: level merging and dropping terms ---------------------------------------------------------------------

def level_key(level: str) -> float | None:
    """Sort key for a level that is a number or a quantile band, else None."""
    if re.fullmatch(r"q\d\d", level):
        return float(level[1:])
    try:
        return float(level)
    except ValueError:
        return None


def wald_merges(fit: Fit, glm: rgl.GLM, threshold: float = MERGE_WALD) -> tuple[dict, list[dict]]:
    """Greedy merging of factor levels while the Wald statistic for equal coefficients is below `threshold`.
    Each pass works in the parameterisation the merges so far define (beta = M theta): under the quadratic
    approximation theta = (M'GM)+ M'G beta, with covariance phi_effective * (M'GM)+. A merge whose contrast is
    not estimable there (a level aliased with another factor's) leaves the fit unchanged and is made freely, with
    statistic 0; an estimable one is Wald-tested. Numeric or banded levels merge only with a neighbour; MISSING
    and OTHER, and nominal levels, with any level."""
    idx = {nm: j for j, nm in enumerate(fit.names)}
    groups = {f: [[lv] for lv in lv_] for f, lv_ in glm.levels.items()}
    ordered = {f: all(level_key(x) is not None for x in lv_ if x not in SPECIAL) and any(x not in SPECIAL for x in lv_)
               for f, lv_ in glm.levels.items()}
    Gb, log = fit.G @ fit.beta, []

    def pos(g):
        keys = [level_key(x) for x in g if x not in SPECIAL]
        return min(keys) if keys else None

    while True:
        # Columns of theta: one per group without the reference level (whose columns are fixed at 0), one per
        # column outside the factors.
        cols, theta_of, used = [], {}, set()
        for f, G in groups.items():
            for g in G:
                js = [idx[f"{f}={lv}"] for lv in g if f"{f}={lv}" in idx]
                used.update(js)
                if len(js) == len(g):
                    theta_of[(f, tuple(g))] = len(cols)
                    cols.append(js)
        cols += [[j] for j in range(fit.k) if j not in used]
        M = np.zeros((fit.k, len(cols)))
        for t, js in enumerate(cols):
            M[js, t] = 1.0
        Gm = M.T @ fit.G @ M
        Gmp = np.linalg.pinv(Gm, rcond=1e-10, hermitian=True)
        theta, Pm = Gmp @ (M.T @ Gb), Gmp @ Gm
        best = None
        for f, G in groups.items():
            if ordered[f]:
                band = sorted([g for g in G if pos(g) is not None], key=pos)
                pairs = list(zip(band, band[1:])) + [(g, h) for g in G if pos(g) is None for h in G if h is not g]
            else:
                pairs = list(itertools.combinations(G, 2))
            for g, h in pairs:
                c = np.zeros(len(cols))
                for grp, sign in ((g, 1.0), (h, -1.0)):
                    t = theta_of.get((f, tuple(grp)))
                    if t is not None:
                        c[t] += sign
                if np.linalg.norm(c - Pm @ c) > 1e-6 * np.linalg.norm(c):
                    stat = 0.0  # not estimable: an aliased merge, free
                else:
                    var = fit.phi_effective * float(c @ Gmp @ c)
                    stat = float(c @ theta) ** 2 / var if var > 0 else 0.0
                if best is None or stat < best[0]:
                    best = (stat, f, g, h)
        if best is None or best[0] >= threshold:
            break
        stat, f, g, h = best
        groups[f] = [x for x in groups[f] if x is not g and x is not h] + [sorted(g + h)]
        log.append({"factor": f, "levels": "+".join(g), "merged_with": "+".join(h), "wald": stat})
    merges = {f: {lv: "+".join(g) for g in G for lv in g} for f, G in groups.items()
              if len(G) < len(glm.levels[f])}
    return merges, log


def assemble(line: rgl.Line, d: pd.DataFrame, s: dict, parts: dict, base: rgl.LineModel | None) -> rgl.LineModel:
    """A LineModel from refitted GLM components, taking any component not refitted from `base`."""
    small = parts.get("small", parts.get("whole"))
    small = base.small if small is None else small
    if s["threshold"] is None:
        return rgl.LineModel(small)
    tail = glm_lines.gpd_tail(line.claims[line.claims.row_id.isin(d.row_id)], s["threshold"])
    return rgl.LineModel(small, parts.get("large", base.excess if base else None), tail)


def storing_build(s: dict, trees, store: dict):
    """The production build, keeping each window's model so variants can share the parts they do not change."""
    def build(line, yrs):
        store[tuple(yrs)] = line.build(yrs, s["threshold"], s["alpha"], trees)
        return store[tuple(yrs)]
    return build


def merged_build(s: dict, store: dict, log: list):
    def build(line, yrs):
        d, parts = line.rows(yrs), {}
        for comp in components(line, s):
            glm, y = fit_component(line, comp, d)
            merges, steps = wald_merges(Fit(glm, d, y, line.p, d.customer_id.to_numpy()), glm)
            log.extend({"line": line.name, "fit_years": f"{yrs[0]}-{yrs[-1]}", "component": comp["component"], **r}
                       for r in steps)
            parts[comp["component"]] = fit_component(line, comp, d, merges)[0] if merges else glm
        return assemble(line, d, s, parts, store.get(tuple(yrs)))
    return build


def without_term(line: rgl.Line, comp: dict, term: str) -> dict:
    """The component's spec with one term removed: a factor or spline from its numeric or categorical lists, a
    continuous column from its continuous list, or the year trend."""
    if term == "t":
        return {**comp, "trend": False}
    part = copy.deepcopy(comp["part"])
    for key in ("numeric", "categorical", "continuous"):
        if key in part:
            part[key] = [c for c in rgl.resolve(line.cfg, part[key]) if c != term]
    if isinstance(part.get("smooth"), list):
        part["smooth"] = [c for c in part["smooth"] if c != term]
    return {**comp, "part": part}


def dropped_build(s: dict, store: dict, comp: dict, term: str):
    def build(line, yrs):
        d = line.rows(yrs)
        return assemble(line, d, s, {comp["component"]: fit_component(line, without_term(line, comp, term), d)[0]},
                        store.get(tuple(yrs)))
    return build


def s2(cfg: dict, summary_ev: pd.DataFrame) -> dict:
    rows, log, dropped = [], [], []
    for name, line in ft.lines_for(cfg).items():
        s, store = settings(line, summary_ev), {}
        trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
        base = window_preds(line, s, storing_build(s, trees, store))
        merged = window_preds(line, s, merged_build(s, store, log))
        rows += compare(line, base, merged, "S2 level merging", f"Wald < {MERGE_WALD:g}", cfg)
        line.use_window("final")
        for comp in components(line, s):
            glm, _ = fit_component(line, comp, line.rows(rgl.WINDOWS["final"][0]))
            tested = []
            for term in dict.fromkeys(term_of(nm, glm) for nm in glm.names):
                alt = window_preds(line, s, dropped_build(s, store, comp, term))
                tested.append(compare(line, base, alt, f"S2 drop term ({comp['component']})", term, cfg))
            holm(tested)
            dropped += [r for rs in tested for r in rs]
    return {"ev_s2_level_merges.csv": pd.DataFrame(log), "ev_s2_merge_two_year.csv": pd.DataFrame(rows),
            "ev_s2_drop_term_two_year.csv": pd.DataFrame(dropped)}


# ---- S3: pairwise interactions ---------------------------------------------------------------------------------

def numeric_rep(name: str, x: np.ndarray) -> dict:
    """A numeric term for interactions: clipped to its 1st-99th percentiles, standardised, missing at the mean."""
    lo, hi = np.nanquantile(x, [0.01, 0.99])
    xc = np.clip(x, lo, hi)
    mean, sd = float(np.nanmean(xc)), float(np.nanstd(xc)) or 1.0
    return {"kind": "num", "name": name,
            "fn": lambda df: np.nan_to_num((np.clip(df[name].to_numpy(dtype=float), lo, hi) - mean) / sd)}


def grouped_rep(name: str, labels_fn, labels: np.ndarray, expo: np.ndarray, order: dict | None = None) -> dict:
    """A categorical term for interactions: at most five levels kept; otherwise four groups, by exposure-weighted
    quartile of `order` (the fitted coefficient) or, without one, the four largest levels plus the rest pooled.
    The reference group is the one with the most exposure."""
    ex = pd.Series(expo).groupby(labels).sum().sort_values(ascending=False)
    if len(ex) <= 5:
        gmap = {lv: lv for lv in ex.index}
    elif order is not None:
        lv = sorted(ex.index, key=lambda x: order.get(x, 0.0))
        mid = (ex[lv].cumsum() - ex[lv] / 2) / ex.sum()
        gmap = {x: f"g{min(3, int(4 * m)) + 1}" for x, m in zip(lv, mid)}
    else:
        gmap = {x: (x if i < 4 else "rest") for i, x in enumerate(ex.index)}
    gex = ex.groupby(pd.Series(gmap)).sum()
    ref = gex.idxmax()
    fallback = "rest" if "rest" in gex.index else ref  # a level not seen when the groups were defined
    return {"kind": "cat", "name": name, "groups": sorted(gex.index), "ref": ref, "map": gmap,
            "fn": lambda df: pd.Series(labels_fn(df)).map(gmap).fillna(fallback).to_numpy()}


def z_columns(ra: dict, rb: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Interaction columns of two terms: products of the numeric values and of non-reference group indicators."""
    pa, pb = dict(zip(names_of(ra), main_columns(ra, df))), dict(zip(names_of(rb), main_columns(rb, df)))
    return {f"{a} x {b}": va * vb for a, va in pa.items() for b, vb in pb.items()}


def names_of(r: dict) -> list[str]:
    return [r["name"]] if r["kind"] == "num" else [f"{r['name']}:{g}" for g in r["groups"] if g != r["ref"]]


def main_columns(r: dict, df: pd.DataFrame) -> list[np.ndarray]:
    if r["kind"] == "num":
        return [r["fn"](df)]
    lab = r["fn"](df)
    return [(lab == g).astype(float) for g in r["groups"] if g != r["ref"]]


def modelled(a: str, b: str) -> bool:
    """Whether the pair's interaction is already a model feature, or one term is built from the other."""
    return any({a, b} == {k, t} or {a, b} <= v for k, v in MODELLED.items() for t in v)


def glm_reps(glm: rgl.GLM, fit: Fit, d: pd.DataFrame) -> dict:
    f, out = glm._frame(d), {}
    for term in dict.fromkeys(t for t in fit.terms if t is not None):
        if term == "t":
            centre = glm.centre
            out[term] = {"kind": "num", "name": "t", "fn": lambda df, c=centre: df.yr.to_numpy(dtype=float) - c}
        elif term in glm.levels:
            coef = {lv: (fit.beta[fit.names.index(f"{term}={lv}")] if f"{term}={lv}" in fit.names else 0.0)
                    for lv in glm.levels[term]}
            out[term] = grouped_rep(term, lambda df, t=term, sp=glm.spec: sp.factors(df)[t].to_numpy(),
                                    f[term].to_numpy(), d.exposure.to_numpy(), coef)
        else:
            out[term] = numeric_rep(term, d[term].to_numpy(dtype=float))
    return out


def screen_glm(line: rgl.Line, comp: dict, d: pd.DataFrame) -> tuple[list[dict], dict]:
    glm, y = fit_component(line, comp, d)
    fit = Fit(glm, d, y, line.p, d.customer_id.to_numpy())
    reps, rows = glm_reps(glm, fit, d), []
    for a, b in itertools.combinations(reps, 2):
        row = {"line": line.name, "model": comp["component"], "term_a": a, "term_b": b}
        if modelled(a, b):
            rows.append({**row, "already_modelled": True})
            continue
        z = [v for v in z_columns(reps[a], reps[b], d).values() if v.any()]
        res = (score_test(fit.X, fit.Gp, fit.w, fit.r, np.column_stack(z), fit.order, fit.starts, fit.phi_effective)
               if z else EMPTY)
        rows.append({**row, "columns": len(z), **res, "already_modelled": False})
    return rows, reps


def fit_offset(N: np.ndarray, y: np.ndarray, off: np.ndarray, p: float, iters: int = 50) -> np.ndarray:
    """Tweedie log-link fit of y on N with offset `off` by Fisher scoring."""
    b = np.zeros(N.shape[1])
    b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(iters):
        mu = np.exp(off + N @ b)
        step = np.linalg.lstsq(gram(N, mu ** (2 - p)), N.T @ ((y - mu) * mu ** (1 - p)), rcond=None)[0]
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def screen_xgb(line: rgl.Line, s: dict, trees, d: pd.DataFrame, only=None, model="small (XGBoost)"
               ) -> tuple[list[dict], dict]:
    """Interaction screen of the small-claim XGBoost on its own fitted rows: the XGBoost prediction is an offset,
    each pair's main effects are refitted as nuisance terms, and the pair's interaction columns are score-tested.
    The model-based statistic uses the Pearson dispersion of the XGBoost on these rows. `only` restricts the
    screen to the given pairs, screened even if already modelled (for the power check)."""
    y = line.targets(d, s["threshold"])[0]
    mu0 = line.build(SCREEN_YEARS, s["threshold"], s["alpha"], trees).small.predict(d)
    off = np.log(mu0)
    phi = float(((y - mu0) ** 2 / mu0 ** line.p).sum() / len(y))
    lm = line.cfg["lines"]["models"][line.name]
    expo, reps = d.exposure.to_numpy(), {}
    for c in lm["numeric"]:
        reps[c] = numeric_rep(c, d[c].to_numpy(dtype=float))
    for c in lm["categorical"]:
        def raw(df, c=c):
            return df[c].astype("string").fillna("MISSING").to_numpy()
        reps[c] = grouped_rep(c, raw, raw(d), expo)
    order, starts = cluster_order(d.customer_id.to_numpy())
    rows = []
    for a, b in (only or itertools.combinations(reps, 2)):
        row = {"line": line.name, "model": model, "term_a": a, "term_b": b}
        if only is None and modelled(a, b):
            rows.append({**row, "already_modelled": True})
            continue
        z = [v for v in z_columns(reps[a], reps[b], d).values() if v.any()]
        if not z:
            rows.append({**row, "columns": 0, **EMPTY, "already_modelled": False})
            continue
        N = np.column_stack([np.ones(len(d))] + main_columns(reps[a], d) + main_columns(reps[b], d))
        mu = np.exp(off + N @ fit_offset(N, y, off, line.p))
        w, r = mu ** (2 - line.p), (y - mu) * mu ** (1 - line.p)
        res = score_test(N, np.linalg.pinv(gram(N, w), hermitian=True), w, r, np.column_stack(z), order, starts, phi)
        rows.append({**row, "columns": len(z), **res, "already_modelled": False})
    return rows, reps


def two_year_p(mean: float, lo: float, hi: float) -> float:
    """One-sided p-value of the two-year mean gain, from its approximate 95% interval."""
    return float(stats.norm.sf(mean / ((hi - lo) / 3.92)))


def with_features(line: rgl.Line, cfg: dict, cols: dict) -> rgl.Line:
    new = rgl.Line(line.name, line.df.assign(exposure=line.df.exposure_plain, **cols), line.claims, cfg)
    new.expected, new.month_days, new.claim_months = line.expected, line.month_days, line.claim_months
    return new


def candidate_variant(line: rgl.Line, model: str, reps: dict, a: str, b: str, tag: str) -> tuple[rgl.Line, list]:
    """The line with the interaction of a and b added as fixed feature columns to `model`."""
    z = z_columns(reps[a], reps[b], line.df)
    cols = {f"{tag}_{i}": v for i, v in enumerate(z.values())}
    vcfg = copy.deepcopy(line.cfg)
    if model == "small (XGBoost)":
        vcfg["lines"]["models"][line.name]["numeric"] = vcfg["lines"]["models"][line.name]["numeric"] + list(cols)
    else:
        m = vcfg["glm_lines"]["models"][line.name]
        part = m["excess"] if model == "large" else m
        part["continuous"] = rgl.resolve(vcfg, part.get("continuous", [])) + list(cols)
    return with_features(line, vcfg, cols), list(cols)


def variant_build(s: dict, store: dict, model: str, trend_by: dict | None = None, n_cols: int = 0):
    """For a GLM-component candidate: refit only that component and share the rest with the production build.
    When a small-claim candidate interacts the trend with another term (`trend_by`), the large claims, which
    take the small-claim model's trend, get its exposure-weighted average over the fitting rows rather than the
    reference group's coefficient."""
    def build(line, yrs):
        d = line.rows(yrs)
        comp = next(c for c in components(line, s) if c["component"] == model)
        glm = fit_component(line, comp, d)[0]
        if trend_by is not None and glm.trend:
            g = np.column_stack(main_columns(trend_by, d))
            coef = glm.model.coef_[-1 - n_cols:-1]  # the candidate's columns sit just before the trend
            glm.slope = float(np.average(glm.slope + g @ coef, weights=d.exposure))
        return assemble(line, d, s, {model: glm}, store.get(tuple(yrs)))
    return build


def holm(tested: list[list[dict]]) -> None:
    """Adds the one-sided p-value of each candidate's two-year mean and Holm's step-down decision at 2.5%, across
    the candidates scored for one model; adoption also needs both years to improve."""
    ps = [two_year_p(rs[0]["two_year_mean"], rs[0]["two_year_ci_low_95"], rs[0]["two_year_ci_high_95"]) for rs in tested]
    passed, m = True, len(ps)
    for rank, i in enumerate(np.argsort(ps)):
        passed = passed and ps[i] < 0.025 / (m - rank)
        for r in tested[i]:
            r["p_two_year_one_sided"], r["passes_holm"] = ps[i], passed
            r["adopt"] = bool(passed and r["passes_significance_rule"])


def s3(cfg: dict, summary_ev: pd.DataFrame) -> dict:
    screen, scored, power = [], [], []
    for name, line in ft.lines_for(cfg).items():
        s = settings(line, summary_ev)
        line.use_window("tune")
        d = line.rows(SCREEN_YEARS)
        trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
        families = [(comp["component"], *screen_glm(line, comp, d)) for comp in components(line, s)]
        if line.spec["small"] == "xgb":
            families.append(("small (XGBoost)", *screen_xgb(line, s, trees, d)))
            power += power_check(line, s, trees, d)
        store, xgb_base = {}, None
        glm_base = window_preds(line, s, storing_build(s, trees, store))
        for model, rows, reps in families:
            live = [r for r in rows if not r["already_modelled"]]
            for r, qi in zip(live, bh([r["p"] for r in live])):
                r["q_bh"], r["fdr_significant"] = qi, bool(qi < FDR)
            top = sorted([r for r in live if r["df"] > 0], key=lambda r: (r["p"], -r["score"]))[:TOP]
            for r in rows:
                r["scored_out_of_time"] = any(r is t for t in top)
            print(f"  {name} {model}: {len(live)} pairs screened, {sum(r['fdr_significant'] for r in live)} pass BH; "
                  f"scoring {len(top)}", flush=True)
            screen += rows
            tested = []
            for r in top:
                v, cols = candidate_variant(line, model, reps, r["term_a"], r["term_b"], f"zz{len(scored) + len(tested)}")
                if model == "small (XGBoost)":
                    xgb_base = ft.bagged(line, s, trees) if xgb_base is None else xgb_base
                    alt, ref = ft.bagged(v, s, trees), xgb_base
                else:
                    trend_by = None
                    if model == "small" and "t" in (r["term_a"], r["term_b"]):
                        trend_by = reps[r["term_b"] if r["term_a"] == "t" else r["term_a"]]
                    alt = window_preds(v, s, variant_build(s, store, model, trend_by, len(cols)))
                    ref = glm_base
                tested.append(compare(line, ref, alt, f"S3 interaction ({model})", f"{r['term_a']} x {r['term_b']}",
                                      cfg))
            holm(tested)
            scored += [x for rs in tested for x in rs]
    return {"ev_s3_interaction_screen.csv": pd.DataFrame(screen), "ev_s3_interaction_two_year.csv": pd.DataFrame(scored),
            "ev_s3_power_check.csv": pd.DataFrame(power)}


def power_check(line: rgl.Line, s: dict, trees, d: pd.DataFrame) -> list[dict]:
    """Positive control for the XGBoost screen: refit it on 2021-22 without the A28 peril interaction features and
    screen the two pairs they were built from, which A28 found to be real out of time (D-29). A screen with power
    should flag them far beyond the Benjamini-Hochberg threshold of the full screen."""
    vcfg = copy.deepcopy(line.cfg)
    lm = vcfg["lines"]["models"][line.name]
    lm["numeric"] = [c for c in lm["numeric"] if c not in MODELLED]
    pairs = [(sorted(v)[1], sorted(v)[0]) for v in MODELLED.values()]
    rows, _ = screen_xgb(with_features(line, vcfg, {}), s, trees, d, only=pairs,
                         model="small (XGBoost) without the A28 interactions")
    for r in rows:
        r["power_check"] = True
        print(f"  power check {r['term_a']} x {r['term_b']}: p = {r['p']:.2e}", flush=True)
    return rows


# ---- Runner ---------------------------------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    which = args[args.index("--tests") + 1].split(",") if "--tests" in args else ["S1", "S2", "S3"]
    unknown = set(which) - {"S1", "S2", "S3"}
    if unknown:
        raise SystemExit(f"unknown tests: {sorted(unknown)}")
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    rec["tests"] = which
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        summary_ev = pd.read_csv(OUT / "ev_g6_summary.csv")
        frames = {}
        for test, fn in (("S1", s1), ("S2", s2), ("S3", s3)):
            if test in which:
                frames.update(fn(cfg, summary_ev))
        outputs = [write_csv(df, OUT / name) for name, df in frames.items()]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {name: df.round(6).to_dict("records") for name, df in frames.items()
                          if name not in ("ev_s1_ae_by_level.csv", "ev_s3_interaction_screen.csv")}
        rec["status"] = "success"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost", "scipy")}}
        rec.update(ft.PROVENANCE)
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
