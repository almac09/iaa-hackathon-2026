"""Unit tests for the statistical tests of python/stat_tests.py (A29), on simulated Tweedie data."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import glm_lines  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import stat_tests as st  # noqa: E402

P = 1.5


class _FakeLine:
    name = "test"


def tweedie(mu, phi, rng):
    """Compound Poisson-gamma draws with mean mu and variance phi mu^1.5."""
    n = rng.poisson(mu ** (2 - P) / (phi * (2 - P)))
    shape, scale = (2 - P) / (P - 1), phi * (P - 1) * mu ** (P - 1)
    return np.where(n > 0, rng.gamma(np.maximum(shape * n, 1e-9), scale), 0.0)


def simulate(n=20_000, seed=1, a_effect=(0.0, 0.3, 0.3), interaction=0.0, levels=("A", "B", "C"), phi=30.0,
             probs=None):
    rng = np.random.default_rng(seed)
    a = rng.choice(list(levels), n, p=probs)
    d = pd.DataFrame({"yr": rng.choice([2021, 2022], n), "exposure": rng.uniform(0.2, 1.0, n), "a": a,
                      "b": rng.choice(["x", "y"], n), "x": rng.normal(size=n),
                      "customer_id": rng.integers(0, n // 2, n)})
    eff = dict(zip(levels, a_effect))
    log_rate = np.log(200) + d.a.map(eff) + 0.2 * d.x + interaction * d.x * (d.a == levels[-1])
    d["y"] = tweedie(np.exp(log_rate) * d.exposure, phi, rng)
    return d


def fit(d, alpha=0.0, cats=("a", "b"), cont=("x",), trend=True, merges=None):
    spec = glm_lines.FactorSpec(d, [], list(cats))
    glm = st.GLMPlus(spec, list(cont), alpha, trend, merges).fit(d, d.y.to_numpy(), P)
    return glm, st.Fit(glm, d, d.y.to_numpy(), P, d.customer_id.to_numpy())


def score(f, Z, phi=None):
    Z = np.asarray(Z, dtype=float)
    return st.score_test(f.X, f.Gp, f.w, f.r, Z if Z.ndim == 2 else Z[:, None], f.order, f.starts,
                         f.phi if phi is None else phi)


def test_glmplus_without_merges_is_the_production_glm():
    d = simulate()
    spec = glm_lines.FactorSpec(d, [], ["a", "b"])
    base = rgl.GLM(spec, ["x"], 1e-4, True).fit(d, d.y.to_numpy(), P)
    plus = st.GLMPlus(spec, ["x"], 1e-4, True).fit(d, d.y.to_numpy(), P)
    assert np.array_equal(base.predict(d, 2023), plus.predict(d, 2023))


def test_cluster_meat_matches_a_direct_sum():
    rng = np.random.default_rng(3)
    Z, r, cl = rng.normal(size=(500, 3)), rng.normal(size=500), rng.integers(0, 60, 500)
    order, starts = st.cluster_order(cl)
    direct = sum(np.outer(s, s) for s in (Z[cl == c].T @ r[cl == c] for c in np.unique(cl)))
    assert np.allclose(st.meat(Z, r, order, starts, step=7), direct)


def test_fit_penalty_matches_sklearns_optimum():
    d = simulate()
    glm, f = fit(d, alpha=1e-4)
    assert f.dev == pytest.approx(models.tweedie_deviance(d.y.to_numpy(), glm.predict(d), P).sum())
    # sklearn's optimum: the score of half the deviance equals the ridge gradient lam * beta (none on the intercept).
    pen = f.lam * np.r_[0.0, f.beta[1:]]
    assert np.abs(f.X.T @ f.r - pen).max() < 1e-6 * np.abs(pen).max()
    for wrong in (0.0, 2.0):  # neither no penalty nor twice the penalty satisfies it
        assert np.abs(f.X.T @ f.r - wrong * pen).max() > 0.5 * np.abs(pen).max()


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R is not installed")
def test_type3_f_tests_match_r_glm_with_statmod(tmp_path):
    d = simulate()
    glm, f = fit(d)
    ours = pd.DataFrame(st.type3(_FakeLine(), {"component": "whole"}, glm, f)).set_index("term")
    d.assign(t=d.yr - glm.centre).to_csv(tmp_path / "d.csv", index=False)
    script = f"""
        if (!requireNamespace("statmod", quietly = TRUE)) quit(status = 3)
        d <- read.csv("{tmp_path / 'd.csv'}"); d$a <- factor(d$a); d$b <- factor(d$b)
        fam <- statmod::tweedie(var.power = 1.5, link.power = 0)
        ctl <- glm.control(epsilon = 1e-12, maxit = 200)
        full <- glm(y ~ a + b + x + t, family = fam, offset = log(exposure), data = d, control = ctl)
        phi <- sum(residuals(full, type = "pearson")^2) / full$df.residual
        out <- sapply(c("a", "b", "x", "t"), function(term) {{
          red <- update(full, as.formula(paste(". ~ . -", term)))
          ((deviance(red) - deviance(full)) / (red$df.residual - full$df.residual)) / phi }})
        write.csv(data.frame(term = names(out), f = out), "{tmp_path / 'r.csv'}", row.names = FALSE)
    """
    res = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if res.returncode == 3:
        pytest.skip("R package statmod is not installed")
    assert res.returncode == 0, res.stderr
    r = pd.read_csv(tmp_path / "r.csv").set_index("term")
    for term in ["a", "b", "x", "t"]:
        assert ours.loc[term, "f_stat"] == pytest.approx(r.loc[term, "f"], rel=1e-4)
    assert (ours["df"] == ours["columns"]).all()
    assert ours.loc["a", "p_f"] < 1e-6 and ours.loc["b", "p_f"] > 0.01 and ours.loc["a", "p_score"] < 1e-6


def test_type3_counts_only_estimable_columns():
    # s is the a x b cell, so a and b are aliased with it: only s's interaction part is estimable beyond them.
    d = simulate()
    d["s"] = d.a + "_" + d.b
    glm, f = fit(d, alpha=1e-4, cats=("a", "b", "s"))
    t3 = pd.DataFrame(st.type3(_FakeLine(), {"component": "whole"}, glm, f)).set_index("term")
    assert t3.loc["a", "df"] == 0 and t3.loc["b", "df"] == 0 and t3.loc["s", "df"] == 2
    assert t3.loc["a", "p_f"] == 1.0 and t3.loc["s", "columns"] == 5
    assert f.rank == f.k - 3


def test_score_test_agrees_with_the_likelihood_ratio_and_is_quiet_under_the_null():
    d = simulate(interaction=0.15)
    glm, f = fit(d)
    z = np.column_stack([d.x * (d.a == "B"), d.x * (d.a == "C")])
    res = score(f, z)
    glm2, f2 = fit(d.assign(zb=z[:, 0], zc=z[:, 1]), cont=("x", "zb", "zc"))
    assert res["df"] == 2
    assert res["score_model"] == pytest.approx((f.dev - f2.dev) / f.phi, rel=0.15)
    assert 0.6 < res["score_robust"] / res["score_model"] < 1.6  # the data follow the model, so the two agree
    assert res["p"] < 1e-4
    null = simulate(seed=7)
    _, fn = fit(null)
    assert score(fn, np.column_stack([null.x * (null.a == "B"), null.x * (null.a == "C")]))["p"] > 0.001


@pytest.mark.parametrize("alpha", [0.0, 1e-3])
def test_score_test_ignores_columns_already_in_the_model(alpha):
    d = simulate()
    glm, f = fit(d, alpha=alpha)
    assert score(f, (d.b == "y"))["df"] == 0
    # A column in the span plus a genuinely new one: one degree of freedom, whatever the penalty.
    both = np.column_stack([(d.b == "y"), d.x * (d.a == "C")])
    assert score(f, both)["df"] == 1
    assert score(f, np.column_stack([d.x * (d.a == "C"), 2 * d.x * (d.a == "C")]))["df"] == 1  # aliased pair


def test_score_test_is_not_fooled_by_a_level_without_claims():
    # Level D has no effect, about 1% of rows and about 1.5 expected claims; its claims are removed. The sandwich
    # alone sees no variation there and calls it significant; the model-based bound keeps the test honest.
    d = simulate(phi=3000.0, a_effect=(0.0, 0.0, 0.0, 0.0), levels=("A", "B", "C", "D"),
                 probs=[0.33, 0.33, 0.33, 0.01])
    d.loc[d.a == "D", "y"] = 0.0
    _, f = fit(d, alpha=1e-4, cats=("b",))
    res = score(f, (d.a == "D"), phi=f.phi_effective)
    assert res["score_robust"] > 10 * res["score_model"]
    assert res["p"] > 0.01


def test_wald_merges_join_equal_levels_and_respect_order():
    d = simulate(n=40_000, a_effect=(0.0, 0.0, 0.5, 0.5), levels=("A", "B", "C", "D"))
    glm, f = fit(d)
    merges, log = st.wald_merges(f, glm)
    a = merges["a"]
    assert a["A"] == a["B"] == "A+B" and a["C"] == a["D"] == "C+D"
    assert log and all(r["wald"] < st.MERGE_WALD for r in log)
    # Numeric levels merge only with neighbours: 1 and 3 share an effect but 2 sits between them.
    glm, f = fit(simulate(n=40_000, a_effect=(0.0, 0.6, 0.0), levels=("1", "2", "3")))
    assert "a" not in st.wald_merges(f, glm)[0]
    # The same effects on nominal labels do merge the outer pair, so the line above tests adjacency.
    glm, f = fit(simulate(n=40_000, a_effect=(0.0, 0.6, 0.0), levels=("p", "q", "r")))
    m = st.wald_merges(f, glm)[0]["a"]
    assert m["p"] == m["r"] != m["q"]


def test_wald_merges_merge_aliased_levels_freely_and_still_test_the_rest():
    # As motor's marketing segment against credit band: the segment is a coarsening of credit, so wholly aliased.
    d = simulate(n=40_000, a_effect=(0.0, 0.5, 0.0, 0.0, 0.0, 0.0), levels=("A", "B", "C", "D", "E", "F"))
    d["s"] = np.select([d.a == "E", d.a == "F"], ["budget", "mainstream"], "premium")
    glm, f = fit(d, alpha=1e-4, cats=("a", "s"))
    assert f.rank == f.k - 2
    merges, log = st.wald_merges(f, glm)
    # The two aliased directions are spent first, at no cost; which factor gives up the levels is arbitrary.
    assert [r["wald"] for r in log[:2]] == [0.0, 0.0] and all(r["wald"] > 0 for r in log[2:])
    free, _ = st.wald_merges(f, glm, threshold=1e-12)
    refit, _ = fit(d, alpha=1e-4, cats=("a", "s"), merges=free)
    assert refit.predict(d) == pytest.approx(glm.predict(d), rel=1e-3)  # free merges leave the fit unchanged
    # After all merges, B's real effect is still there and the null levels share one rate.
    final, _ = fit(d, alpha=1e-4, cats=("a", "s"), merges=merges)
    rows = d.iloc[[0, 0, 0]].assign(a=["A", "B", "C"], s="premium")
    pred = final.predict(rows)
    assert pred[1] / pred[0] == pytest.approx(np.exp(0.5), rel=0.15) and pred[2] == pytest.approx(pred[0])


def test_wald_merges_merge_a_level_without_claims():
    d = simulate(phi=3000.0, a_effect=(0.0, 0.0, 0.0, 0.0), levels=("A", "B", "C", "D"),
                 probs=[0.33, 0.33, 0.33, 0.01])
    d.loc[d.a == "D", "y"] = 0.0
    glm, f = fit(d, alpha=1e-4)
    merges, _ = st.wald_merges(f, glm)
    assert "+" in merges["a"]["D"]


def test_merged_refit_uses_the_groups():
    d = simulate(n=40_000, a_effect=(0.0, 0.0, 0.5), levels=("A", "B", "C"))
    glm, _ = fit(d, merges={"a": {"A": "A+B", "B": "A+B"}})
    assert glm.levels["a"] == ["A+B", "C"]
    rows = d.iloc[[0, 0]].assign(a=["A", "B"])  # identical apart from the merged level
    pred = glm.predict(rows)
    assert pred[0] == pytest.approx(pred[1])


def test_without_term_removes_one_term():
    line = type("L", (), {"cfg": {"model": {"numeric": ["z_log_si"]}}})()
    comp = {"component": "whole", "trend": True,
            "part": {"numeric": ["age"], "categorical": ["f_a", "f_b"], "continuous": "model.numeric", "smooth": "all"}}
    assert st.without_term(line, comp, "f_a")["part"]["categorical"] == ["f_b"]
    assert st.without_term(line, comp, "z_log_si")["part"]["continuous"] == []
    assert st.without_term(line, comp, "age")["part"]["numeric"] == []
    assert st.without_term(line, comp, "t")["trend"] is False
    assert comp["part"]["categorical"] == ["f_a", "f_b"]  # the original is untouched


def test_level_keys():
    assert st.level_key("q03") == 3.0 and st.level_key("2") == 2.0 and st.level_key("R01") is None
    assert st.level_key("a1 <=25") is None


def test_benjamini_hochberg():
    assert np.allclose(st.bh([0.01, 0.04, 0.03, 0.5]), [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_interaction_columns():
    df = pd.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "c", "a"], "h": ["x", "y", "x", "x"]})
    num = {"kind": "num", "name": "u", "fn": lambda x: x.u.to_numpy()}
    cat = st.grouped_rep("g", lambda x: x.g.to_numpy(), df.g.to_numpy(), np.ones(4))
    cat2 = st.grouped_rep("h", lambda x: x.h.to_numpy(), df.h.to_numpy(), np.ones(4))
    assert cat["ref"] == "a" and cat2["ref"] == "x"
    assert len(st.z_columns(num, cat, df)) == 2
    assert len(st.z_columns(cat, cat2, df)) == 2
    assert list(st.z_columns(num, num, df).values())[0].tolist() == [1.0, 4.0, 9.0, 16.0]


def test_grouped_rep_pools_small_levels_and_unseen_ones_into_the_rest():
    lab = np.array(list("aaaaabbbbccdef"))
    rep = st.grouped_rep("g", lambda x: x.g.to_numpy(), lab, np.ones(len(lab)))
    assert len(rep["groups"]) == 5 and {"a", "b", "c", "rest"} <= set(rep["groups"])
    assert rep["fn"](pd.DataFrame({"g": ["zzz"]}))[0] == "rest"


def test_modelled_pairs_are_recognised():
    assert st.modelled("soil_type", "x_clay_dry") and st.modelled("dry_summer_index", "soil_type")
    assert st.modelled("x_fz3_rain", "rainfall_mm") and not st.modelled("soil_type", "flood_zone")
    assert not st.modelled("x_clay_dry", "x_fz3_rain")


def test_fit_offset_recovers_coefficients():
    rng = np.random.default_rng(5)
    n = 50_000
    off = np.log(rng.uniform(50, 150, n))
    x = rng.normal(size=n)
    y = tweedie(np.exp(off + 0.1 + 0.3 * x), 20.0, rng)
    b = st.fit_offset(np.column_stack([np.ones(n), x]), y, off, P)
    assert b == pytest.approx([0.1, 0.3], abs=0.05)


def test_holm_needs_both_years_and_steps_down():
    def cand(mean, half):
        return [{"two_year_mean": mean, "two_year_ci_low_95": mean - half, "two_year_ci_high_95": mean + half,
                 "passes_significance_rule": mean - half > 0} for _ in range(2)]
    tested = [cand(0.002, 0.001), cand(0.0006, 0.0005), cand(-0.001, 0.001)]
    st.holm(tested)
    assert tested[0][0]["adopt"] and not tested[2][0]["adopt"]
    # The second has a one-sided p of about 0.009, under Holm's 0.025 / 2 at its step, so it is kept too.
    assert tested[1][0]["p_two_year_one_sided"] == pytest.approx(0.0094, abs=0.0005)
    assert tested[1][0]["passes_holm"] and tested[1][0]["adopt"]
    vetoed = [cand(0.002, 0.001), cand(0.0006, 0.0005), cand(-0.001, 0.001)]
    for r in vetoed[0]:
        r["passes_significance_rule"] = False  # a significant two-year mean, but one backtest year is worse
    st.holm(vetoed)
    assert vetoed[0][0]["passes_holm"] and not vetoed[0][0]["adopt"]
