"""Reconcile the Python and R runs against each other and against the exploratory results.

Checks, in order:
  1. SQL evidence (ev_0*): Python and R files must be byte-identical (same SQL, same DuckDB version).
  2. Model evidence (ev_m*): numbers must agree within the tolerances in config.json.
  3. Exploratory reference: the Python results must reproduce the session's numbers within tolerance.
Writes audit/evidence/reconciliation_report.md, appends to the run log, and exits 1 on any failure.

Usage (from the pack root):  python reconcile/reconcile.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402


def main() -> int:
    cfg = common.load_config()
    tol = cfg["reconcile"]
    ev = common.PACK_ROOT / cfg["paths"]["evidence_dir"]
    py_dir, r_dir = ev / "python", ev / "r"
    rows = []  # (check, item, result, detail)

    def add(check, item, ok, detail=""):
        rows.append((check, item, "PASS" if ok else "FAIL", detail))

    # 1. SQL evidence: identical bytes.
    py_sql = sorted(p.name for p in py_dir.glob("ev*_0*.csv"))
    r_sql = sorted(p.name for p in r_dir.glob("ev*_0*.csv"))
    add("sql evidence", "same set of files", py_sql == r_sql, f"{len(py_sql)} python, {len(r_sql)} r")
    for name in sorted(set(py_sql) & set(r_sql)):
        same = (py_dir / name).read_bytes() == (r_dir / name).read_bytes()
        add("sql evidence", name, same, "byte-identical" if same else "files differ")

    # 2. Model evidence within tolerance.
    rd = lambda d, n: pd.read_csv(d / n, keep_default_na=False, na_values=[""])
    sp, sr = rd(py_dir, "ev_m2_split_test.csv"), rd(r_dir, "ev_m2_split_test.csv")
    s = sp.merge(sr, on="model", suffixes=("_py", "_r"), validate="1:1")
    for _, x in s.iterrows():
        d = abs(x.d2_py - x.d2_r)
        add("split test D2", x.model, d <= tol["tol_d2_python_vs_r"], f"py {x.d2_py:.5f} r {x.d2_r:.5f} diff {d:.5f}")
        d = abs(x.pred_over_actual_py - x.pred_over_actual_r)
        add("split test pred/actual", x.model, d <= tol["tol_ratio_python_vs_r"], f"diff {d:.5f}")

    cp, cr = rd(py_dir, "ev_m2_coefficients.csv"), rd(r_dir, "ev_m2_coefficients.csv")
    c = cp.merge(cr, on=["component", "param"], suffixes=("_py", "_r"), how="outer", indicator=True)
    add("coefficients", "same parameters in every component", (c._merge == "both").all(),
        f"{(c._merge != 'both').sum()} unmatched")
    c = c[c._merge == "both"]
    # A level with (almost) no claims drives its coefficient towards -inf, held back only by the
    # ridge penalty; the objective is flat there, so solvers stop at slightly different points with
    # no effect on predictions (audit A11). Those parameters are listed as WARN, and the
    # prediction-level check below is the binding test for the component.
    # Intercepts are excluded: a large intercept is just log mean severity, not separation.
    sep = (c.param != "(intercept)") & (np.maximum(c.coef_py.abs(), c.coef_r.abs()) > tol["quasi_separation_abs_coef"])
    for _, x in c[sep].iterrows():
        rows.append(("coefficients", f"{x.component} | {x.param}", "WARN",
                     f"quasi-separated: py {x.coef_py:.4f} r {x.coef_r:.4f} (relativity ~{np.exp(x.coef_py):.1e})"))
    for comp, g in c[~sep].groupby("component"):
        d = (g.coef_py - g.coef_r).abs()
        worst = g.loc[d.idxmax(), "param"]
        add("coefficients", comp, d.max() <= tol["tol_coef_python_vs_r"], f"max abs diff {d.max():.5f} ({worst})")

    kp, kr = rd(py_dir, "ev_m2_component_totals.csv"), rd(r_dir, "ev_m2_component_totals.csv")
    k = kp.merge(kr, on="component", suffixes=("_py", "_r"), validate="1:1")
    for _, x in k.iterrows():
        d = abs(x.pred_total_py / x.pred_total_r - 1)
        add("component predictions", x.component, d <= tol["tol_component_total_rel"],
            f"py {x.pred_total_py:,.0f} r {x.pred_total_r:,.0f} rel diff {d:.2e}")

    op, orr = rd(py_dir, "ev_m1_oneway_scores.csv"), rd(r_dir, "ev_m1_oneway_scores.csv")
    o = op.merge(orr, on=["group", "factor"], suffixes=("_py", "_r"), validate="1:1")
    d = (o.d2_gain_py - o.d2_gain_r).abs()
    add("one-way scores", f"{len(o)} factors, d2_gain", d.max() <= 1e-6, f"max abs diff {d.max():.2e}")
    add("one-way scores", "levels per factor", (o.n_levels_py == o.n_levels_r).all())

    tp, tr = rd(py_dir, "ev_m1_oneway_tables.csv"), rd(r_dir, "ev_m1_oneway_tables.csv")
    t = tp.merge(tr, on=["factor", "level"], suffixes=("_py", "_r"), how="outer", indicator=True)
    add("one-way tables", "same factor levels", (t._merge == "both").all(), f"{len(t)} rows")
    t = t[t._merge == "both"]
    for col in ["exposure", "n_claims", "loss_cost", "relativity", "rel_2021_22", "rel_2023_24"]:
        a, b = t[f"{col}_py"].to_numpy(float), t[f"{col}_r"].to_numpy(float)
        ok = np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        add("one-way tables", col, ok, f"max abs diff {np.nanmax(np.abs(a - b)):.2e}")

    dp, dr = rd(py_dir, "ev_m0_design_summary.csv"), rd(r_dir, "ev_m0_design_summary.csv")
    add("design", "rows, parameters and reference levels", dp.astype(str).equals(dr.astype(str)))

    # 3. Reproduction of the exploratory session's numbers (Python results).
    ref = cfg["exploratory_reference"]
    for model, v in ref["split_test_d2"].items():
        got = float(sp.loc[sp.model == model, "d2"].iloc[0])
        add("vs exploratory", f"split D2: {model}", abs(got - v) <= tol["tol_vs_exploratory"],
            f"exploratory {v:.4f} pack {got:.4f}")
    for f, v in ref["oneway_d2_gain"].items():
        got = float(op.loc[op.factor == f, "d2_gain"].iloc[0])
        add("vs exploratory", f"one-way gain: {f}", abs(got - v) <= tol["tol_vs_exploratory"],
            f"exploratory {v:.4f} pack {got:.4f}")

    report = pd.DataFrame(rows, columns=["check", "item", "result", "detail"])
    n_fail = int((report.result == "FAIL").sum())
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = ev / "reconciliation_report.md"
    with open(out, "w") as f:
        f.write(f"# Reconciliation report\n\nGenerated {stamp}. {len(report)} checks, {n_fail} failed, "
                f"{(report.result == 'WARN').sum()} warnings (see audit A11).\n\n")
        f.write("| Check | Item | Result | Detail |\n|---|---|---|---|\n")
        for r in report.itertuples(index=False):
            f.write(f"| {r.check} | {r.item} | {r.result} | {r.detail} |\n")
    rec = common.new_run_record("reconcile", cfg)
    rec.update(status="success" if n_fail == 0 else "failed", n_checks=len(report), n_failed=n_fail,
               report_sha256=common.sha256_file(out), finished_utc=stamp)
    common.append_run_log(cfg, rec)
    print(report.groupby(["check", "result"]).size().to_string())
    if n_fail:
        print(report[report.result == "FAIL"].to_string(index=False))
    print(f"\n{len(report)} checks, {n_fail} failed. Report: {out.relative_to(common.PACK_ROOT)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
