"""The final review's recommendations (A27, items 1-6), each tested under the pack's rules (audit A28).

  T1 address corrections (D-26 R11 elevation in feet, D-27 year built 1900): corrections, applied on correctness
     grounds like D-24 but measured here, jointly and one at a time, on every line.
  T2 buildings peril interactions (clay x dry summer, flood zone 3+ x rainfall) in the XGBoost, the large-claim GLM
     and both.
  T3 buildings prediction shape: a calibration power on the predicted rate and a factor for rows of exposure <= 0.1,
     each estimated on one backtest year and scored on the other.
  T4 motor own-line credibility k tuned on claims capped at 1m with a finer grid, against k = 8. Rule fixed before
     the run (#157): take the capped choice if 2024 improves significantly on the full target and 2023 improves both
     capped at 1m and with rows of 10m or more left out; otherwise k = 6.
  T5 contents direct-channel flag.
  T6 motor bodily-injury lift D-11 off. It changes the target, so both models are scored against both yardsticks
     (actual with the lift and without it).
Base for T2-T6: the production model without CANN, with the address corrections applied. Buildings comparisons
average the predictions of four XGBoost seeds (production seed first) with the tree count fixed at the base's tuned
count, so seed noise does not pass for a gain. Optional additions also need a significant two-year mean (D-23).

Usage (from the pack root, after python/run_glm_lines.py):
  python python/final_tests.py --tests T1 --run-label "A28 final run: T1"
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
import significance  # noqa: E402
from combined_tests import baseline_build, window_preds  # noqa: E402
from gap_tests import compare, settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "final_tests"
EVAL = rgl.EVAL_WINDOWS
SEEDS = [20261021, 11, 12, 13]
INTER = ["x_clay_dry", "x_fz3_rain"]
OUT = common.PACK_ROOT / "audit" / "evidence" / "glm_lines"
PROVENANCE: dict = {}  # SQL and evidence hashes written by prepare(), merged into the run record


def lines_for(cfg: dict) -> dict:
    con = common.open_duckdb(cfg)
    lines = rgl.prepare(con, cfg, PROVENANCE)[0]
    con.close()
    return lines


def with_decisions(cfg: dict, **apply) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in apply.items():
        out["decisions"][k]["apply"] = v
    return out


def variant(line: rgl.Line, cfg: dict) -> rgl.Line:
    new = rgl.Line(line.name, line.df.assign(exposure=line.df.exposure_plain), line.claims, cfg)
    new.expected, new.month_days, new.claim_months = line.expected, line.month_days, line.claim_months
    return new


def bagged(line: rgl.Line, s: dict, trees) -> dict:
    """Window predictions averaged over SEEDS for XGBoost lines (tree count fixed); one fit for GLM lines."""
    if line.spec["small"] != "xgb":
        return window_preds(line, s, baseline_build(s, trees))
    keep, runs = line.cfg["xgboost"]["seed"], []
    for seed in SEEDS:
        line.cfg["xgboost"]["seed"] = seed
        runs.append(window_preds(line, s, baseline_build(s, trees)))
    line.cfg["xgboost"]["seed"] = keep
    return {w: np.mean([r[w] for r in runs], axis=0) for w in EVAL}


def t1(cfg, summary):
    rows = []
    raw = lines_for(with_decisions(cfg, D26_r11_elevation_feet=False, D27_year_built_placeholder=False))
    fixed = lines_for(cfg)
    singles = {"R11 elevation only": lines_for(with_decisions(cfg, D27_year_built_placeholder=False)),
               "year built only": lines_for(with_decisions(cfg, D26_r11_elevation_feet=False))}
    for name, line in raw.items():
        s = settings(line, summary)
        trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
        base = bagged(line, s, trees)
        rows += compare(line, base, bagged(fixed[name], s, trees), "T1 address corrections", "both", cfg)
        if name == "buildings":
            for label, ln in singles.items():
                rows += compare(line, base, bagged(ln[name], s, trees), "T1 address corrections", label, cfg)
    return {"ev_f1_address_corrections.csv": pd.DataFrame(rows)}


def t2_t3(cfg, summary, which):
    out = {}
    line = lines_for(cfg)["buildings"]
    s = settings(line, summary)
    trees = line.tune_trees(s["threshold"])
    base = bagged(line, s, trees)
    if "T2" in which:
        rows = []
        for label, xgb_on, glm_on in [("XGBoost", True, False), ("large-claim GLM", False, True), ("both", True, True)]:
            vcfg = copy.deepcopy(cfg)
            if xgb_on:
                vcfg["lines"]["models"]["buildings"]["numeric"] = vcfg["lines"]["models"]["buildings"]["numeric"] + INTER
            if glm_on:
                vcfg["glm_lines"]["models"]["buildings"]["excess"]["continuous"] = INTER
            rows += compare(line, base, bagged(variant(line, vcfg), s, trees), "T2 peril interactions", label, cfg)
        out["ev_f2_peril_interactions.csv"] = pd.DataFrame(rows)
    if "T3" in which:
        se = 0.1
        ys = {w: line.rows([rgl.WINDOWS[w][1]]).y_full.to_numpy() for w in EVAL}
        ex = {w: line.rows([rgl.WINDOWS[w][1]]).exposure_plain.to_numpy() for w in EVAL}
        est = {w: rgl.fit_shape(ys[w], base[w], ex[w], line.p, se) for w in EVAL}
        other = {"tune": "backtest", "backtest": "tune"}
        rows, shapes = [], []
        for label, use_a, use_f in [("calibration power", True, False), ("short-row factor", False, True), ("both", True, True)]:
            alt = {w: rgl.shape_adjust(base[w], ex[w], est[other[w]][0] if use_a else 1.0,
                                       est[other[w]][1] if use_f else 1.0, se) for w in EVAL}
            rows += compare(line, base, alt, "T3 prediction shape", label, cfg)
        shapes = [{"score_year": rgl.WINDOWS[w][1], "power_fitted": est[w][0], "short_factor_fitted": est[w][1],
                   "short_rows": int((ex[w] <= se).sum())} for w in EVAL]
        out["ev_f3_prediction_shape.csv"] = pd.DataFrame(rows)
        out["ev_f3b_prediction_shape_estimates.csv"] = pd.DataFrame(shapes)
    return out


def t4(cfg, summary):
    line = lines_for(cfg)["motor"]
    s = settings(line, summary)
    years, val_year = rgl.WINDOWS["tune"]
    line.use_window("tune")
    val = line.rows([val_year])
    tune_model = line.build(years, s["threshold"], s["alpha"], None, trend=False)
    grid = [0.5, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, np.inf]
    y_cap = np.minimum(val.y_full.to_numpy(), 1e6)
    curve = []
    for k in grid:
        mu = line.predict(tune_model, "tune", {**s["ks"], "own": k}, val)
        curve.append({"k": k, "d2_2023_full": models.d2_tweedie(val.y_full.to_numpy(), mu, line.p),
                      "d2_2023_capped_1m": models.d2_tweedie(y_cap, mu, line.p)})
    curve = pd.DataFrame(curve)
    k_capped = float(curve.loc[curve.d2_2023_capped_1m.idxmax(), "k"])
    print(f"  motor k chosen on capped 2023: {k_capped}", flush=True)
    base = window_preds(line, s, baseline_build(s, None))
    rows = []
    for k in sorted({k_capped, 6.0}):
        alt = window_preds(line, {**s, "ks": {**s["ks"], "own": k}}, baseline_build(s, None))
        r = compare(line, base, alt, "T4 motor credibility", f"k = {k:g} (against k = 8)", cfg)
        for w, rr in zip(EVAL, r):
            v = line.rows([rgl.WINDOWS[w][1]])
            y = v.y_full.to_numpy()
            keep = y < 1e7
            rr["gain_capped_1m"] = models.d2_tweedie(np.minimum(y, 1e6), alt[w], line.p) - models.d2_tweedie(np.minimum(y, 1e6), base[w], line.p)
            rr["gain_without_10m_rows"] = models.d2_tweedie(y[keep], alt[w][keep], line.p) - models.d2_tweedie(y[keep], base[w][keep], line.p)
        rows += r
    res = pd.DataFrame(rows)
    capped = res[res.variant.str.startswith(f"k = {k_capped:g} ")]
    r23, r24 = capped[capped.score_year == 2023].iloc[0], capped[capped.score_year == 2024].iloc[0]
    rule_capped = bool(r24.ci_low_95 > 0 and r23.gain_capped_1m > 0 and r23.gain_without_10m_rows > 0)
    res["k_capped_choice"] = k_capped
    res["rule_takes_capped_choice"] = rule_capped
    res["k_adopted_by_rule"] = k_capped if rule_capped else 6.0
    print(f"  motor credibility: rule takes {'k = %g' % k_capped if rule_capped else 'k = 6'}", flush=True)
    return {"ev_f4_motor_credibility.csv": res, "ev_f4b_motor_k_curve_2023.csv": curve}


def t5(cfg, summary):
    line = lines_for(cfg)["contents"]
    s = settings(line, summary)
    base = window_preds(line, s, baseline_build(s, None))
    vcfg = copy.deepcopy(cfg)
    m = vcfg["glm_lines"]["models"]["contents"]
    m["categorical"] = rgl.resolve(vcfg, m["categorical"]) + ["f_direct"]
    rows = compare(line, base, window_preds(variant(line, vcfg), s, baseline_build(s, None)), "T5 contents channel", "direct flag", cfg)
    return {"ev_f5_contents_direct.csv": pd.DataFrame(rows)}


def t6(cfg, summary):
    on = lines_for(cfg)["motor"]
    off = lines_for(with_decisions(cfg, D11_ibner_motor_bi=False))["motor"]
    s = settings(on, summary)
    p_on, p_off = window_preds(on, s, baseline_build(s, None)), window_preds(off, s, baseline_build(s, None))
    bs, rows = cfg["glm_lines"]["bootstrap"], []
    for w in EVAL:
        yv = rgl.WINDOWS[w][1]
        v_on, v_off = on.rows([yv]), off.rows([yv])
        assert (v_on.row_id.to_numpy() == v_off.row_id.to_numpy()).all()
        for basis, y in (("actual with the lift", v_on.y_full.to_numpy()), ("actual without the lift", v_off.y_full.to_numpy())):
            diff = significance.paired_cluster_bootstrap(y, p_off[w], p_on[w], v_on.customer_id.to_numpy(), on.p, bs["n"], bs["seed"])
            gain = models.d2_tweedie(y, p_off[w], on.p) - models.d2_tweedie(y, p_on[w], on.p)
            rows.append({"score_year": yv, "yardstick": basis, "d2_lift_on_model": models.d2_tweedie(y, p_on[w], on.p),
                         "d2_lift_off_model": models.d2_tweedie(y, p_off[w], on.p), "pred_ratio_off_over_on": p_off[w].sum() / p_on[w].sum(),
                         "actual_total": y.sum(), **significance.summarise(diff, gain)})
        print(f"  D-11 {yv}: " + "; ".join(f"{r['yardstick']} {r['d2_gain']:+.5f}" for r in rows if r["score_year"] == yv), flush=True)
    return {"ev_f6_d11_off_two_yardsticks.csv": pd.DataFrame(rows)}


def main() -> None:
    args = sys.argv[1:]
    which = args[args.index("--tests") + 1].split(",") if "--tests" in args else ["T1", "T2", "T3", "T4", "T5", "T6"]
    unknown = set(which) - {"T1", "T2", "T3", "T4", "T5", "T6"}
    if unknown:
        raise SystemExit(f"unknown tests: {sorted(unknown)}")
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    rec["tests"] = which
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        summary = pd.read_csv(OUT / "ev_g6_summary.csv")
        frames = {}
        if "T1" in which:
            frames.update(t1(cfg, summary))
        if "T2" in which or "T3" in which:
            frames.update(t2_t3(cfg, summary, which))
        if "T4" in which:
            frames.update(t4(cfg, summary))
        if "T5" in which:
            frames.update(t5(cfg, summary))
        if "T6" in which:
            frames.update(t6(cfg, summary))
        outputs = [write_csv(df, OUT / name) for name, df in frames.items()]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {name: df.round(6).to_dict("records") for name, df in frames.items()}
        rec["status"] = "success"
        for name, df in frames.items():
            print(f"\n{name}\n{df.round(5).to_string(index=False)}", flush=True)
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost")}}
        rec.update(PROVENANCE)
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
