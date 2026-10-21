"""Three-line XGBoost runner and combined submission (audit A15).

For each of buildings, contents and motor, on the developed and capped target (sql/09-11):
  tune   fit 2021-22, early-stop on 2023, grid over depth and leaf size;
  test   fit 2021-23 with the backfitted year trend, score 2024 (also without trend, and exposure only);
  final  fit 2021-24 with the trend, predict the 2025 rows, apply the large-claim loading.
Then one submission with every line from its own model, and a record in audit/run_log.jsonl.

Usage (from the pack root):  python python/run_lines.py
Needs the competition's sample_submission.csv at config submission.sample_file.
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
import make_submission  # noqa: E402
import models  # noqa: E402
import xgb_models  # noqa: E402
from run_xgb import deciles, write_csv  # noqa: E402

RUNNER = "lines"


def features(cfg: dict, spec: dict) -> tuple[list[str], list[str]]:
    resolve = lambda v: cfg["model"][v.split(".")[1]] if isinstance(v, str) else v  # noqa: E731
    return resolve(spec["numeric"]), resolve(spec["categorical"])


def median_hessian(y: np.ndarray, expo: np.ndarray, base: float, p: float) -> float:
    """Median Tweedie hessian per row at the base rate: the unit in which min_child_weight is set."""
    mu = base * expo
    return float(np.median(-y * (1 - p) * mu ** (1 - p) + (2 - p) * mu ** (2 - p)))


class Line:
    """One line's data split by year, with its feature matrices and targets."""

    def __init__(self, name: str, df: pd.DataFrame, numeric: list[str], categorical: list[str]):
        self.name, self.numeric = name, numeric
        self.train, self.test = df[df.src == "train"], df[df.src == "test"]
        self.cats = xgb_models.category_levels(self.train, categorical)

    def part(self, years):
        d = self.train[self.train.yr.isin(years)]
        return d, xgb_models.matrix(d, self.numeric, self.cats)

    @staticmethod
    def base(d: pd.DataFrame) -> float:
        return d.claim_cost_dev.sum() / d.exposure.sum()

    @staticmethod
    def load(d: pd.DataFrame, apply: bool) -> float:
        """Large-claim loading (D-12): the excess over the cap, spread over the line in proportion to cost."""
        return 1 + d.excess_dev.sum() / d.claim_cost_dev.sum() if apply else 1.0


def run_line(line: Line, cfg: dict, p: float) -> dict:
    xc, lc = cfg["xgboost"], cfg["lines"]
    n_iter = lc["trend_iterations"] if cfg["decisions"]["D13_trend_to_2025"]["apply"] else 0
    cap = cfg["decisions"]["D12_cap_large_claims"]["apply"]
    seed, nthread = xc["seed"], xc["nthread"]
    out = {"tuning": [], "backtest": [], "trend": []}

    # Tuning: never sees 2024.
    t_fit, X_tf = line.part([2021, 2022])
    t_val, X_tv = line.part([2023])
    t_base = line.base(t_fit)
    mh = median_hessian(t_fit.claim_cost_dev.to_numpy(), t_fit.exposure.to_numpy(), t_base, p)
    for depth, rows in itertools.product(lc["grid"]["max_depth"], lc["min_child_rows"]):
        params = {**xc["fixed_params"], "max_depth": depth, "min_child_weight": rows * mh}
        b = xgb_models.fit(X_tf, t_fit.claim_cost_dev.to_numpy(), t_fit.exposure.to_numpy(), t_base, p, params,
                           xc["max_rounds"], seed, nthread,
                           val=(X_tv, t_val.claim_cost_dev.to_numpy(), t_val.exposure.to_numpy()),
                           early_stopping=xc["early_stopping_rounds"])
        n_trees = b.best_iteration + 1
        pred = xgb_models.predict(b, X_tv, t_val.exposure.to_numpy(), t_base, n_trees)
        out["tuning"].append({"line": line.name, "max_depth": depth, "min_child_rows": rows,
                              "min_child_weight": rows * mh, "n_trees": n_trees,
                              "hit_max_rounds": n_trees == xc["max_rounds"],
                              "d2_2023_developed": models.d2_tweedie(t_val.claim_cost_dev.to_numpy(), pred, p)})
        print(f"  {line.name} tune depth {depth} rows {rows}: {n_trees} trees, "
              f"D2 {out['tuning'][-1]['d2_2023_developed']:.4f}", flush=True)
    best = max(out["tuning"], key=lambda r: r["d2_2023_developed"])
    params = {**xc["fixed_params"], "max_depth": best["max_depth"], "min_child_weight": best["min_child_weight"]}
    n_trees = best["n_trees"]

    def fit_trend(d, X, stage):
        booster, slope, centre, hist = xgb_models.fit_with_trend(
            X, d.claim_cost_dev.to_numpy(), d.exposure.to_numpy(), d.yr.to_numpy(), line.base(d), p, params,
            n_trees, seed, nthread, n_iter=n_iter)
        out["trend"] += [{"line": line.name, "stage": stage, "centre": centre, **h} for h in hist]
        return booster, slope, centre

    # Backtest: fit 2021-23, score 2024 as the 2025 prediction will be made.
    b_fit, X_bf = line.part([2021, 2022, 2023])
    b_val, X_bv = line.part([2024])
    e_bv = b_val.exposure.to_numpy()
    y_rep, y_full = b_val.claim_cost.to_numpy(), (b_val.claim_cost_dev + b_val.excess_dev).to_numpy()
    load, base = line.load(b_fit, cap), line.base(b_fit)
    booster, slope, centre = fit_trend(b_fit, X_bf, "backtest")
    flat = xgb_models.fit(X_bf, b_fit.claim_cost_dev.to_numpy(), b_fit.exposure.to_numpy(), base, p, params,
                          n_trees, seed, nthread)
    preds = {"exposure only": base * e_bv * load,
             "XGBoost, no trend": xgb_models.predict(flat, X_bv, e_bv, base) * load,
             "XGBoost, with trend": xgb_models.predict(booster, X_bv, e_bv, base,
                                                       extra=np.full(len(e_bv), slope * (2024 - centre))) * load}
    for k, v in preds.items():
        out["backtest"].append({"line": line.name, "model": k, "d2_vs_reported": models.d2_tweedie(y_rep, v, p),
                                "d2_vs_developed": models.d2_tweedie(y_full, v, p),
                                "pred_over_developed": v.sum() / y_full.sum(),
                                "trend_pa": np.expm1(slope) if "with trend" in k else np.nan})
    out["deciles"] = deciles(f"{line.name}: XGBoost, with trend", y_full, preds["XGBoost, with trend"], e_bv)

    # Final: fit 2021-24, predict 2025.
    X_all = xgb_models.matrix(line.train, line.numeric, line.cats)
    X_test = xgb_models.matrix(line.test, line.numeric, line.cats)
    load, base = line.load(line.train, cap), line.base(line.train)
    booster, slope, centre = fit_trend(line.train, X_all, "final")
    e_test = line.test.exposure.to_numpy()
    pred = xgb_models.predict(booster, X_test, e_test, base, extra=np.full(len(e_test), slope * (2025 - centre))) * load
    gain = pd.Series(booster.get_score(importance_type="total_gain")).reindex(line.numeric + list(line.cats),
                                                                             fill_value=0.0)
    out["importance"] = pd.DataFrame({"line": line.name, "feature": gain.index, "gain_share": gain / gain.sum()})
    last = line.train[line.train.yr == 2024]
    rate_2024 = (last.claim_cost_dev + last.excess_dev).sum() / last.exposure.sum()
    out["summary"] = {"line": line.name, "n_test_rows": len(pred), "n_features": len(line.numeric) + len(line.cats),
                      "max_depth": params["max_depth"], "min_child_rows": best["min_child_rows"], "n_trees": n_trees,
                      "trend_pa": np.expm1(slope), "large_claim_load": load,
                      "rate_2025": pred.sum() / e_test.sum(), "developed_rate_2024": rate_2024,
                      "rate_2025_over_2024": pred.sum() / e_test.sum() / rate_2024,
                      "pred_min": pred.min(), "pred_max": pred.max()}
    out["pred"] = pd.Series(pred, index=line.test.row_id.to_numpy())
    return out


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    lc, sc = cfg["lines"], cfg["submission"]
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / RUNNER
    try:
        sample_path = common.PACK_ROOT / sc["sample_file"]
        rec["sample_sha256"] = common.sha256_file(sample_path)
        sample = pd.read_csv(sample_path, dtype={"row_id": str})
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        rec["sql_sha256"] = common.run_sql_files(con, cfg)
        for rel in lc["sql_files"]:
            rec["sql_sha256"][rel] = common.sha256_file(common.PACK_ROOT / rel)
            con.execute((common.PACK_ROOT / rel).read_text())
        outputs = common.export_sql_evidence(con, out_dir, prefixes=("ev_09", "ev_10", "ev_11"))
        p = cfg["model"]["tweedie_power"]
        results = {}
        for name, spec in lc["models"].items():
            numeric, categorical = features(cfg, spec)
            xgb_models.check_features(numeric + categorical, cfg)
            df = con.execute(f"SELECT * FROM {spec['table']} ORDER BY row_id").df()
            print(f"{name}: {len(df)} rows, {len(numeric) + len(categorical)} features", flush=True)
            results[name] = run_line(Line(name, df, numeric, categorical), cfg, p)
        con.close()

        cat = lambda key: pd.concat([pd.DataFrame(r[key]) if isinstance(r[key], list) else r[key]  # noqa: E731
                                     for r in results.values()])
        summary = pd.DataFrame([r["summary"] for r in results.values()])
        outputs += [write_csv(cat("tuning"), out_dir / "ev_l1_tuning.csv"),
                    write_csv(cat("backtest"), out_dir / "ev_l2_backtest_2024.csv"),
                    write_csv(cat("trend"), out_dir / "ev_l3_trend_backfitting.csv"),
                    write_csv(cat("deciles")[["model", "decile", "exposure", "actual", "expected", "a_over_e"]],
                              out_dir / "ev_l4_backtest_deciles.csv"),
                    write_csv(cat("importance").sort_values(["line", "gain_share"], ascending=[True, False]),
                              out_dir / "ev_l5_importance.csv"),
                    write_csv(summary, out_dir / "ev_l6_summary.csv")]

        by_prefix = {lc["models"][k]["prefix"]: r["pred"] for k, r in results.items()}
        sub = make_submission.build_submission(sample, by_prefix)
        path = common.PACK_ROOT / lc["out_file"].format(date=dt.date.today().strftime("%Y%m%d"))
        path.parent.mkdir(parents=True, exist_ok=True)
        sub.round(6).to_csv(path, index=False)

        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o)
                                 for o in outputs + [path]}
        rec["metrics"] = {"summary": summary.round(6).to_dict("records"),
                          "backtest": cat("backtest").round(6).to_dict("records")}
        rec["status"] = "success"
        print(cat("backtest").round(4).to_string(index=False))
        print(summary.round(4).to_string(index=False))
        print(f"Submission: {path.relative_to(common.PACK_ROOT)} ({len(sub)} rows)")
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        import duckdb, sklearn, xgboost  # noqa: E401  versions recorded for the audit trail
        rec["versions"] = {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "pandas": pd.__version__,
                           "numpy": np.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
