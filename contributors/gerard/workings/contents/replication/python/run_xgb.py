"""XGBoost runner for the contents replication pack (audit A13).

Steps: verify input hashes -> run the SQL layer -> refit GLM A as the benchmark -> for each feature
set, tune on (fit years minus the last, early-stopped on the last) and refit on all fit years ->
score the score year once -> paired bootstrap against GLM A -> append a record to audit/run_log.jsonl.

Usage (from the pack root):  python python/run_xgb.py
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
import models  # noqa: E402
import xgb_models  # noqa: E402

RUNNER = "xgboost"
GLM_A = "A single Tweedie"


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.round(6).to_csv(path, index=False)
    return path


def load_rows(cfg: dict, rec: dict) -> pd.DataFrame:
    """Training rows: the banded features plus the raw columns of the 'raw' feature set."""
    rec["inputs_sha256"] = common.verify_inputs(cfg)
    con = common.open_duckdb(cfg)
    rec["sql_sha256"] = common.run_sql_files(con, cfg)
    raw = cfg["xgboost"]["feature_sets"]["raw"]
    cols = ", ".join(f"m.{c}" for c in raw["numeric"] + raw["categorical"])
    df = con.execute(f"SELECT f.*, {cols} FROM contents_features f JOIN contents_model m USING (row_id) "
                     "WHERE f.src = 'train' ORDER BY f.row_id").df()
    con.close()
    return df


def deciles(model: str, y: np.ndarray, pred: np.ndarray, expo: np.ndarray) -> pd.DataFrame:
    """Actual vs expected by decile of predicted rate (equal row counts)."""
    d = pd.DataFrame({"y": y, "pred": pred, "e": expo})
    d["decile"] = pd.qcut((pred / expo).argsort().argsort(), 10, labels=False) + 1
    g = d.groupby("decile").agg(exposure=("e", "sum"), actual=("y", "sum"), expected=("pred", "sum")).reset_index()
    return g.assign(model=model, a_over_e=g.actual / g.expected)


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / RUNNER
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        feats = load_rows(cfg, rec)
        mc, xc = cfg["model"], cfg["xgboost"]
        p = mc["tweedie_power"]
        fit = feats[feats.yr <= mc["fit_year_max"]]
        val = feats[feats.yr == mc["score_year"]]
        t_fit = feats[feats.yr <= xc["tune_fit_year_max"]]
        t_val = feats[feats.yr == xc["tune_val_year"]]
        if not (xc["tune_fit_year_max"] < xc["tune_val_year"] <= mc["fit_year_max"] < mc["score_year"]):
            raise ValueError("Tuning years must lie inside the fit years and before the score year.")
        if (feats.exposure <= 0).any():
            raise ValueError("Exposure must be positive for the log offset.")
        y_val, e_val = val.claim_cost.to_numpy(), val.exposure.to_numpy()
        base = fit.claim_cost.sum() / fit.exposure.sum()
        t_base = t_fit.claim_cost.sum() / t_fit.exposure.sum()

        # Benchmark: GLM A exactly as in run_all.py.
        levels = models.factor_levels(feats, mc["factors"])
        X_fit, _ = models.design_matrix(fit, levels, mc["numeric"])
        X_val, _ = models.design_matrix(val, levels, mc["numeric"])
        glm = models.fit_tweedie(X_fit, fit.claim_cost.to_numpy(), fit.exposure.to_numpy(), p, mc["alpha"])
        preds = {"exposure only": base * e_val, GLM_A: glm.predict(X_val) * e_val}
        pack_split = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "python" / "ev_m2_split_test.csv"
        if pack_split.exists():
            ref = pd.read_csv(pack_split).set_index("model").d2[GLM_A]
            rec["glm_a_vs_pack_d2_diff"] = float(models.d2_tweedie(y_val, preds[GLM_A], p) - ref)

        sets = {"banded": (mc["numeric"], mc["factors"]),
                "raw": (xc["feature_sets"]["raw"]["numeric"], xc["feature_sets"]["raw"]["categorical"])}
        tuning, importance, chosen = [], [], {}
        for name, (numeric, categorical) in sets.items():
            xgb_models.check_features(numeric + categorical, cfg)
            cats = xgb_models.category_levels(feats, categorical)
            X = {k: xgb_models.matrix(d, numeric, cats) for k, d in
                 (("t_fit", t_fit), ("t_val", t_val), ("fit", fit), ("val", val))}
            grid = [dict(zip(xc["grid"], v)) for v in itertools.product(*xc["grid"].values())]
            rows = []
            for g in grid:
                params = {**xc["fixed_params"], **g}
                b = xgb_models.fit(X["t_fit"], t_fit.claim_cost.to_numpy(), t_fit.exposure.to_numpy(), t_base, p,
                                   params, xc["max_rounds"], xc["seed"], xc["nthread"],
                                   val=(X["t_val"], t_val.claim_cost.to_numpy(), t_val.exposure.to_numpy()),
                                   early_stopping=xc["early_stopping_rounds"])
                n_trees = b.best_iteration + 1
                pred = xgb_models.predict(b, X["t_val"], t_val.exposure.to_numpy(), t_base, n_trees)
                rows.append({"feature_set": name, **g, "n_trees": n_trees,
                             "hit_max_rounds": n_trees == xc["max_rounds"],
                             "d2_tune_val": models.d2_tweedie(t_val.claim_cost.to_numpy(), pred, p),
                             "pred_over_actual_tune_val": pred.sum() / t_val.claim_cost.sum()})
            tuning += rows
            best = max(rows, key=lambda r: r["d2_tune_val"])
            params = {**xc["fixed_params"], **{k: best[k] for k in xc["grid"]}}
            b = xgb_models.fit(X["fit"], fit.claim_cost.to_numpy(), fit.exposure.to_numpy(), base, p,
                               params, best["n_trees"], xc["seed"], xc["nthread"])
            label = f"XGBoost {name}"
            preds[label] = xgb_models.predict(b, X["val"], e_val, base)
            chosen[label] = {"n_features": len(numeric) + len(categorical), "n_trees": best["n_trees"], **params}
            gain = pd.Series(b.get_score(importance_type="total_gain")).reindex(numeric + categorical, fill_value=0.0)
            importance.append(pd.DataFrame({"feature_set": name, "feature": gain.index,
                                            "gain_share": gain / gain.sum()}))

        # d2_level_matched rescales each model to the actual score-year total. It uses the outcome, so it
        # is a diagnostic only: it separates risk ranking from the year-on-year level shift (audit A13).
        split = pd.DataFrame([{"model": k, "d2": models.d2_tweedie(y_val, v, p),
                               "pred_over_actual": v.sum() / y_val.sum(),
                               "d2_level_matched": models.d2_tweedie(y_val, v * y_val.sum() / v.sum(), p),
                               **chosen.get(k, {})}
                              for k, v in preds.items()])
        outputs = [write_csv(split, out_dir / "ev_x1_split_test.csv"),
                   write_csv(pd.DataFrame(tuning).sort_values(["feature_set", "max_depth", "min_child_weight"]),
                             out_dir / "ev_x2_tuning.csv"),
                   write_csv(pd.concat(importance).sort_values(["feature_set", "gain_share"], ascending=[True, False]),
                             out_dir / "ev_x3_importance.csv")]

        bs = xc["bootstrap"]
        boot = []
        for label in chosen:
            diff = xgb_models.paired_bootstrap_d2(y_val, preds[label], preds[GLM_A], p, bs["n"], bs["seed"])
            boot.append({"model": label, "vs": GLM_A, "d2_diff": models.d2_tweedie(y_val, preds[label], p)
                         - models.d2_tweedie(y_val, preds[GLM_A], p), "boot_mean": diff.mean(),
                         "boot_p2_5": np.percentile(diff, 2.5), "boot_p97_5": np.percentile(diff, 97.5),
                         "share_not_better": (diff <= 0).mean(), "n_boot": bs["n"]})
        boot = pd.DataFrame(boot)
        outputs.append(write_csv(boot, out_dir / "ev_x4_bootstrap_vs_glm_a.csv"))
        dec = pd.concat([deciles(k, y_val, v, e_val) for k, v in preds.items() if k != "exposure only"])
        outputs.append(write_csv(dec[["model", "decile", "exposure", "actual", "expected", "a_over_e"]],
                                 out_dir / "ev_x5_deciles.csv"))

        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {"split_test_d2": dict(zip(split.model, split.d2.round(6))),
                          "bootstrap_vs_glm_a": {r.model: {"d2_diff": round(r.d2_diff, 6),
                                                           "p2_5": round(r.boot_p2_5, 6),
                                                           "p97_5": round(r.boot_p97_5, 6)}
                                                 for r in boot.itertuples()}}
        rec["status"] = "success"
        print(split[["model", "d2", "pred_over_actual", "d2_level_matched"]].round(4).to_string(index=False))
        print(boot.round(4).to_string(index=False))
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
