"""Python runner for the contents replication pack.

Steps: verify input hashes -> run the SQL layer -> export SQL evidence -> one-way analysis ->
GLM structure comparison -> append a record to audit/run_log.jsonl.

Usage (from the pack root):  python python/run_all.py
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import models  # noqa: E402

RUNNER = "python"


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.round(6).to_csv(path, index=False)
    return path


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / RUNNER
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        rec["sql_sha256"] = common.run_sql_files(con, cfg)
        outputs = common.export_sql_evidence(con, out_dir)
        feats = con.execute("SELECT * FROM contents_features WHERE src = 'train'").df()
        con.close()

        mc = cfg["model"]
        p, alpha = mc["tweedie_power"], mc["alpha"]
        fit = feats[feats.yr <= mc["fit_year_max"]]
        val = feats[feats.yr == mc["score_year"]]

        # A07: one-way tables and out-of-time scores for every factor.
        tables, scores = [], []
        for group, factors in cfg["oneway"]["factors"].items():
            for f in factors:
                t = models.oneway_table(feats, f)
                tables.append(t)
                scores.append({"group": group, "factor": f, "n_levels": len(t),
                               "d2_gain": models.oneway_d2_gain(fit, val, f, p),
                               "rel_min": t.relativity.min(), "rel_max": t.relativity.max()})
        cols = ["factor", "level", "exposure", "exposure_pct", "n_claims", "freq_per_1000", "severity",
                "loss_cost", "relativity", "rel_2021_22", "rel_2023_24"]
        outputs.append(write_csv(pd.concat(tables)[cols].sort_values(["factor", "level"]),
                                 out_dir / "ev_m1_oneway_tables.csv"))
        scores = pd.DataFrame(scores).sort_values("factor")
        outputs.append(write_csv(scores, out_dir / "ev_m1_oneway_scores.csv"))

        # A09: four GLM structures with identical factors, fitted on fit years, scored on the score year.
        levels = models.factor_levels(feats, mc["factors"])
        X_fit, names = models.design_matrix(fit, levels, mc["numeric"])
        X_val, _ = models.design_matrix(val, levels, mc["numeric"])
        e_fit, e_val = fit.exposure.to_numpy(), val.exposure.to_numpy()
        y_val = val.claim_cost.to_numpy()
        coefs, preds, totals = [], {}, []

        def tweedie_component(name, target):
            m = models.fit_tweedie(X_fit, fit[target].to_numpy(), e_fit, p, alpha)
            coefs.append(models.coef_frame(name, m, names))
            pred = m.predict(X_val) * e_val
            totals.append({"component": name, "pred_total": pred.sum()})
            return pred

        base = fit.claim_cost.sum() / fit.exposure.sum()
        preds["exposure only"] = base * e_val
        preds["A single Tweedie"] = tweedie_component("A total", "claim_cost")
        preds["B per-peril Tweedie"] = sum(tweedie_component(f"B {k}", v[0]) for k, v in mc["perils"].items())
        preds["C per-size-band Tweedie"] = sum(tweedie_component(f"C {b}", b) for b in mc["size_bands"])
        d_parts = []
        for k, (cost_col, n_col) in mc["perils"].items():
            n = fit[n_col].to_numpy()
            fm = models.fit_poisson(X_fit, n, e_fit, alpha)
            has = n > 0
            sm = models.fit_gamma(X_fit[has], fit[cost_col].to_numpy()[has] / n[has], n[has], alpha)
            coefs += [models.coef_frame(f"D freq {k}", fm, names), models.coef_frame(f"D sev {k}", sm, names)]
            d_parts.append(fm.predict(X_val) * e_val * sm.predict(X_val))
            totals.append({"component": f"D {k}", "pred_total": d_parts[-1].sum()})
        preds["D per-peril freq x sev"] = sum(d_parts)
        outputs.append(write_csv(pd.DataFrame(totals).sort_values("component"),
                                 out_dir / "ev_m2_component_totals.csv"))

        split = pd.DataFrame([{"model": k, "d2": models.d2_tweedie(y_val, v, p), "pred_over_actual": v.sum() / y_val.sum()}
                              for k, v in preds.items()])
        outputs.append(write_csv(split, out_dir / "ev_m2_split_test.csv"))
        coefs = pd.concat(coefs)
        outputs.append(write_csv(coefs, out_dir / "ev_m2_coefficients.csv"))
        rel = coefs[coefs.component.str.startswith("C ")].copy()
        rel = rel[rel.param.str.split("=").str[0].isin(mc["relativity_params"])]
        rel = rel.assign(relativity=np.exp(rel.coef)).pivot(index="param", columns="component", values="relativity")
        outputs.append(write_csv(rel.reset_index(), out_dir / "ev_m2_band_relativities.csv"))
        design = pd.DataFrame({"item": ["n_fit_rows", "n_score_rows", "n_params_incl_intercept"] +
                                       [f"reference_level:{f}" for f in levels],
                               "value": [len(fit), len(val), len(names) + 1] + [lv[0] for lv in levels.values()]})
        outputs.append(write_csv(design, out_dir / "ev_m0_design_summary.csv"))

        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {"split_test_d2": dict(zip(split.model, split.d2.round(6))),
                          "oneway_top10": dict(scores.nlargest(10, "d2_gain")[["factor", "d2_gain"]]
                                               .round(6).itertuples(index=False))}
        rec["status"] = "success"
        print(split.round(4).to_string(index=False))
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        import duckdb, sklearn  # noqa: E401  versions recorded for the audit trail
        rec["versions"] = {"python": sys.version.split()[0], "duckdb": duckdb.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
