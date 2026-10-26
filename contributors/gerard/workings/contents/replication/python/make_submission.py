"""Kaggle submission files with contents predictions from GLM A and from XGBoost (banded).

Each model is refitted on every training year and predicts the 2025 contents rows. Rows of other
lines get config submission.other_lines_value. Row order and row_ids follow the sample submission.

Usage (from the pack root):  python python/make_submission.py
Needs the competition's sample_submission.csv at config submission.sample_file.
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
import xgb_models  # noqa: E402

RUNNER = "submission"


def predict_contents(cfg: dict, feats: pd.DataFrame) -> dict[str, pd.Series]:
    mc, xc, sc = cfg["model"], cfg["xgboost"], cfg["submission"]
    p = mc["tweedie_power"]
    train, test = feats[feats.src == "train"], feats[feats.src == "test"]
    e_train, e_test = train.exposure.to_numpy(), test.exposure.to_numpy()
    y = train.claim_cost.to_numpy()

    levels = models.factor_levels(train, mc["factors"])
    X_train, _ = models.design_matrix(train, levels, mc["numeric"])
    X_test, _ = models.design_matrix(test, levels, mc["numeric"])
    glm = models.fit_tweedie(X_train, y, e_train, p, mc["alpha"])

    xgb_models.check_features(mc["numeric"] + mc["factors"], cfg)
    cats = xgb_models.category_levels(train, mc["factors"])
    base = y.sum() / e_train.sum()
    xs = sc["xgb_banded"]
    params = {**xc["fixed_params"], "max_depth": xs["max_depth"], "min_child_weight": xs["min_child_weight"]}
    booster = xgb_models.fit(xgb_models.matrix(train, mc["numeric"], cats), y, e_train, base, p, params,
                             xs["n_trees"], xc["seed"], xc["nthread"])

    return {"glm_a": pd.Series(glm.predict(X_test) * e_test, index=test.row_id),
            "xgb_banded": pd.Series(xgb_models.predict(booster, xgb_models.matrix(test, mc["numeric"], cats),
                                                       e_test, base), index=test.row_id)}


def build_submission(sample: pd.DataFrame, preds: dict[str, pd.Series],
                     other_value: float | None = None) -> pd.DataFrame:
    """Predictions in the sample's row order. `preds` maps a row_id prefix (line) to its predictions by row_id;
    rows of any other line get `other_value`, which must then be given."""
    pred = pd.Series(np.nan, index=sample.index)
    covered = pd.Series(False, index=sample.index)
    for prefix, p in preds.items():
        rows = sample.row_id.str.startswith(prefix)
        if set(sample.row_id[rows]) != set(p.index) or p.index.duplicated().any():
            raise ValueError(f"Line {prefix} row_ids differ from the sample submission's.")
        pred[rows] = sample.row_id[rows].map(p)
        covered |= rows
    if not covered.all():
        if other_value is None:
            raise ValueError(f"{(~covered).sum()} rows have no prediction and no fill value was given.")
        pred[~covered] = other_value
    out = pd.DataFrame({"row_id": sample.row_id, "prediction": pred.to_numpy()})
    # Tweedie deviance is infinite at a zero prediction for a row with a claim, so refuse it.
    if out.prediction.isna().any() or not (out.prediction > 0).all():
        raise ValueError("Every prediction must be present and strictly positive.")
    return out


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    sc = cfg["submission"]
    rec = common.new_run_record(RUNNER, cfg)
    try:
        sample_path = common.PACK_ROOT / sc["sample_file"]
        rec["sample_sha256"] = common.sha256_file(sample_path)
        sample = pd.read_csv(sample_path, dtype={"row_id": str})
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        con = common.open_duckdb(cfg)
        rec["sql_sha256"] = common.run_sql_files(con, cfg)
        feats = con.execute("SELECT * FROM contents_features ORDER BY row_id").df()
        con.close()

        out_dir = common.PACK_ROOT / sc["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.date.today().strftime("%Y%m%d")
        train = feats[feats.src == "train"]
        test_expo = feats.loc[feats.src == "test", "exposure"].sum()
        rec["outputs_sha256"], rec["metrics"] = {}, {}
        for name, contents in predict_contents(cfg, feats).items():
            sub = build_submission(sample, {sc["contents_prefix"]: contents}, sc["other_lines_value"])
            path = out_dir / f"submission_contents_{name}_{stamp}.csv"
            sub.round(6).to_csv(path, index=False)
            rec["outputs_sha256"][path.relative_to(common.PACK_ROOT).as_posix()] = common.sha256_file(path)
            rec["metrics"][name] = {"contents_rows": int(len(contents)), "contents_total": round(contents.sum(), 2),
                                    "contents_rate_per_exposure": round(contents.sum() / test_expo, 4),
                                    "min": round(contents.min(), 6), "max": round(contents.max(), 2)}
            print(f"{path.name}: {len(sub)} rows, contents rate per exposure "
                  f"{contents.sum() / test_expo:.2f}, min {contents.min():.4f}, max {contents.max():.2f}")
        rates = train.groupby("yr").apply(lambda d: d.claim_cost.sum() / d.exposure.sum(), include_groups=False)
        rec["metrics"]["training_rate_by_year"] = rates.round(2).to_dict()
        print("Training claim cost per exposure by year:", rates.round(2).to_dict())
        rec["status"] = "success"
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
