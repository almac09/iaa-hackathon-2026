"""Location information beyond region: residual checks and feature tests under the two-year rule (audit A26).

The address table has no coordinates, postcode or street, and address_id carries no geography (every ID decile has
the same region mix), so true geocoding is not possible. What does vary by location below region is each address's
weather: region explains 59-97% of an address's 2021-25 average, and rainfall at one address correlates 0.81-0.94
from year to year, so the averages work as a climate fingerprint of the place.

  L0 residuals: actual / expected of the production model (without CANN) on 2023 and 2024 by region, region x urban
     density, and quintiles of each address's climate, with z = (A - E) / sd from a Tweedie variance phi mu^p.
  L1 yearly weather (rainfall, freezing days, wind exposure, hail days, mean temperature of the row's year), as
     cubic splines, added to the contents and motor small-claim GLMs (buildings already has it).
  L2 address climate (the same five, averaged over 2021-25), added to the contents and motor GLMs, and to the
     buildings XGBoost features with the baseline's tree count.
  L3 a region x urban density factor added to the contents and motor GLMs.
Optional additions, so adopted only if they pass the two-year rule and the two-year mean gain is significant (D-23).

Usage (from the pack root, after python/run_glm_lines.py):
  python python/geo_tests.py --run-label "A26 geolocation"
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
import run_glm_lines as rgl  # noqa: E402
from combined_tests import baseline_build, window_preds  # noqa: E402
from gap_tests import compare, settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "geo_tests"
WEATHER = ["rainfall_mm", "freezing_days", "wind_exposure_index", "hail_days", "mean_temperature_c"]
CLIMATE = [f"clim_{c}" for c in WEATHER]


def climate(con) -> pd.DataFrame:
    return con.execute(f"""SELECT address_id, {", ".join(f"avg({c}) AS clim_{c}" for c in WEATHER)}
                           FROM weather GROUP BY address_id""").df()


def variant(line: rgl.Line, df: pd.DataFrame, cfg: dict) -> rgl.Line:
    """A copy of `line` with new columns or settings, starting from plain exposure (D-22 sets it per window)."""
    new = rgl.Line(line.name, df.assign(exposure=df.exposure_plain), line.claims, cfg)
    new.expected, new.month_days, new.claim_months = line.expected, line.month_days, line.claim_months
    return new


def add_numeric(cfg: dict, line: str, cols: list[str], xgb: bool = False) -> dict:
    out = copy.deepcopy(cfg)
    if xgb:
        out["lines"]["models"][line]["numeric"] = out["lines"]["models"][line]["numeric"] + cols
    else:
        out["glm_lines"]["models"][line]["numeric"] = out["glm_lines"]["models"][line]["numeric"] + cols
    return out


def add_categorical(cfg: dict, line: str, col: str) -> dict:
    out = copy.deepcopy(cfg)
    m = out["glm_lines"]["models"][line]
    m["categorical"] = rgl.resolve(out, m["categorical"]) + [col]
    return out


def residuals(line: rgl.Line, preds: dict) -> list[dict]:
    """Actual / expected by location group for each backtest year."""
    rows = []
    for w, mu in preds.items():
        yv = rgl.WINDOWS[w][1]
        v = line.rows([yv])
        y = v.y_full.to_numpy()
        phi = float(((y - mu) ** 2 / mu ** line.p).mean())
        frame = pd.DataFrame({"y": y, "mu": mu, "var": phi * mu ** line.p, "region": v.region.to_numpy(),
                              "region_urban": (v.region + "_" + v.urban_density).to_numpy(),
                              **{f"{c}_quintile": pd.qcut(v[c], 5, labels=False, duplicates="drop").to_numpy() for c in CLIMATE}})
        for key in ["region", "region_urban"] + [f"{c}_quintile" for c in CLIMATE]:
            g = frame.groupby(key).agg(actual=("y", "sum"), expected=("mu", "sum"), var=("var", "sum"), rows=("y", "size"))
            for level, r in g.iterrows():
                rows.append({"line": line.name, "score_year": yv, "grouping": key, "level": str(level),
                             "rows": int(r.rows), "actual_over_expected": r.actual / r.expected,
                             "z": (r.actual - r.expected) / np.sqrt(r["var"])})
    return rows


def main() -> None:
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    out_dir = common.PACK_ROOT / cfg["paths"]["evidence_dir"] / "glm_lines"
    try:
        rec["inputs_sha256"] = common.verify_inputs(cfg)
        summary = pd.read_csv(out_dir / "ev_g6_summary.csv")
        con = common.open_duckdb(cfg)
        lines = rgl.prepare(con, cfg, rec)[0]
        clim = climate(con)
        regions = con.execute("SELECT address_id, region, urban_density FROM addresses").df()
        con.close()
        results, resid = [], []
        for name, line in lines.items():
            s = settings(line, summary)
            trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
            df = line.df.merge(clim, on="address_id", how="left")
            if "region" not in df or "urban_density" not in df:
                df = df.drop(columns=[c for c in ("region", "urban_density") if c in df]).merge(regions, on="address_id", how="left")
            df["region_urban"] = df.region.astype(str) + "_" + df.urban_density.astype(str)
            if df[CLIMATE].isna().any().any():
                raise ValueError(f"{name}: rows without an address climate")
            base_line = variant(line, df, line.cfg)
            base = window_preds(base_line, s, baseline_build(s, trees))
            resid += residuals(base_line, base)
            print(f"{name}: residuals done", flush=True)
            tests = []
            if line.spec["small"] == "glm":
                tests += [("L1 yearly weather", "five weather splines", add_numeric(line.cfg, name, WEATHER)),
                          ("L2 address climate", "five climate splines", add_numeric(line.cfg, name, CLIMATE)),
                          ("L3 region x urban", "region x urban density factor", add_categorical(line.cfg, name, "region_urban"))]
            else:
                tests += [("L2 address climate", "five climate features (baseline trees)", add_numeric(line.cfg, name, CLIMATE, xgb=True))]
            for test, label, vcfg in tests:
                results += compare(base_line, base, window_preds(variant(line, df, vcfg), s, baseline_build(s, trees)),
                                   test, label, cfg)
        res, rs = pd.DataFrame(results), pd.DataFrame(resid)
        outputs = [write_csv(rs, out_dir / "ev_geo1_location_residuals.csv"),
                   write_csv(res, out_dir / "ev_geo2_location_feature_tests.csv")]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        flagged = rs.assign(abs_z=rs.z.abs()).groupby(["line", "grouping", "level"]).agg(
            min_abs_z=("abs_z", "min"), ae_2023=("actual_over_expected", "first"), ae_2024=("actual_over_expected", "last"),
            z_2023=("z", "first"), z_2024=("z", "last")).reset_index()
        flagged = flagged[(flagged.min_abs_z > 2) & (np.sign(flagged.z_2023) == np.sign(flagged.z_2024))]
        rec["metrics"] = {"feature_tests": res.round(6).to_dict("records"), "residual_groups_off_both_years": len(flagged)}
        rec["status"] = "success"
        print("Location groups off in the same direction on both years with |z| > 2:")
        print(flagged.round(3).to_string(index=False) if len(flagged) else "  none")
        print(res[["line", "test", "variant", "score_year", "d2_baseline", "d2_variant", "d2_gain", "ci_low_95", "ci_high_95",
                   "passes_two_year_rule", "passes_significance_rule"]].round(5).to_string(index=False))
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
