"""Scratch (A31 round 2): C0 itself with the data-harmonisation fixes of PREREGISTERED_RULE.md (R1, R2, R4).

Builds both markets once, then predicts C0 for the emerging 2024 halves and the 2025 test rows with:
  c0_mi       R1  fields missing completely at random imputed ten times from same-region emerging training rows
                  (x_clay_dry rebuilt from the imputed soil), predictions averaged;
  c0_vage     R2  motor vehicle_age minus 2 (floored at 0);
  c0_mi_vage  R1 and R2 together;
  c0_csy      R4  buildings 2025 rows with customer_since_year minus 1 (test rows only),
and runs R4's developed backtest (fit 2021-23, score 2024; customer_since_year as is against minus 1).
Writes r2_<line>.parquet (row_id, split, the prediction columns) and r2_backtest.json. Row-level: delete after use."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "python")
import common  # noqa: E402
import emerging as em  # noqa: E402
import run_glm_lines as rgl  # noqa: E402

HERE = Path(__file__).resolve().parent
K, SEED = 10, 2031
cfg = common.load_config()
ctx = em.Context(cfg, common.new_run_record("scratch_a31_r2", cfg))
P = em.P


def is_gap(s: pd.Series) -> np.ndarray:
    return (s.isna() | (s.astype("string") == "MISSING")).to_numpy()


def impute_predict(name, d, year, donors, rng):
    """R1: average C0 over K imputations of the MCAR fields, each drawn from same-region donor rows."""
    cols = list(ctx.fill_values(name))
    out = np.zeros(len(d))
    for _ in range(K):
        dk = d.copy()
        for col in cols:
            gap = is_gap(dk[col])
            if not gap.any():
                continue
            pool = donors[~is_gap(donors[col])]
            vals = np.array(dk[col].astype(object).to_numpy(), dtype=object)  # a writable copy
            for reg, idx in pd.Series(np.flatnonzero(gap)).groupby(dk.region.to_numpy()[gap]):
                src = pool.loc[pool.region == reg, col]
                src = src if len(src) else pool[col]
                vals[idx.to_numpy()] = src.to_numpy()[rng.integers(0, len(src), len(idx))]
            dk[col] = pd.Series(vals, index=dk.index).astype(d[col].dtype)
            if col == "soil_type" and "x_clay_dry" in dk:  # sql/11: x_clay_dry = (soil = clay) x dry summer index
                dk.loc[gap, "x_clay_dry"] = (dk.loc[gap, "soil_type"] == "clay").astype(float) * dk.loc[gap, "dry_summer_index"]
        out += ctx.models[name].predict(dk, year=year)
    return out / K


def vage(d):
    d = d.copy()
    d["vehicle_age"] = (d.vehicle_age - 2).clip(lower=0)
    return d


cache = {ln: pd.concat([pd.read_parquet(HERE / "cache" / f"emg_{ln}.parquet"),
                        pd.read_parquet(HERE / "cache" / "holdout_B" / f"emg_{ln}_B.parquet")]) for ln in em.BAND}
rng = np.random.default_rng(SEED)
for name in ("buildings", "contents", "motor"):
    line = ctx.line(name)
    split = cache[name].set_index("row_id").split
    region = cache[name].set_index("row_id").region
    frames = []
    for yr, rows, donors in ((2024, line.rows([2024]), line.rows([2023])),
                             (2025, line.test, line.rows([2023, 2024]))):
        d = rows.assign(region=region.reindex(rows.row_id).to_numpy())
        donors = donors.assign(region=region.reindex(donors.row_id).to_numpy())
        assert d.region.notna().all() and donors.region.notna().all()
        r = pd.DataFrame({"row_id": d.row_id.to_numpy(), "split": split.reindex(d.row_id).to_numpy(),
                          "c0": ctx.base(name, d, yr)})
        r["c0_mi"] = impute_predict(name, d, yr, donors, rng)
        if name == "motor":
            r["c0_vage"] = ctx.models[name].predict(vage(d), year=yr)
            r["c0_mi_vage"] = impute_predict(name, vage(d), yr, vage(donors), rng)
        if name == "buildings" and yr == 2025:
            r["c0_csy"] = ctx.models[name].predict(d.assign(customer_since_year=d.customer_since_year - 1), year=yr)
        frames.append(r)
    out = pd.concat(frames, ignore_index=True)
    ref = cache[name].set_index("row_id").c0.reindex(out.row_id).to_numpy()
    assert np.allclose(out.c0.to_numpy(), ref, rtol=1e-9), f"{name}: C0 does not match the cache"
    out.to_parquet(HERE / f"r2_{name}.parquet", index=False)
    print(name, {c: round(float(out[c].mean() / out.c0.mean()), 4) for c in out.columns if c.startswith("c0_")}, flush=True)

# R4: developed buildings backtest (fit 2021-23, score 2024), customer_since_year as is against minus 1.
ln, s = ctx.dev["buildings"], ctx.settings["buildings"]
bt_years, bt_year = rgl.WINDOWS["backtest"]
ln.use_window("backtest")
tr = ln.tune_trees(s["threshold"])
m = ln.build(bt_years, s["threshold"], s["alpha"], tr)
v = ln.rows([bt_year])
no_hist = {g: np.inf for g in s["ks"]}
mu_raw = ln.predict(m, "backtest", no_hist, v, year=bt_year)
mu_csy = ln.predict(m, "backtest", no_hist, v.assign(customer_since_year=v.customer_since_year - 1), year=bt_year)
ln.use_window("final")
y, e = v.y_full.to_numpy(float), v.exposure_plain.to_numpy(float)
dev = lambda mu: em.models.tweedie_deviance(y, mu, P)  # noqa: E731
null = em.models.tweedie_deviance(y, y.sum() / e.sum() * e, P)
codes, uniq = pd.factorize(v.customer_id.astype(str))
g = lambda x: np.bincount(codes, weights=x, minlength=len(uniq))  # noqa: E731
ga, gb, gn = g(dev(mu_raw)), g(dev(mu_csy)), g(null)
boot = np.array([((ga - gb) * w).sum() / (gn * w).sum() for w in
                 (np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(1000))])
res = {"rows": len(v), "trees": tr[0], "d2_raw": 1 - ga.sum() / gn.sum(), "d2_csy_minus_1": 1 - gb.sum() / gn.sum(),
       "gain": (ga - gb).sum() / gn.sum(), "ci_low_95": float(np.quantile(boot, 0.025)),
       "ci_high_95": float(np.quantile(boot, 0.975)), "p_one_sided": float((boot <= 0).mean()),
       "nb_share": float(v.new_business.astype(bool).mean()),
       "nb_mean_ratio_csy_over_raw": float(mu_csy[v.new_business.astype(bool).to_numpy()].mean()
                                           / mu_raw[v.new_business.astype(bool).to_numpy()].mean())}
(HERE / "r2_backtest.json").write_text(json.dumps(res, indent=1, default=float))
print("R4 developed backtest:", res, flush=True)
