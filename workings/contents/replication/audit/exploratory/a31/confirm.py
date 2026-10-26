"""Scratch (A31): score frozen variations once on the held-back 2024 half B (PREREGISTERED_RULE.md).

    python confirm.py v2_residual_gbm v5_buhlmann_straub ...
Each variation is fitted on emerging 2023 (and the developed market), as in exploration, and scored on B with the
harness metric; the gain over C0 is bootstrapped by customer and Holm-adjusted across the variations named."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as h  # noqa: E402

LINES = h.LINES


def load_b(line):
    return pd.read_parquet(HERE / "cache" / "holdout_B" / f"emg_{line}_B.parquet")


def module(key):
    spec = importlib.util.spec_from_file_location(key, HERE / "variations" / key / "variation.py")
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE / "variations" / key))
    spec.loader.exec_module(m)
    return m


def gain_on(score_rows, preds, n=1000, seed=2031):
    y = np.concatenate([score_rows[ln].y_full.to_numpy(float) for ln in LINES])
    e = np.concatenate([score_rows[ln].exposure_plain.to_numpy(float) for ln in LINES])
    cust = np.concatenate([score_rows[ln].customer_id.astype(str).to_numpy() for ln in LINES])
    d_alt, d_c0, per_line = [], [], {}
    for ln in LINES:
        yl, c0 = score_rows[ln].y_full.to_numpy(float), score_rows[ln].c0.to_numpy(float)
        mu = np.asarray(preds[ln], float)
        assert mu.shape == yl.shape and np.all(np.isfinite(mu)) and np.all(mu > 0)
        da, dc = h.unit_dev(yl, mu * h.opt_scale(yl, mu)), h.unit_dev(yl, c0 * h.opt_scale(yl, c0))
        el = score_rows[ln].exposure_plain.to_numpy(float)
        nl = h.unit_dev(yl, yl.sum() / el.sum() * el).sum()
        per_line[ln] = {"d2_lm": 1 - da.sum() / nl, "c0_d2_lm": 1 - dc.sum() / nl}
        d_alt.append(da), d_c0.append(dc)
    d_alt, d_c0 = np.concatenate(d_alt), np.concatenate(d_c0)
    null = h.unit_dev(y, y.sum() / e.sum() * e)
    codes, uniq = pd.factorize(cust)
    g = lambda v: np.bincount(codes, weights=v, minlength=len(uniq))  # noqa: E731
    ga, gc, gn = g(d_alt), g(d_c0), g(null)
    rng = np.random.default_rng(seed)
    gains = np.array([((gc - ga) * w).sum() / (gn * w).sum()
                      for w in (np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                                for _ in range(n))])
    return {"pooled_d2_lm": 1 - d_alt.sum() / null.sum(), "c0_pooled_d2_lm": 1 - d_c0.sum() / null.sum(),
            "gain": float((gc - ga).sum() / gn.sum()), "boot": gains, "per_line": per_line}


def main(keys):
    train = {ln: h.load(ln).query("split == 'fit2023'") for ln in LINES}
    score_rows = {ln: load_b(ln) for ln in LINES}
    dev = {ln: h.load_dev(ln) for ln in LINES}
    out = []
    for key in keys:
        preds = module(key).predict(train, score_rows, dev, 2024)
        r = gain_on(score_rows, preds)
        out.append({"key": key, **{k: v for k, v in r.items() if k != "boot"},
                    "p_one_sided": float((r["boot"] <= 0).mean())})
    m = len(out)
    for rank, row in enumerate(sorted(out, key=lambda r: r["p_one_sided"])):  # Holm, one-sided 0.025
        row["holm_threshold"] = 0.025 / (m - rank)
    passed = True
    for row in sorted(out, key=lambda r: r["p_one_sided"]):
        passed = passed and row["p_one_sided"] < row["holm_threshold"]
        row["passes_holm"] = passed
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
