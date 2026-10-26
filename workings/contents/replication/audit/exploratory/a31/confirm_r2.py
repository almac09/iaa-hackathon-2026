"""Scratch (A31 round 2): R1, R2 and R3 scored on A2024 (reported) and confirmed once on B2024, Holm over m = 3."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm as cf  # noqa: E402
import harness as h  # noqa: E402

LINES = h.LINES
r2 = {ln: pd.read_parquet(HERE / f"r2_{ln}.parquet").set_index("row_id") for ln in LINES}


def column(ln, rows, col):
    """A round-2 prediction column aligned to `rows`; lines without the column keep C0."""
    src = r2[ln][col] if col in r2[ln] else r2[ln].c0
    out = src.reindex(rows.row_id).to_numpy(float)
    assert np.isfinite(out).all()
    return out


halves = {"A2024": {ln: h.load(ln).query("split == 'A2024'") for ln in LINES},
          "B2024": {ln: cf.load_b(ln) for ln in LINES}}
train = {ln: h.load(ln).query("split == 'fit2023'") for ln in LINES}
r3 = cf.module("r3_slope")
results = []
for half, rows in halves.items():
    for ln in LINES:  # the cached C0 must be the C0 that round 2 recomputed
        assert np.allclose(column(ln, rows[ln], "c0"), rows[ln].c0.to_numpy(float), rtol=1e-9)
    variants = {"R1 multiple imputation": {ln: column(ln, rows[ln], "c0_mi") for ln in LINES},
                "R2 vehicle age minus 2": {ln: column(ln, rows[ln], "c0_vage") for ln in LINES},
                "R3 calibration power": r3.predict(train, rows, None, 2024)}
    for key, preds in variants.items():
        r = cf.gain_on(rows, preds)
        results.append({"half": half, "key": key, **{k: v for k, v in r.items() if k != "boot"},
                        "ci_low_95": float(np.quantile(r["boot"], 0.025)),
                        "ci_high_95": float(np.quantile(r["boot"], 0.975)),
                        "p_one_sided": float((r["boot"] <= 0).mean())})
b = sorted([r for r in results if r["half"] == "B2024"], key=lambda r: r["p_one_sided"])
passed = True
for rank, row in enumerate(b):  # Holm, one-sided 0.025, m = 3
    row["holm_threshold"] = 0.025 / (len(b) - rank)
    passed = passed and row["p_one_sided"] < row["holm_threshold"]
    row["passes_holm"] = passed
print(json.dumps(results, indent=1, default=float))
