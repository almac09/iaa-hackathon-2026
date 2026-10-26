"""Scratch (A31 round 2, R3): per-line calibration power on C0, fitted on emerging 2023 (PREREGISTERED_RULE.md)."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the a31 folder, which holds harness.py
import harness as h  # noqa: E402

CAP = {"buildings": 50_000.0, "contents": None, "motor": 30_000.0}


def capped(y, c0, cap):
    """Row target capped, the excess spread back pro rata to C0."""
    if cap is None:
        return y
    yc = np.minimum(y, cap)
    return yc + (y - yc).sum() * c0 / c0.sum()


def fit_power(y, c0):
    x = np.log(c0)
    f = lambda t: h.unit_dev(y, np.exp(t[0] + t[1] * x)).sum()  # noqa: E731
    return minimize(f, x0=[np.log(y.sum() / c0.sum()), 1.0], method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}).x[1]


def predict(train, score, dev, year):
    out = {}
    for ln in h.LINES:
        t = train[ln]
        b = fit_power(capped(t.y_full.to_numpy(float), t.c0.to_numpy(float), CAP[ln]), t.c0.to_numpy(float))
        out[ln] = score[ln].c0.to_numpy(float) ** b
        print(f"  {ln}: power {b:.3f}", flush=True)
    return out


if __name__ == "__main__":
    train = {ln: h.load(ln).query("split == 'fit2023'") for ln in h.LINES}
    score = {ln: h.load(ln).query("split == 'A2024'") for ln in h.LINES}
    print(h.report(predict(train, score, None, 2024), "R3 calibration power (A2024)"))
