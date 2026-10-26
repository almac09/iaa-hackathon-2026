"""Post hoc calibration slopes: p0 on held-out 2023 and on 2024; TabPFN's p on 2024; and how much of log(ratio) is real
for occurrence (logistic coefficient with logit p0 alongside) against for small cost (Tweedie GLM, offset log C0 small)."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, TweedieRegressor

from vhelp import LINES, S, SEEDS, scored

sc = scored()
tp = {n: pd.read_parquet(S / f"pred_tabpfn_{n}.parquet") for n in LINES}


def z(f):
    return np.c_[np.log(f.c0_total_2023 / f.exposure), np.log(f.exposure)]


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def slope(occ, lp):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), occ)
    return float(m.coef_[0, 0])


for n in LINES:
    d = sc[n]
    f23 = pd.read_parquet(S / f"y23_{n}.parquet", columns=["exposure", "c0_total_2023", "n_claims"])
    occ23 = (f23.n_claims.to_numpy() > 0).astype(int)
    draws = {int(k): v for k, v in json.loads((S / f"draws_{n}.json").read_text()).items()}
    occ = (d.n_claims.to_numpy() > 0).astype(int)
    small, large = d.c0_small.to_numpy(float), (d.c0_total - d.c0_small).to_numpy(float)
    ys = d.y_small.to_numpy(float)
    out = {"p0_slope_rest23": [], "p0_slope_24": [], "p_slope_24": [], "logratio_coef_occ": [], "logratio_coef_cost": []}
    for s in SEEDS:
        pos = np.asarray(draws[s])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(f23.iloc[pos]), occ23[pos])
        rest = np.setdiff1d(np.arange(len(f23)), pos)
        out["p0_slope_rest23"].append(slope(occ23[rest], logit(lr.predict_proba(z(f23.iloc[rest]))[:, 1])))
        p0 = lr.predict_proba(z(d))[:, 1]
        r = (tp[n][f"A3_w1.0_draw{s}"].to_numpy(float) - large) / small
        p = r * p0
        out["p0_slope_24"].append(slope(occ, logit(p0)))
        out["p_slope_24"].append(slope(occ, logit(p)))
        m = LogisticRegression(C=1e6, max_iter=1000).fit(np.c_[logit(p0), np.log(r)], occ)
        out["logratio_coef_occ"].append(float(m.coef_[0, 1]))
        # Tweedie GLM (log link) of small cost on log ratio with offset log C0 small: fit y/c0 with weights c0^(2-p)
        t = TweedieRegressor(power=1.5, link="log", alpha=0.0, max_iter=1000).fit(np.log(r).reshape(-1, 1), ys / small, sample_weight=small ** 0.5)
        out["logratio_coef_cost"].append(float(t.coef_[0]))
    print(n, {k: (round(float(np.mean(v)), 3), round(float(np.min(v)), 3), round(float(np.max(v)), 3)) for k, v in out.items()})
