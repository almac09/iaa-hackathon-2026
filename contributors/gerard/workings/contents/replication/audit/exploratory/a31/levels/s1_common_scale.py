"""(1) Exact p = 1.5 identity for segment rescales, and the one-common-scale fit to the three public scores."""
import itertools, json
import numpy as np, pandas as pd
from common_ll import *

t = test_rows()
assert t[["asis", "nb", "lr"]].notna().all().all() and len(t) == 209063
print("exposure vs exposure_plain max abs diff:", float((t.exposure - t.exposure_plain).abs().max()))
segs = sorted(t.seg.unique())
B = t.groupby("seg").asis.apply(lambda v: np.sqrt(v).sum())
R = {}
for f in ("asis", "nb", "lr"):
    r = (t[f] / t.asis).groupby(t.seg)
    assert (r.max() - r.min()).max() < 1e-9, f
    R[f] = r.mean()
print(pd.DataFrame({"B_sqrt_sum": B, "r_nb": R["nb"], "r_lr": R["lr"], "rows": t.seg.value_counts()}).to_string())

def solve_common(S):
    # S_j = alpha - beta * sum_s B_s (k r^-1/2 + r^1/2); differences vs as-is are linear in k after the ratio
    P = {f: float((B * (R[f] ** -0.5 - 1)).sum()) for f in ("nb", "lr")}
    Q = {f: float((B * (R[f] ** 0.5 - 1)).sum()) for f in ("nb", "lr")}
    d2, d3 = S["asis"] - S["nb"], S["asis"] - S["lr"]
    rho = d2 / d3
    k = (Q["nb"] - rho * Q["lr"]) / (rho * P["lr"] - P["nb"])
    beta = d3 / (k * P["lr"] + Q["lr"])
    Btot = float(B.sum())
    alpha = S["asis"] + beta * Btot * (k + 1)
    gain = beta * Btot * (np.sqrt(k) - 1) ** 2
    return dict(k=k, beta=beta, b_tot=beta * Btot, alpha=alpha, C_over_N=alpha - 1, gain=gain, best=S["asis"] + gain)

base = solve_common(SCORES)
print("common scale, scores as shown:", {a: round(v, 6) for a, v in base.items()})
# rounding box: Kaggle shows 5 dp
res = []
for d in itertools.product([-5e-6, 5e-6], repeat=3):
    S = {f: SCORES[f] + e for f, e in zip(("asis", "nb", "lr"), d)}
    res.append(solve_common(S))
res = pd.DataFrame(res)
print("rounding box range:\n", res.agg(["min", "max"]).round(6).to_string())

# the audit's quadratic in log of the file's total ratio
tot = {f: float(t[f].sum() / t.asis.sum()) for f in ("asis", "nb", "lr")}
x = np.log([tot[f] for f in ("asis", "nb", "lr")]); y = np.array([SCORES[f] for f in ("asis", "nb", "lr")])
c2, c1, c0 = np.polyfit(x, y, 2)
xs = -c1 / (2 * c2)
print("totals", {k: round(v, 4) for k, v in tot.items()}, "quadratic in log total: optimum x%.4f, score %.5f, gain %+.5f, curvature c2 %.3f" % (np.exp(xs), c0 - c1 ** 2 / (4 * c2), c0 - c1 ** 2 / (4 * c2) - SCORES["asis"], c2))
# same exact-deviance model but treating each file as a pure common rescale at its total ratio (a shortcut)
def solve_total(S):
    r = np.array([tot["nb"], tot["lr"]])
    Pj, Qj = r ** -0.5 - 1, r ** 0.5 - 1
    rho = (S["asis"] - S["nb"]) / (S["asis"] - S["lr"])
    k = (Qj[0] - rho * Qj[1]) / (rho * Pj[1] - Pj[0])
    bt = (S["asis"] - S["lr"]) / (k * Pj[1] + Qj[1])
    return k, bt, bt * (np.sqrt(k) - 1) ** 2
print("shortcut (files as pure common rescales at their totals): k %.4f, b_tot %.4f, gain %+.5f" % solve_total(SCORES))
json.dump({"segments": {s: {"B": float(B[s]), "r_nb": float(R["nb"][s]), "r_lr": float(R["lr"][s])} for s in segs},
           "common": base, "rounding": res.agg(["min", "max"]).to_dict()}, open("s1_common_scale.json", "w"), indent=1)
