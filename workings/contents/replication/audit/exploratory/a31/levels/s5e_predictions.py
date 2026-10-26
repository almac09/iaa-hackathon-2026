"""(2f/3c) Across all solutions consistent with the three scores (C/N 0.22-0.37, box): predicted public score of
the audit's x0.912 file, of probes x0.80 / x1.25, the best common scale and its gain; and robustness of the motor
NB range to +-1.5% errors in the public segment weights and +-0.000005 rounding."""
import json
import numpy as np
from scipy.linalg import null_space
from common_ll import *
j = json.load(open("s1_common_scale.json"))["segments"]
segs = list(j); B0 = np.array([j[s]["B"] for s in segs])
R = {"asis": np.ones(6), "nb": np.array([j[s]["r_nb"] for s in segs]), "lr": np.array([j[s]["r_lr"] for s in segs])}
F = ("asis", "nb", "lr")
grid = np.linspace(-1.5, 1.5, 41)
Z = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1)
def family(lo, hi, B, S):
    Bt = B.sum(); out = []
    for cn in np.arange(0.22, 0.3701, 0.03):
        for bt in np.arange(0.45, 0.70001, 0.005):
            beta = bt / Bt
            M = np.array([B * R[f] ** -0.5 for f in F]); rhs = np.array([(1 + cn - S[f]) / beta - (B * R[f] ** 0.5).sum() for f in F])
            K = np.linalg.lstsq(M, rhs, rcond=None)[0][:, None] + null_space(M) @ Z * 1.2
            ok = np.all((K >= lo) & (K <= hi), axis=0)
            if ok.any():
                out.append((cn, beta, K[:, ok]))
    return out
def S_of(t, cn, beta, K, B):
    return 1 + cn - beta * ((B * t ** -0.5)[:, None] * K).sum(0) - beta * (B * t ** 0.5).sum()
for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = family(lo, hi, B0, SCORES)
    res = {k: [] for k in ("x0.912", "x0.80", "x1.25", "kbar", "gain_common", "gain_seg")}
    for cn, beta, K in fam:
        for lab, c in (("x0.912", 0.912), ("x0.80", 0.80), ("x1.25", 1.25)):
            res[lab].append(S_of(np.full(6, c), cn, beta, K, B0))
        kbar = (B0[:, None] * K).sum(0) / B0.sum()
        res["kbar"].append(kbar)
        res["gain_common"].append(beta * B0.sum() * (np.sqrt(kbar) - 1) ** 2)
        res["gain_seg"].append(beta * (B0[:, None] * (np.sqrt(K) - 1) ** 2).sum(0))
    res = {k: np.concatenate(v) for k, v in res.items()}
    print(f"box [{lo}, {hi}] ({len(res['kbar'])} solutions):")
    for k, v in res.items():
        print(f"   {k}: min {v.min():.5f}  median {np.median(v):.5f}  max {v.max():.5f}")
    print(f"   share of solutions where the x0.912 file beats as-is (0.15601): {(res['x0.912'] > SCORES['asis']).mean():.3f}")
# robustness of motor NB / motor ratio ranges, tight box
rng = np.random.default_rng(4)
mnb, mratio = [], []
for rep in range(20):
    B = B0 * (1 + rng.uniform(-0.015, 0.015, 6))
    S = {f: v + rng.uniform(-5e-6, 5e-6) for f, v in SCORES.items()}
    K = np.concatenate([k for _, _, k in family(0.7, 1.4, B, S)], axis=1)
    mnb.append([K[segs.index("motor_nb")].min(), K[segs.index("motor_nb")].max()])
    q = K[segs.index("motor_nb")] / K[segs.index("motor_ren")]
    mratio.append([q.min(), q.max()])
mnb, mratio = np.array(mnb), np.array(mratio)
print(f"perturbed (20 draws), box [0.7, 1.4]: motor NB scale range lowest min {mnb[:, 0].min():.3f}, highest max {mnb[:, 1].max():.3f}; "
      f"motor NB/ren ratio lowest min {mratio[:, 0].min():.2f}, highest max {mratio[:, 1].max():.2f}")
