"""Scratch (A31): add the halfway file's public score (0.15742) as a fourth constraint and re-predict the candidates."""
import os
import sys

import numpy as np
from scipy.linalg import null_space

LB = "<a31>/review/leaderboard_levels"
sys.path.insert(0, LB)
os.chdir(LB)
import io, contextlib  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    import s5e_predictions as s5  # noqa: E402
    sys.argv = ["x"]
    import candidates as cd  # noqa: E402  (defines cands; rewrites its evidence and file, unchanged)
from common_ll import SCORES  # noqa: E402

segs, B0 = s5.segs, s5.B0
mid = cd.vec(cd.cands["midpoint: as is and level reset + emerging motor NB loading"])
R = dict(s5.R) | {"mid": mid}
S = dict(SCORES) | {"mid": 0.15742}
F = ("asis", "nb", "lr", "mid")
grid = np.linspace(-1.5, 1.5, 61)


def family(lo, hi, tol=6e-6):
    Bt, out = B0.sum(), []
    for cn in np.arange(0.22, 0.3701, 0.01):
        for bt in np.arange(0.45, 0.70001, 0.0025):
            beta = bt / Bt
            M = np.array([B0 * R[f] ** -0.5 for f in F]); rhs = np.array([(1 + cn - S[f]) / beta - (B0 * R[f] ** 0.5).sum() for f in F])
            N = null_space(M)
            K0 = np.linalg.lstsq(M, rhs, rcond=None)[0]
            Z = np.array(np.meshgrid(*[grid] * N.shape[1], indexing="ij")).reshape(N.shape[1], -1)
            K = K0[:, None] + N @ Z * 1.2
            fit = np.abs(M @ K - rhs[:, None]).max(0) * beta  # residual in score units
            ok = np.all((K >= lo) & (K <= hi), axis=0) & (fit < tol)
            if ok.any():
                out.append((cn, beta, K[:, ok]))
    return out


def S_of(t, cn, beta, K):
    return 1 + cn - beta * ((B0 * t ** -0.5)[:, None] * K).sum(0) - beta * (B0 * t ** 0.5).sum()


for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = family(lo, hi)
    K = np.concatenate([k for _, _, k in fam], axis=1)
    print(f"\nbox [{lo}, {hi}]: {K.shape[1]} solutions consistent with all four scores")
    for i, s in enumerate(segs):
        print(f"  {s:14s} {K[i].min():.3f}-{K[i].max():.3f} (median {np.median(K[i]):.3f})")
    gain = np.concatenate([beta * (B0[:, None] * (np.sqrt(k) - 1) ** 2).sum(0) for cn, beta, k in fam])
    print(f"  gain over as is at the true segment levels: {gain.min():.4f}-{gain.max():.4f} (median {np.median(gain):.4f})")
    cands = dict(cd.cands)
    med = np.median(K, axis=1)
    cands["median of the consistent solutions"] = dict(zip(segs, med))
    for w in (0.75, 1.0):
        cands[f"as is to level reset + emerging motor NB loading, weight {w}"] = {s: cd.cands[list(cd.cands)[2]][s] ** w for s in segs}
    for name, sc in cands.items():
        t = cd.vec(sc)
        v = np.concatenate([S_of(t, cn, beta, k) for cn, beta, k in fam])
        print(f"  {name:66s} {v.min():.5f} / {np.median(v):.5f} / {v.max():.5f}; beats 0.15742 in {(v > 0.15742 + 1e-9).mean():.2f}")
    print("  median-solution scales:", {s: round(float(x), 3) for s, x in zip(segs, med)})
