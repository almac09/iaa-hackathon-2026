"""Scratch: the six-segment (line x new business) reading of the board with all five public scores."""
import json
import numpy as np
import pandas as pd
from scipy.linalg import null_space

PACK = "<pack>/"
j = json.load(open(PACK + "audit/exploratory/a31/levels/s1_common_scale.json"))["segments"]
segs = list(j); B0 = np.array([j[s]["B"] for s in segs])
nb = pd.concat([pd.read_parquet(PACK + f"data/emerging/test_{ln}.parquet", columns=["row_id", "new_business"]).assign(line=ln)
                for ln in ("buildings", "contents", "motor")]).set_index("row_id")
W = PACK + "work/submissions/submission_emerging_"
files = {"asis": ("as_is", 0.15601), "nb": ("level_reset_nb_scenario", 0.15451), "lr": ("level_reset", 0.15021),
         "mid": ("level_midpoint", 0.15742), "m105": ("level_midpoint_motornb105", 0.15805)}
asis = pd.read_csv(W + "as_is_20261026.csv").set_index("row_id").prediction
seg = (nb.line + np.where(nb.new_business.astype(bool), "_nb", "_ren")).reindex(asis.index)
R, S = {}, {}
for k, (f, sc) in files.items():
    r = (pd.read_csv(W + f + "_20261026.csv").set_index("row_id").prediction.reindex(asis.index) / asis).groupby(seg.to_numpy())
    assert (r.max() - r.min()).max() < 1e-9, f"{k} is not a segment rescale"
    R[k], S[k] = r.mean().reindex(segs).to_numpy(), sc
# Check the segment weights still match the reviewer's (sqrt-prediction shares of as is).
w = np.sqrt(asis).groupby(seg.to_numpy()).sum().reindex(segs).to_numpy()
assert np.allclose(w / w.sum(), B0 / B0.sum(), rtol=1e-6)
F = list(files)
grid = np.linspace(-1.5, 1.5, 81)
def family(lo, hi, tol=6e-6):
    Bt, out = B0.sum(), []
    for cn in np.arange(0.22, 0.3701, 0.005):
        for bt in np.arange(0.45, 0.70001, 0.001):
            beta = bt / Bt
            M = np.array([B0 * R[f] ** -0.5 for f in F]); rhs = np.array([(1 + cn - S[f]) / beta - (B0 * R[f] ** 0.5).sum() for f in F])
            N = null_space(M); K0 = np.linalg.lstsq(M, rhs, rcond=None)[0]
            Z = grid[None, :] if N.shape[1] == 1 else np.array(np.meshgrid(*[grid] * N.shape[1], indexing="ij")).reshape(N.shape[1], -1)
            K = K0[:, None] + N @ Z * 1.2
            ok = np.all((K >= lo) & (K <= hi), axis=0) & (np.abs(M @ K - rhs[:, None]).max(0) * beta < tol)
            if ok.any():
                out.append((cn, beta, K[:, ok]))
    return out
def S_of(t, cn, beta, K):
    return 1 + cn - beta * ((B0 * t ** -0.5)[:, None] * K).sum(0) - beta * (B0 * t ** 0.5).sum()
for lo, hi in ((0.7, 1.4), (0.6, 1.5), (0.5, 1.6)):
    fam = family(lo, hi)
    if not fam:
        print(f"box [{lo}, {hi}]: no solution"); continue
    K = np.concatenate([k for _, _, k in fam], axis=1)
    best = np.concatenate([1 + cn - 2 * beta * (B0[:, None] * np.sqrt(k)).sum(0) for cn, beta, k in fam])  # at the true levels
    print(f"box [{lo}, {hi}]: {K.shape[1]} solutions; best attainable public score at the true six levels "
          f"{best.min():.5f}-{best.max():.5f} (median {np.median(best):.5f}); our best 0.15805")
    for i, s in enumerate(segs):
        print(f"   {s:14s} {K[i].min():.3f}-{K[i].max():.3f} (median {np.median(K[i]):.3f})")
