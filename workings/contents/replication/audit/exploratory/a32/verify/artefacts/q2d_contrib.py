"""Post hoc: where the A3 (w 0.5) gain is won and lost, by exposure band; and a cross-fitted cost recalibration of the
ratio (the best a monotone use of the saved ratio could do, as an optimistic bound)."""
import numpy as np
import pandas as pd

from vhelp import LINES, S, SEEDS, Scorer, scored

sc = scored()
scr = Scorer(sc)
tp = {n: pd.read_parquet(S / f"pred_tabpfn_{n}.parquet") for n in LINES}
E = scr.E
BANDS = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.99, 1.01]
band = pd.cut(E, BANDS, labels=False).to_numpy() if hasattr(pd.cut(E, BANDS, labels=False), "to_numpy") else pd.cut(E, BANDS, labels=False)
cols = [np.concatenate([tp[n][f"A3_w0.5_draw{s}"].to_numpy(float) for n in LINES]) for s in SEEDS]
devs = [scr.lm_dev(c) for c in cols]
tot = scr.null.sum()
print("A3 w0.5 contribution to the pooled gain by exposure band", BANDS[1:])
for i, n in enumerate(LINES):
    k = scr.line == i
    contrib = [np.mean([(scr.dev_c0[k & (band == b)].sum() - d[k & (band == b)].sum()) / tot for d in devs]) for b in range(len(BANDS) - 1)]
    rows = [int((k & (band == b)).sum()) for b in range(len(BANDS) - 1)]
    print(f"  {n:9s} contrib {[round(x, 5) for x in contrib]} sum {sum(contrib):+.5f}; rows {rows}")
# the ratio in the tiniest bands
large = np.concatenate([(sc[n].c0_total - sc[n].c0_small).to_numpy(float) for n in LINES])
small = np.concatenate([sc[n].c0_small.to_numpy(float) for n in LINES])
r1 = [(np.concatenate([tp[n][f"A3_w1.0_draw{s}"].to_numpy(float) for n in LINES]) - large) / small for s in SEEDS]
for i, n in enumerate(LINES):
    k = scr.line == i
    med = [float(np.mean([np.median(r[k & (band == b)]) for r in r1])) for b in range(len(BANDS) - 1)]
    print(f"  {n:9s} median ratio by band {[round(x, 2) for x in med]}")

# variant: ratio neutralised (set to 1) for rows with exposure < 0.1 -- shows how much of the loss the tiny rows carry
for cut in (0.05, 0.1, 0.25):
    g = []
    for s, r in zip(SEEDS, r1):
        rr = np.where(E < cut, 1.0, r)
        g.append(scr.gain(small * rr ** 0.5 + large))
    print(f"  ratio set to 1 below exposure {cut}: pooled gain w0.5 {np.mean(g):+.5f}")

# cross-fitted cost recalibration: on one half of customers, A/E of small cost against C0 small by decile of log ratio
# (deciles cut on that half), applied as a multiplier on the other half; per line and draw.
rng = np.random.default_rng(11)
half = rng.integers(0, 2, scr.ncust)[scr.codes].astype(bool)
ys = np.concatenate([sc[n].y_small.to_numpy(float) for n in LINES])
gains = []
for r in r1:
    mult = np.ones(len(ys))
    for i in range(3):
        for A, B in ((half, ~half), (~half, half)):
            ka, kb = (scr.line == i) & A, (scr.line == i) & B
            edges = np.quantile(np.log(r[ka]), np.linspace(0, 1, 11)[1:-1])
            qa, qb = np.searchsorted(edges, np.log(r[ka])), np.searchsorted(edges, np.log(r[kb]))
            lvl = ys[ka].sum() / small[ka].sum()
            ae = np.array([ys[ka][qa == j].sum() / small[ka][qa == j].sum() / lvl for j in range(10)])
            mult[kb] = ae[qb]
    gains.append(scr.gain(small * mult + large))
print(f"cross-fitted decile cost recalibration of the ratio: pooled gain {np.mean(gains):+.5f} (draws {[round(x, 5) for x in gains]})")
