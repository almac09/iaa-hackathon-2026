"""Second pass: explicit row-level evaluator with selectable level / baseline conventions."""
import json, sys
import numpy as np, pandas as pd
D = '<scratch>/a32/'
EV = '<pack>/audit/evidence/challengers/'
LINES = ['buildings', 'contents', 'motor']; SEEDS = list(range(2032, 2037))
frames = []
for li, ln in enumerate(LINES):
    y24 = pd.read_parquet(D + f'y24_{ln}.parquet', columns=['row_id', 'customer_id', 'exposure_plain', 'c0_total'])
    out = pd.read_parquet(D + f'y24out_{ln}.parquet')
    pl = pd.read_parquet(D + f'pred_lgbm_{ln}.parquet'); pt = pd.read_parquet(D + f'pred_tabpfn_{ln}.parquet')
    f = y24.merge(out, on='row_id', validate='1:1', suffixes=('', '_o')).merge(pl, on='row_id', validate='1:1').merge(pt, on='row_id', validate='1:1')
    f['line'] = li; frames.append(f)
R = pd.concat(frames, ignore_index=True)
ARMCOLS = ['c0_total', 'lgbm_source'] + [c for c in R.columns if c.startswith('A2_') or c.startswith('A3_')]
y = R.y_full.to_numpy(float); e = R.exposure_plain.to_numpy(float); line = R.line.to_numpy()
P = R[ARMCOLS].to_numpy(float); NM = len(ARMCOLS); col = {c: j for j, c in enumerate(ARMCOLS)}
dev = lambda yy, mu: 4 * (yy / np.sqrt(mu) + np.sqrt(mu) - 2 * np.sqrt(yy))

def levels(mask):
    k = np.zeros((3, NM))
    for li in range(3):
        s = mask & (line == li)
        k[li] = (y[s, None] / np.sqrt(P[s])).sum(0) / np.sqrt(P[s]).sum(0)
    return k
def rate(mask, kind):
    if kind == 'ratio': return y[mask].sum() / e[mask].sum()
    return (y[mask] / np.sqrt(e[mask])).sum() / np.sqrt(e[mask]).sum()

def evaluate(mask, lev_mask=None, base_mask=None, base='ratio', pooled='pooled', level=True):
    lev_mask = mask if lev_mask is None else lev_mask; base_mask = mask if base_mask is None else base_mask
    k = levels(lev_mask) if level else np.ones((3, NM))
    Dm = np.zeros((3, NM)); Db = np.zeros(3)
    for li in range(3):
        s = mask & (line == li)
        Dm[li] = dev(y[s, None], P[s] * k[li]).sum(0)
        r = rate(base_mask & (line == li), base)
        Db[li] = dev(y[s], r * e[s]).sum()
    if pooled == 'pooled':
        r = rate(base_mask, base); Dbp = dev(y[mask], r * e[mask]).sum()
    else:
        Dbp = Db.sum()
    return 1 - Dm.sum(0) / Dbp, 1 - Dm / Db[:, None]

def gain(d2, arm):
    if arm == 'A2': idx = [col[f'A2_draw{s}'] for s in SEEDS]
    else: idx = [col[f'A3_w0.5_draw{s}'] for s in SEEDS]
    return d2[..., idx] - d2[..., [col['c0_total']]]

allm = np.ones(len(y), bool)
