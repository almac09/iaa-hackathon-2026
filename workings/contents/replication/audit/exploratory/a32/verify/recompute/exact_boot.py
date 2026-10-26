"""Reproduce the reported bootstrap draw-for-draw: customer codes in first-appearance order (buildings, contents, motor),
default_rng(2032), integers(0, n, n) per resample; levels and null rate fixed at full-sample values."""
import json
import numpy as np, pandas as pd
D = '<scratch>/a32/'
LINES = ['buildings', 'contents', 'motor']; SEEDS = range(2032, 2037)
dev = lambda y, mu: 4 * (y / np.sqrt(mu) + np.sqrt(mu) - 2 * np.sqrt(y))
Y, E, C, L, preds = [], [], [], [], {}
for li, ln in enumerate(LINES):
    a = pd.read_parquet(D + f'y24_{ln}.parquet', columns=['row_id', 'customer_id', 'exposure_plain', 'c0_total'])
    o = pd.read_parquet(D + f'y24out_{ln}.parquet', columns=['row_id', 'customer_id', 'y_full'])
    pl = pd.read_parquet(D + f'pred_lgbm_{ln}.parquet'); pt = pd.read_parquet(D + f'pred_tabpfn_{ln}.parquet')
    meta = json.load(open(D + 'lgbm_meta.json'))[ln]['decay_chosen']
    chk = max(np.abs(pl[f'A2_draw{s}'] - pl[f'A2_d{meta[str(s)]}_draw{s}']).max() for s in SEEDS)
    print(ln, 'A2_draw == chosen-decay column, max abs diff', chk)
    Y.append(o.y_full.to_numpy(float)); E.append(a.exposure_plain.to_numpy(float)); C.append(o.customer_id.astype(str).to_numpy()); L.append(np.full(len(a), li))
    for c in ['c0_total'] + [f'A2_draw{s}' for s in SEEDS] + [f'A3_w0.5_draw{s}' for s in SEEDS]:
        preds.setdefault(c, []).append((a if c == 'c0_total' else (pl if c.startswith('A2') else pt))[c].to_numpy(float))
Y, E, C, L = map(np.concatenate, (Y, E, C, L)); preds = {k: np.concatenate(v) for k, v in preds.items()}
null = dev(Y, Y.sum() / E.sum() * E)
def lmdev(m):
    d = np.zeros(len(Y))
    for li in range(3):
        s = L == li; k = (Y[s] / np.sqrt(m[s])).sum() / np.sqrt(m[s]).sum(); d[s] = dev(Y[s], k * m[s])
    return d
dv = {k: lmdev(v) for k, v in preds.items()}
codes, uniq = pd.factorize(C)
n = len(uniq); rng = np.random.default_rng(2032)
gn = np.bincount(codes, weights=null, minlength=n); gc = np.bincount(codes, weights=dv['c0_total'], minlength=n)
fam = {'A2': [f'A2_draw{s}' for s in SEEDS], 'A3': [f'A3_w0.5_draw{s}' for s in SEEDS]}
gd = {f: np.mean([np.bincount(codes, weights=dv[c], minlength=n) for c in cs], axis=0) for f, cs in fam.items()}
num = {f: np.zeros(2000) for f in fam}; den = np.zeros(2000)
for b in range(2000):
    w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    den[b] = gn @ w
    for f in fam: num[f][b] = (gc - gd[f]) @ w
for f in fam:
    bt = num[f] / den
    print(f, 'p', (bt <= 0).mean(), 'ci', np.quantile(bt, [0.025, 0.975]).tolist(), 'sd', bt.std())
