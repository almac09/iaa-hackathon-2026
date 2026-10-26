"""Paired customer cluster bootstrap, matched convention: per-line levels (re-solved in each resample, or fixed),
pooled baseline = one ratio-of-sums rate over all three lines (re-solved in each resample, or fixed)."""
import json, sys
import numpy as np, pandas as pd
D = '<scratch>/a32/'
LINES = ['buildings', 'contents', 'motor']; SEEDS = list(range(2032, 2037))
MODELS = ['c0_total'] + [f'A2_draw{s}' for s in SEEDS] + [f'A3_w0.5_draw{s}' for s in SEEDS]; NM = len(MODELS)
fr = []
for li, ln in enumerate(LINES):
    a = pd.read_parquet(D + f'y24_{ln}.parquet', columns=['row_id', 'customer_id', 'exposure_plain', 'c0_total'])
    o = pd.read_parquet(D + f'y24out_{ln}.parquet', columns=['row_id', 'y_full'])
    pl = pd.read_parquet(D + f'pred_lgbm_{ln}.parquet', columns=['row_id'] + MODELS[1:6])
    pt = pd.read_parquet(D + f'pred_tabpfn_{ln}.parquet', columns=['row_id'] + MODELS[6:])
    f = a.merge(o, on='row_id', validate='1:1').merge(pl, on='row_id', validate='1:1').merge(pt, on='row_id', validate='1:1'); f['line'] = li; fr.append(f)
R = pd.concat(fr, ignore_index=True)
y = R.y_full.to_numpy(float); e = R.exposure_plain.to_numpy(float); line = R.line.to_numpy(); P = R[MODELS].to_numpy(float)
_, cid = np.unique(R.customer_id.to_numpy(), return_inverse=True); NC = cid.max() + 1
# full-sample levels for the fixed-level variant
kfull = np.zeros((3, NM))
for li in range(3):
    s = line == li; kfull[li] = (y[s, None] / np.sqrt(P[s])).sum(0) / np.sqrt(P[s]).sum(0)
rfull = y.sum() / e.sum()
# per line block: S3, A0, B0, Y, E, A_m (NM), B_m (NM), Dfix_m (NM)
cols = []
for li in range(3):
    s = (line == li).astype(float)[:, None]
    blk = np.column_stack([np.sqrt(y), y / np.sqrt(e), np.sqrt(e), y, e, y[:, None] / np.sqrt(P), np.sqrt(P),
                           4 * (y[:, None] / np.sqrt(P * kfull[li]) + np.sqrt(P * kfull[li]) - 2 * np.sqrt(y)[:, None])])
    cols.append(blk * s)
Z = np.hstack(cols); K = Z.shape[1] // 3
M = np.zeros((NC, Z.shape[1])); np.add.at(M, cid, Z)
def gains(S):
    S = np.atleast_2d(S); n = S.shape[0]
    Dm = np.zeros((n, NM)); Dfix = np.zeros((n, NM)); tot = np.zeros((n, 5))
    for li in range(3):
        s = S[:, li * K:(li + 1) * K]
        Dm += 8 * (np.sqrt(s[:, 5:5 + NM] * s[:, 5 + NM:5 + 2 * NM]) - s[:, [0]])
        Dfix += s[:, 5 + 2 * NM:]
        tot += s[:, :5]
    S3, A0, B0, Y, E = tot.T
    r = Y / E
    Db = 4 * (A0 / np.sqrt(r) + np.sqrt(r) * B0 - 2 * S3)
    Dbf = 4 * (A0 / np.sqrt(rfull) + np.sqrt(rfull) * B0 - 2 * S3)
    out = {}
    for nm, Dmod, Dbase in [('resolved', Dm, Db), ('fixed', Dfix, Dbf)]:
        d2 = 1 - Dmod / Dbase[:, None]
        out[nm] = np.stack([(d2[:, 1:6] - d2[:, [0]]).mean(1), (d2[:, 6:11] - d2[:, [0]]).mean(1)], 1)
    return out
pt_est = gains(M.sum(0))
res = {'point_resolved': pt_est['resolved'][0].tolist(), 'point_fixed': pt_est['fixed'][0].tolist(), 'n_customers': int(NC)}
B = 2000
for seed in [int(x) for x in sys.argv[1:]]:
    rng = np.random.default_rng(seed); bs = {'resolved': [], 'fixed': []}
    for c in range(0, B, 50):
        cnt = np.stack([np.bincount(rng.integers(0, NC, NC), minlength=NC) for _ in range(50)]).astype(float)
        g = gains(cnt @ M)
        for k in bs: bs[k].append(g[k])
    for k in bs:
        b = np.vstack(bs[k])
        res[f'seed{seed}_{k}'] = dict(p=(b <= 0).mean(0).tolist(), ci_A2=np.quantile(b[:, 0], [0.025, 0.975]).tolist(),
                                      ci_A3=np.quantile(b[:, 1], [0.025, 0.975]).tolist(), sd=b.std(0).tolist(), mean=b.mean(0).tolist(),
                                      p_A3_normal=float(__import__('math').erfc(pt_est[k][0, 1] / b[:, 1].std() / 2 ** 0.5) / 2) if False else None)
print(json.dumps(res, indent=1))
