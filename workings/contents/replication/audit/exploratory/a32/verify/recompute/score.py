"""Independent recomputation of A32 Stage 1 (does not import challengers.py).
Tweedie p=1.5 unit deviance d(y,mu) = 4*(y/sqrt(mu) + sqrt(mu) - 2*sqrt(y)).
For mu = k*m the deviance-optimal k = sum(y/sqrt(m)) / sum(sqrt(m)) and the minimised
deviance is 8*(sqrt(A*B) - S3), A = sum(y/sqrt(m)), B = sum(sqrt(m)), S3 = sum(sqrt(y))."""
import json, sys
import numpy as np, pandas as pd

D = '<scratch>/a32/'
LINES = ['buildings', 'contents', 'motor']
SEEDS = list(range(2032, 2037))
MODELS = ['c0_total'] + [f'A2_draw{s}' for s in SEEDS] + [f'A3_w0.5_draw{s}' for s in SEEDS] + ['lgbm_source']
NM = len(MODELS)

frames = []
for li, line in enumerate(LINES):
    y24 = pd.read_parquet(D + f'y24_{line}.parquet', columns=['row_id', 'customer_id', 'exposure_plain', 'c0_total'])
    out = pd.read_parquet(D + f'y24out_{line}.parquet')
    pl = pd.read_parquet(D + f'pred_lgbm_{line}.parquet')
    pt = pd.read_parquet(D + f'pred_tabpfn_{line}.parquet')
    # alignment by key, not position
    f = y24.merge(out, on='row_id', how='inner', validate='1:1', suffixes=('', '_o'))
    f = f.merge(pl, on='row_id', how='inner', validate='1:1').merge(pt, on='row_id', how='inner', validate='1:1')
    assert len(f) == 100000 and (f.customer_id == f.customer_id_o).all()
    f['line'] = li
    frames.append(f[['row_id', 'customer_id', 'line', 'y_full', 'exposure_plain', 'new_business', 'age_band'] + MODELS])
R = pd.concat(frames, ignore_index=True)
y = R.y_full.to_numpy(float); e = R.exposure_plain.to_numpy(float); line = R.line.to_numpy()
P = R[MODELS].to_numpy(float)
assert (P > 0).all() and (e > 0).all() and (y >= 0).all()
cust_codes, cust_idx = np.unique(R.customer_id.to_numpy(), return_inverse=True)
NC = len(cust_codes)

# row-level columns per line: [S3, A0, B0, Y, E, A_m..., B_m..., Araw...]
def rowcols(mask=None):
    sy = np.sqrt(y)
    cols = [sy, y / np.sqrt(e), np.sqrt(e), y, e]
    cols += [y / np.sqrt(P[:, j]) for j in range(NM)]
    cols += [np.sqrt(P[:, j]) for j in range(NM)]
    X = np.column_stack(cols)  # n x K
    if mask is not None:
        X = X * mask[:, None]
    K = X.shape[1]
    Z = np.zeros((len(y), 3 * K))
    for li in range(3):
        sel = line == li
        Z[sel, li * K:(li + 1) * K] = X[sel]
    return Z, K

def metrics(S, K, baseline='opt', pooled_base='per_line'):
    """S: (..., 3K) sums. Returns dict of arrays: pooled lm D2 per model, per-line lm D2, raw pooled D2."""
    S = np.atleast_2d(S)
    Dm = np.zeros((S.shape[0], 3, NM)); Draw = np.zeros((S.shape[0], 3, NM)); Db = np.zeros((S.shape[0], 3))
    for li in range(3):
        s = S[:, li * K:(li + 1) * K]
        S3, A0, B0, Y, E = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
        A = s[:, 5:5 + NM]; B = s[:, 5 + NM:5 + 2 * NM]
        Dm[:, li] = 8 * (np.sqrt(A * B) - S3[:, None])
        Draw[:, li] = 4 * (A + B - 2 * S3[:, None])
        if baseline == 'opt':
            Db[:, li] = 8 * (np.sqrt(A0 * B0) - S3)
        else:  # ratio of sums
            r = Y / E
            Db[:, li] = 4 * (A0 / np.sqrt(r) + np.sqrt(r) * B0 - 2 * S3)
    if pooled_base == 'per_line':
        Dbp = Db.sum(1)
    else:  # one constant rate across all three lines
        s = sum(S[:, li * K:(li + 1) * K] for li in range(3))
        S3, A0, B0, Y, E = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
        Dbp = 8 * (np.sqrt(A0 * B0) - S3) if baseline == 'opt' else 4 * (A0 / np.sqrt(Y / E) + np.sqrt(Y / E) * B0 - 2 * S3)
    pooled = 1 - Dm.sum(1) / Dbp[:, None]
    raw = 1 - Draw.sum(1) / Dbp[:, None]
    perline = 1 - Dm / Db[:, :, None]
    return dict(pooled=pooled, raw=raw, perline=perline)

ia2 = [MODELS.index(f'A2_draw{s}') for s in SEEDS]
ia3 = [MODELS.index(f'A3_w0.5_draw{s}') for s in SEEDS]
def gains(m, key='pooled'):
    x = m[key]
    g2 = x[..., ia2] - x[..., [0]] if key != 'perline' else x[..., ia2] - x[..., [0]]
    g3 = x[..., ia3] - x[..., [0]]
    return g2, g3

res = {}
Z, K = rowcols()
Sfull = Z.sum(0)
for bl in ['opt', 'ratio']:
    for pb in ['per_line', 'pooled']:
        m = metrics(Sfull, K, bl, pb)
        res[f'C0_pooled_lm[{bl},{pb}]'] = float(m['pooled'][0, 0])
        g2, g3 = gains(m)
        res[f'A2_gain[{bl},{pb}]'] = float(g2.mean()); res[f'A3_gain[{bl},{pb}]'] = float(g3.mean())
m = metrics(Sfull, K)
res['C0_raw'] = float(m['raw'][0, 0])
res['A2_raw_mean'] = float(m['raw'][0, ia2].mean()); res['A3_raw_mean'] = float(m['raw'][0, ia3].mean())
res['lgbm_source_pooled'] = float(m['pooled'][0, -1])
res['C0_perline'] = m['perline'][0, :, 0].tolist()
g2, g3 = gains(m)
res['A2_draw_gains'] = g2[0].tolist(); res['A3_draw_gains'] = g3[0].tolist()
pl2 = m['perline'][0][:, ia2] - m['perline'][0][:, [0]]; pl3 = m['perline'][0][:, ia3] - m['perline'][0][:, [0]]
res['A2_perline_gain'] = pl2.mean(1).tolist(); res['A3_perline_gain'] = pl3.mean(1).tolist()
G = np.array([g2.mean(), g3.mean()])

# gate: ten largest scored claims removed (pooled), levels re-solved
top = np.argsort(-y, kind='stable')[:10]
res['top10_y'] = sorted(y[top].round(0).tolist(), reverse=True)
res['top10_lines'] = np.bincount(line[top], minlength=3).tolist()
keep = np.ones(len(y)); keep[top] = 0
Zk, _ = rowcols(keep); mk = metrics(Zk.sum(0), K); a, b = gains(mk)
res['A2_gain_no_top10'] = float(a.mean()); res['A3_gain_no_top10'] = float(b.mean())
# variant: ten largest per line
keep2 = np.ones(len(y))
for li in range(3):
    idx = np.where(line == li)[0]; keep2[idx[np.argsort(-y[idx])[:10]]] = 0
Zk2, _ = rowcols(keep2); mk2 = metrics(Zk2.sum(0), K); a, b = gains(mk2)
res['A2_gain_no_top10_per_line'] = float(a.mean()); res['A3_gain_no_top10_per_line'] = float(b.mean())

# subsets: new business, 17-24; (i) levels re-solved on the subset (ii) levels fixed at full-sample levels
def fixed_level_subset(mask):
    # full-sample levels per line and model; baseline rate full-sample optimal
    out = {}
    Dm = np.zeros((3, NM)); Db = np.zeros(3)
    for li in range(3):
        sl = Sfull[li * K:(li + 1) * K]
        k = sl[5:5 + NM] / sl[5 + NM:5 + 2 * NM]; r = sl[1] / sl[2]
        sel = (line == li) & mask
        yy, ee, PP = y[sel], e[sel], P[sel]
        mu = PP * k[None, :]
        Dm[li] = (4 * (yy[:, None] / np.sqrt(mu) + np.sqrt(mu) - 2 * np.sqrt(yy)[:, None])).sum(0)
        mu0 = r * ee
        Db[li] = (4 * (yy / np.sqrt(mu0) + np.sqrt(mu0) - 2 * np.sqrt(yy))).sum()
    d2 = 1 - Dm.sum(0) / Db.sum()
    return float((d2[ia2] - d2[0]).mean()), float((d2[ia3] - d2[0]).mean())
nb = R.new_business.astype(str).str.lower().isin(['true', '1']).to_numpy()
ag = (R.age_band.astype(str) == '17-24').to_numpy()
res['n_nb'] = int(nb.sum()); res['n_1724'] = int(ag.sum())
for name, msk in [('nb', nb), ('age1724', ag)]:
    Zs, _ = rowcols(msk.astype(float)); ms = metrics(Zs.sum(0), K); a, b = gains(ms)
    res[f'A2_gain_{name}_resolved'] = float(a.mean()); res[f'A3_gain_{name}_resolved'] = float(b.mean())
    res[f'A2_gain_{name}_fixedlevel'], res[f'A3_gain_{name}_fixedlevel'] = fixed_level_subset(msk)

# paired customer cluster bootstrap
Mc = np.zeros((NC, Z.shape[1]))
np.add.at(Mc, cust_idx, Z)
assert np.allclose(Mc.sum(0), Sfull)
B = 2000
boot = np.zeros((B, 2))
for seed_label in [int(sys.argv[1]) if len(sys.argv) > 1 else 20261026]:
    rng = np.random.default_rng(seed_label)
    chunk = 50
    for c0 in range(0, B, chunk):
        cnt = np.stack([np.bincount(rng.integers(0, NC, NC), minlength=NC) for _ in range(chunk)]).astype(float)
        Sb = cnt @ Mc
        mb = metrics(Sb, K); a, b = gains(mb)
        boot[c0:c0 + chunk, 0] = a.mean(1); boot[c0:c0 + chunk, 1] = b.mean(1)
p = (boot <= 0).mean(0)
p_alt = ((boot <= 0).sum(0) + 1) / (B + 1)
ci = np.quantile(boot, [0.025, 0.975], axis=0)
res['boot_seed'] = seed_label
res['p_one_sided'] = p.tolist(); res['p_plus1'] = p_alt.tolist()
res['ci95_A2'] = ci[:, 0].tolist(); res['ci95_A3'] = ci[:, 1].tolist()
res['boot_sd'] = boot.std(0).tolist(); res['boot_mean'] = boot.mean(0).tolist()
# Holm, m=2, one-sided alpha 0.025
order = np.argsort(p); alpha = 0.025; passes = [False, False]
for rank, j in enumerate(order):
    if p[j] < alpha / (2 - rank):
        passes[j] = True
    else:
        break
res['holm_thresholds'] = {['A2', 'A3'][j]: alpha / (2 - rank) for rank, j in enumerate(order)}
res['holm_pass'] = passes
for j, arm in enumerate(['A2', 'A3']):
    res[f'{arm}_gate'] = dict(holm=passes[j], gain_ge_0p003=bool(G[j] >= 0.003), no_top10_pos=res[f'{arm}_gain_no_top10'] > 0,
                             nb_noloss=res[f'{arm}_gain_nb_resolved'] >= 0, age_noloss=res[f'{arm}_gain_age1724_resolved'] >= 0)
res['n_customers'] = int(NC)
print(json.dumps(res, indent=1))
