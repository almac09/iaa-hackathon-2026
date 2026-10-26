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
out = {}
# arms table comparison
ev = pd.read_csv(EV + 'ev_a32_arms.csv').set_index('arm')
d2p, d2l = evaluate(allm)
raw, _ = evaluate(allm, level=False)
cmp = pd.DataFrame({'pooled_lm': d2p, 'raw': raw, 'b': d2l[0], 'c': d2l[1], 'm': d2l[2]}, index=ARMCOLS)
ev = ev.rename(index={'C0': 'c0_total'})
diff = (cmp - ev[['pooled_d2_level_matched', 'pooled_d2_raw', 'buildings_d2_level_matched', 'contents_d2_level_matched', 'motor_d2_level_matched']].reindex(cmp.index).to_numpy())
out['arms_table_max_abs_diff'] = diff.abs().max().to_dict()
out['arms_missing_in_ev'] = [c for c in ARMCOLS if c not in ev.index]
out['C0'] = dict(pooled_lm=d2p[col['c0_total']], raw=raw[col['c0_total']], perline=d2l[:, col['c0_total']].tolist())
for arm in ['A2', 'A3']:
    g = gain(d2p, arm); out[f'{arm}_gain'] = g.mean(); out[f'{arm}_draws'] = g.tolist()
    out[f'{arm}_spread'] = [g.min(), g.max(), g.std(ddof=1)]
    out[f'{arm}_perline'] = gain(d2l, arm).mean(1).tolist()
    out[f'{arm}_raw_mean'] = raw[[col[f'A2_draw{s}'] if arm == 'A2' else col[f'A3_w0.5_draw{s}'] for s in SEEDS]].mean()
# alternative baselines
for base in ['ratio', 'opt']:
    for pooled in ['pooled', 'per_line']:
        d, _ = evaluate(allm, base=base, pooled=pooled)
        out[f'alt[{base},{pooled}]'] = dict(C0=d[col['c0_total']], A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
# top-10 removed (pooled ranking and per-line ranking)
top = np.argsort(-y, kind='stable')[:10]; km = allm.copy(); km[top] = False
km2 = allm.copy()
for li in range(3):
    idx = np.where(line == li)[0]; km2[idx[np.argsort(-y[idx], kind='stable')[:10]]] = False
out['top10_rows_by_line'] = np.bincount(line[top], minlength=3).tolist()
out['top10_ties_at_cut'] = bool(np.sort(y)[-10] == np.sort(y)[-11])
for nm, msk in [('pooled_top10', km), ('perline_top10', km2)]:
    d, _ = evaluate(msk); out[f'no_{nm}_resolved'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
    d, _ = evaluate(msk, lev_mask=allm, base_mask=allm); out[f'no_{nm}_fixed'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
# subsets
nb = R.new_business.astype(str).str.lower().isin(['true', '1']).to_numpy()
ag = (R.age_band.astype(str) == '17-24').to_numpy()
out['n_nb'] = int(nb.sum()); out['n_1724'] = int(ag.sum()); out['nb_by_line'] = np.bincount(line[nb], minlength=3).tolist(); out['age_by_line'] = np.bincount(line[ag], minlength=3).tolist()
for nm, msk in [('nb', nb), ('age1724', ag)]:
    d, _ = evaluate(msk); out[f'{nm}_resolved_all'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
    d, _ = evaluate(msk, lev_mask=allm, base_mask=allm); out[f'{nm}_fixed_all'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
    d, _ = evaluate(msk, lev_mask=allm); out[f'{nm}_fixedlevel_subsetbase'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
    d, _ = evaluate(msk, base_mask=allm); out[f'{nm}_resolvedlevel_fullbase'] = dict(A2=gain(d, 'A2').mean(), A3=gain(d, 'A3').mean())
    d, dl = evaluate(msk); out[f'{nm}_perline_resolved'] = dict(A2=gain(dl, 'A2').mean(1).tolist(), A3=gain(dl, 'A3').mean(1).tolist())
ev1 = pd.read_csv(EV + 'ev_a32_stage1.csv').set_index('family')
out['reported'] = ev1[['gain', 'p_one_sided', 'ci_low_95', 'ci_high_95', 'gain_without_top10', 'gain_new_business', 'gain_age_17_24', 'raw_d2_mean']].to_dict('index')
print(json.dumps(out, indent=1, default=float))
