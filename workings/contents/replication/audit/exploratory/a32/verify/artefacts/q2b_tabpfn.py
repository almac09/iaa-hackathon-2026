"""Post hoc: why a better occurrence ranking does not become a better cost ranking (A3), and draw-pool selection."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from vhelp import LINES, S, SEEDS, Scorer, scored

sc = scored()
scr = Scorer(sc)
tp = {n: pd.read_parquet(S / f"pred_tabpfn_{n}.parquet") for n in LINES}
y23 = {n: pd.read_parquet(S / f"y23_{n}.parquet", columns=["customer_id", "exposure", "exposure_plain", "c0_total_2023",
                                                          "c0_small", "c0_total", "n_claims", "y_small"]) for n in LINES}
draws = {n: {int(k): v for k, v in json.loads((S / f"draws_{n}.json").read_text()).items()} for n in LINES}


def z(f):
    return np.c_[np.log(f.c0_total_2023 / f.exposure), np.log(f.exposure)]


for i, n in enumerate(LINES):
    d = sc[n]
    small, large = d.c0_small.to_numpy(float), (d.c0_total - d.c0_small).to_numpy(float)
    ys = d.y_small.to_numpy(float)
    occ = (d.n_claims.to_numpy() > 0).astype(int)
    e = d.exposure_plain.to_numpy(float)
    f23 = y23[n]
    occ23 = (f23.n_claims.to_numpy() > 0).astype(int)
    acc = {"occ_ae": [], "cost_ae": [], "sev_rel": [], "ratio": []}
    full_auc, full_auc0, gains_full, gains_part = [], [], [], []
    by_e = []
    for s in SEEDS:
        r = (tp[n][f"A3_w1.0_draw{s}"].to_numpy(float) - large) / small
        pos = np.asarray(draws[n][s])
        p0 = LogisticRegression(C=1e6, max_iter=1000).fit(z(f23.iloc[pos]), occ23[pos]).predict_proba(z(d))[:, 1]
        q = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False).to_numpy()
        g = pd.DataFrame({"q": q, "occ": occ, "p0": p0, "ys": ys, "c0s": small, "r": r, "nc": d.n_claims.to_numpy()}).groupby("q").sum()
        lvl_o, lvl_c = occ.sum() / p0.sum(), ys.sum() / small.sum()
        acc["occ_ae"].append((g.occ / g.p0 / lvl_o).to_numpy())
        acc["cost_ae"].append((g.ys / g.c0s / lvl_c).to_numpy())
        sev = g.ys / g.nc
        acc["sev_rel"].append((sev / (ys.sum() / d.n_claims.sum())).to_numpy())
        acc["ratio"].append(pd.Series(r).groupby(q).mean().to_numpy())
        full = e >= 0.99
        full_auc.append(roc_auc_score(occ[full], (small * r)[full]))
        full_auc0.append(roc_auc_score(occ[full], small[full]))
        # occurrence A/E vs p0 and mean ratio by exposure band
        band = pd.cut(e, [0, 0.25, 0.5, 0.75, 0.99, 1.01], labels=False)
        gb = pd.DataFrame({"b": band, "occ": occ, "p0": p0, "p": r * p0, "r": r}).groupby("b").agg(occ=("occ", "sum"), p0=("p0", "sum"), p=("p", "sum"), r=("r", "mean"), n=("r", "size"))
        by_e.append(gb)
    print(f"== {n}")
    for k, v in acc.items():
        print(f"  by ratio decile, mean over draws, {k}:", np.round(np.mean(v, axis=0), 2).tolist())
    print(f"  full-exposure rows ({full.mean():.0%}): AUC C0 small {np.mean(full_auc0):.4f}, C0 small x ratio {np.mean(full_auc):.4f}")
    gb = sum(by_e) / len(by_e)
    print("  by exposure band (0-.25,.25-.5,.5-.75,.75-.99,full): rows", gb.n.round(0).tolist(), " occ A/E vs p0",
          (gb.occ / gb.p0).round(3).tolist(), " occ A/E vs p", (gb.occ / gb.p).round(3).tolist(), " mean ratio", gb.r.round(3).tolist())

    # gain restricted to full-exposure rows (deviance at the all-rows level), w = 0.5
    k = scr.line == i
    full_all = np.zeros(len(scr.Y), bool)
    full_all[np.flatnonzero(k)[e >= 0.99]] = True
    cols = []
    for s in SEEDS:
        m = scr.c0.copy()
        m[k] = tp[n][f"A3_w0.5_draw{s}"].to_numpy(float)
        cols.append(m)
    gf = np.mean([(scr.dev_c0[full_all].sum() - scr.lm_dev(c)[full_all].sum()) / scr.null[full_all].sum() for c in cols])
    gp = np.mean([(scr.dev_c0[k & ~full_all].sum() - scr.lm_dev(c)[k & ~full_all].sum()) / scr.null[k & ~full_all].sum() for c in cols])
    print(f"  A3 w0.5 gain on full-exposure rows {gf:+.5f}, on part-year rows {gp:+.5f}")

# --- draw pool selection (amendment 1): is the pool a biased sample of 2023? ----------------------------------
print("== draw pool vs the rest of 2023 (occurrence and small-cost A/E against C0's 2023 prediction)")
for n in LINES:
    f23 = y23[n]
    pool = ~f23.customer_id.isin(set(sc[n].customer_id)).to_numpy()
    occ23 = (f23.n_claims.to_numpy() > 0)
    for lab, msk in (("pool", pool), ("excluded (in 2024 sample)", ~pool)):
        print(f"  {n:9s} {lab:26s} rows {msk.sum():7d}  occurrence {occ23[msk].mean():.4f}  small A/E vs C0 {f23.y_small[msk].sum() / f23.c0_small[msk].sum():.3f}"
              f"  mean exposure {f23.exposure_plain[msk].mean():.3f}  part-year share {(f23.exposure_plain[msk] < 0.99).mean():.3f}")
    drawn = np.zeros(len(f23), bool)
    for s in SEEDS:
        drawn[np.asarray(draws[n][s])] = True
    print(f"  {n:9s} union of draws rows {drawn.sum()}, part-year share {(f23.exposure_plain[drawn] < 0.99).mean():.3f}; "
          f"2024 scored part-year share {(sc[n].exposure_plain < 0.99).mean():.3f}; 2024 scored NB share {sc[n].new_business.mean():.3f}")
