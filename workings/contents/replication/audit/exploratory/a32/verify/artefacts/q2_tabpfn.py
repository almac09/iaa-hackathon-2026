"""Post hoc diagnostics of the A3 (TabPFN) arm from its saved predictions. Prints aggregates only."""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from vhelp import LINES, S, SEEDS, Scorer, scored

np.set_printoptions(precision=5, suppress=True)
sc = scored()
scr = Scorer(sc)
tp = {n: pd.read_parquet(S / f"pred_tabpfn_{n}.parquet") for n in LINES}
lg = {n: pd.read_parquet(S / f"pred_lgbm_{n}.parquet") for n in LINES}
meta = json.loads((S / "tabpfn_meta.json").read_text())


def pooled(col_fn):
    return np.concatenate([col_fn(n) for n in LINES])


# --- 0. reproduce the reported numbers ---------------------------------------------------------------------------
print("C0 pooled LM D2", round(scr.d2(scr.c0), 5))
for fam, pat in (("A2", "A2_draw{s}"), ("A3", "A3_w0.5_draw{s}")):
    src = lg if fam == "A2" else tp
    gains = [scr.gain(pooled(lambda n: src[n][pat.format(s=s)].to_numpy(float))) for s in SEEDS]
    print(fam, "gain", round(float(np.mean(gains)), 5), [round(g, 5) for g in gains])

# --- 1. recover the ratio and check the w-family is ratio^w ---------------------------------------------------
ratio = {}
for n in LINES:
    d = sc[n]
    small, large = d.c0_small.to_numpy(float), (d.c0_total - d.c0_small).to_numpy(float)
    for s in SEEDS:
        r1 = (tp[n][f"A3_w1.0_draw{s}"].to_numpy(float) - large) / small
        for w in (0.25, 0.5, 0.75):
            rw = (tp[n][f"A3_w{w}_draw{s}"].to_numpy(float) - large) / small
            err = np.max(np.abs(rw - r1 ** w) / r1 ** w)
            assert err < 1e-6, (n, s, w, err)
        ratio[(n, s)] = r1
print("ratio recovery: (pred - large)/small = ratio^w to 1e-6 for every line, draw and w")

# --- 2. p0 refit on each draw; p = ratio x p0 ------------------------------------------------------------------
y23 = {n: pd.read_parquet(S / f"y23_{n}.parquet", columns=["row_id", "customer_id", "exposure", "c0_total_2023", "n_claims",
                                                          "c0_small"]) for n in LINES}
draws = {n: {int(k): v for k, v in json.loads((S / f"draws_{n}.json").read_text()).items()} for n in LINES}


def z(f):
    return np.c_[np.log(f.c0_total_2023 / f.exposure), np.log(f.exposure)]


def deciles_ae(p, o, k=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), k, labels=False)
    g = pd.DataFrame({"q": q, "p": p, "o": o}).groupby("q").sum()
    return (g.o / g.p).round(2).tolist()


rows = []
for n in LINES:
    d24, f23 = sc[n], y23[n]
    occ24 = (d24.n_claims.to_numpy() > 0).astype(int)
    nb24 = d24.new_business.to_numpy(bool)
    occ23 = (f23.n_claims.to_numpy() > 0).astype(int)
    c0s = d24.c0_small.to_numpy(float)
    c0t = d24.c0_total.to_numpy(float)
    for s in SEEDS:
        pos = np.asarray(draws[n][s])
        dr = f23.iloc[pos]
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(dr), occ23[pos])
        p0 = lr.predict_proba(z(d24))[:, 1]
        p = ratio[(n, s)] * p0
        mrep = meta["lines"][n][str(s)]
        rest = np.setdiff1d(np.arange(len(f23)), pos)
        p0_ctx = lr.predict_proba(z(dr))[:, 1]
        p0_rest = lr.predict_proba(z(f23.iloc[rest]))[:, 1]
        r = ratio[(n, s)]
        rec = {"line": n, "draw": s,
               "mean_p_rep": mrep["mean_p"], "mean_p_rec": float(p.mean()),
               "mean_p0_rep": mrep["mean_p0"], "mean_p0_rec": float(p0.mean()),
               "lr_coef": lr.coef_.ravel().round(3).tolist(),
               "auc24_c0small": roc_auc_score(occ24, c0s), "auc24_c0total": roc_auc_score(occ24, c0t),
               "auc24_p0": roc_auc_score(occ24, p0), "auc24_p": roc_auc_score(occ24, p),
               "auc24_c0s_r05": roc_auc_score(occ24, c0s * r ** 0.5), "auc24_c0s_r1": roc_auc_score(occ24, c0s * r),
               "auc24nb_c0small": roc_auc_score(occ24[nb24], c0s[nb24]), "auc24nb_p": roc_auc_score(occ24[nb24], p[nb24]),
               "auc24nb_c0s_r05": roc_auc_score(occ24[nb24], (c0s * r ** 0.5)[nb24]),
               "auc24nb_p0": roc_auc_score(occ24[nb24], p0[nb24]),
               "auc24_ratio_alone": roc_auc_score(occ24, r),
               "spearman_logr_logc0rate": float(pd.Series(np.log(r)).corr(pd.Series(np.log(c0t / d24.exposure.to_numpy())), method="spearman")),
               "ctx_ae_deciles_p0": deciles_ae(p0_ctx, occ23[pos]),
               "rest23_ae_overall_p0": float(occ23[rest].sum() / p0_rest.sum()),
               "rest23_ae_deciles_p0": deciles_ae(p0_rest, occ23[rest]),
               "rest23_auc_c0small": roc_auc_score(occ23[rest], f23.c0_small.to_numpy()[rest]),
               "y24_ae_overall_p0": float(occ24.sum() / p0.sum()), "y24_ae_overall_p": float(occ24.sum() / p.sum()),
               "y24_ae_deciles_p0": deciles_ae(p0, occ24), "y24_ae_deciles_p": deciles_ae(p, occ24),
               "y24_ae_deciles_ratio_relative": None,
               "log_ratio_sd": float(np.log(r).std()),
               }
        # calibration of the ratio itself: actual/expected of occurrence against p0 by decile of the ratio
        q = pd.qcut(pd.Series(r).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"q": q, "o": occ24, "p0": p0, "r": r}).groupby("q").agg(o=("o", "sum"), p0=("p0", "sum"), r=("r", "mean"))
        lvl = occ24.sum() / p0.sum()
        rec["y24_ae_vs_p0_by_ratio_decile"] = (g.o / g.p0 / lvl).round(2).tolist()
        rec["mean_ratio_by_decile"] = g.r.round(2).tolist()
        rows.append(rec)
        ratio[(n, s, "p0")] = p0

R = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 200)
print(R[["line", "draw", "mean_p_rep", "mean_p_rec", "mean_p0_rep", "mean_p0_rec", "lr_coef"]].to_string())
aucs = [c for c in R.columns if c.startswith("auc") or c.startswith("rest23_auc") or c.startswith("spearman") or c == "log_ratio_sd"]
print(R.groupby("line")[aucs].mean().round(4).T.to_string())
for n in LINES:
    sub = R[R.line == n]
    print(n, "per-draw AUC c0small / p / c0s*r^.5:",
          [(round(a, 4), round(b, 4), round(c, 4)) for a, b, c in zip(sub.auc24_c0small, sub.auc24_p, sub.auc24_c0s_r05)])
for n in LINES:
    sub = R[R.line == n].iloc[0]
    print(n, "draw", sub.draw)
    for k in ("ctx_ae_deciles_p0", "rest23_ae_deciles_p0", "rest23_ae_overall_p0", "y24_ae_overall_p0", "y24_ae_deciles_p0",
              "y24_ae_deciles_p", "y24_ae_vs_p0_by_ratio_decile", "mean_ratio_by_decile"):
        print("   ", k, sub[k])
# averaged across draws: relative A/E by ratio decile
for n in LINES:
    m = np.mean(np.stack(R[R.line == n].y24_ae_vs_p0_by_ratio_decile.to_numpy()), axis=0)
    mr = np.mean(np.stack(R[R.line == n].mean_ratio_by_decile.to_numpy()), axis=0)
    print(n, "mean over draws: occurrence A/E vs p0 by ratio decile", m.round(2), "mean ratio", mr.round(2))

# --- 3. w scan (post hoc, chosen on 2024 itself: an upper bound for the power family) -------------------------
ws = np.round(np.arange(0, 1.01, 0.05), 2)
res = {}
for w in ws:
    cols = []
    for s in SEEDS:
        cols.append(pooled(lambda n: sc[n].c0_small.to_numpy(float) * ratio[(n, s)] ** w + (sc[n].c0_total - sc[n].c0_small).to_numpy(float)))
    res[w] = (float(np.mean([scr.gain(c) for c in cols])),
              [float(np.mean([scr.line_gain(c, i) for c in cols])) for i in range(3)],
              float(np.mean([scr.gain(c, scr.NB) for c in cols])))
print("w scan: pooled gain, per-line gain (b, c, m), NB gain")
for w, (g, lgn, nb) in res.items():
    print(f"  w={w:.2f} pooled {g:+.5f}  lines {[round(x, 5) for x in lgn]}  NB {nb:+.5f}")
best = max(res, key=lambda w: res[w][0])
print("best pooled w", best, res[best][0])
# per-line best w (oracle)
for i, n in enumerate(LINES):
    bw = max(res, key=lambda w: res[w][1][i])
    print(" best w for", n, bw, round(res[bw][1][i], 5))

# cross-fitted w: choose the pooled w on a random half of customers, score on the other half (both ways)
rng = np.random.default_rng(7)
half = rng.integers(0, 2, scr.ncust)[scr.codes].astype(bool)
devs = {w: [scr.lm_dev(pooled(lambda n: sc[n].c0_small.to_numpy(float) * ratio[(n, s)] ** w + (sc[n].c0_total - sc[n].c0_small).to_numpy(float)))
            for s in SEEDS] for w in ws}
out = []
for A, B in ((half, ~half), (~half, half)):
    gA = {w: np.mean([(scr.dev_c0[A].sum() - d[A].sum()) / scr.null[A].sum() for d in devs[w]]) for w in ws}
    wA = max(gA, key=gA.get)
    gB = np.mean([(scr.dev_c0[B].sum() - d[B].sum()) / scr.null[B].sum() for d in devs[wA]])
    out.append((float(wA), float(gB)))
print("cross-fitted w (chosen on one half, gain on the other):", out)
lo, hi, p = scr.boot(devs[best], n_boot=1000)
print(f"oracle-w {best} customer bootstrap 95% [{lo:+.5f}, {hi:+.5f}], one-sided p {p:.3f}")

# --- 4. A2 / LightGBM occurrence AUC for comparison ------------------------------------------------------------
for n in LINES:
    occ24 = (sc[n].n_claims.to_numpy() > 0).astype(int)
    large = (sc[n].c0_total - sc[n].c0_small).to_numpy(float)
    print(n, "AUC lgbm_source small", round(roc_auc_score(occ24, lg[n].lgbm_source.to_numpy() - large), 4),
          "A2 small (mean over draws)", round(np.mean([roc_auc_score(occ24, lg[n][f'A2_draw{s}'].to_numpy() - large) for s in SEEDS]), 4),
          "C0 small", round(roc_auc_score(occ24, sc[n].c0_small.to_numpy()), 4))
