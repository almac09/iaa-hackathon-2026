"""Post hoc: the A2 leaf refit (the protocol's procedure, challengers.py functions unchanged) applied to better-specified
LightGBM source models, fitted on 2021-22 and early-stopped on 2023. Aggregates only."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from vhelp import LINES, S, SEEDS, Scorer, ch, opt_scale, scored, unit_dev

sc = scored()
scr = Scorer(sc)
P = 1.5


def lm_d2_small(y, mu, e):
    base = y.sum() / e.sum() * e
    return 1 - unit_dev(y, mu * opt_scale(y, mu)).sum() / unit_dev(y, base).sum()


def codes(df):
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category" and out[c].cat.categories.size > 32:
            out[c] = out[c].cat.codes.replace(-1, np.nan).astype(float)
    return out


VARIANTS = {"pre-registered": ("pre", None), "C0-like": ("hess", None), "C0-like + codes": ("hess", codes),
            "7 leaves 5,000 rows + codes": ({"num_leaves": 7, "min_data_in_leaf": 5000}, codes)}
results = {v: {"src": {}, "draws": {s: {} for s in SEEDS}, "d23": {}, "trees": {}, "decay": {}} for v in VARIANTS}
for n in LINES:
    feats, fr, draws, c0 = ch.load(S, n)
    cols = feats["numeric"] + feats["categorical"]
    src, y23, y24 = fr["src"], fr["y23"], fr["y24"]
    trend = {k: c0["slope"] * (f.yr.to_numpy() - c0["centre"]) for k, f in fr.items()}
    rate = src.y_small.sum() / (src.exposure * np.exp(trend["src"])).sum()
    init = {k: np.log(f.exposure.to_numpy() * rate) + trend[k] for k, f in fr.items()}
    mu0 = rate * src.exposure.to_numpy()
    med_h = float(np.median(-src.y_small.to_numpy() * (1 - P) * mu0 ** (1 - P) + (2 - P) * mu0 ** (2 - P)))
    large24 = (y24.c0_total - y24.c0_small).to_numpy()
    for v, (pr, tf) in VARIANTS.items():
        t0 = time.time()
        params = {} if pr == "pre" else ({"max_depth": 3, "num_leaves": 8, "min_data_in_leaf": 1,
                                          "min_sum_hessian_in_leaf": 20000 * med_h} if pr == "hess" else pr)
        X = {k: (tf(f[cols]) if tf else f[cols]) for k, f in fr.items()}
        cats = [c for c in feats["categorical"] if str(X["src"][c].dtype) == "category"]
        dtrain = lgb.Dataset(X["src"], label=src.y_small, init_score=init["src"], categorical_feature=cats, free_raw_data=False)
        dval = lgb.Dataset(X["y23"], label=y23.y_small, init_score=init["y23"], reference=dtrain)
        bst = lgb.train({**ch.LGBM, **params}, dtrain, ch.MAX_TREES, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(ch.EARLY_STOP, verbose=False)])
        nt = bst.best_iteration
        values = ch.leaf_values(bst.dump_model(num_iteration=nt)["tree_info"])
        lv = {k: bst.predict(X[k], num_iteration=nt, pred_leaf=True).reshape(len(X[k]), -1) for k in ("y23", "y24")}
        chk = ch.margin(lv["y24"], values, init["y24"]) - (bst.predict(X["y24"], num_iteration=nt, raw_score=True) + init["y24"])
        assert np.abs(chk).max() < 1e-6
        r = results[v]
        r["trees"][n] = nt
        r["d23"][n] = round(lm_d2_small(y23.y_small.to_numpy(), np.exp(ch.margin(lv["y23"], values, init["y23"])), y23.exposure_plain.to_numpy()), 5)
        r["src"][n] = np.exp(ch.margin(lv["y24"], values, init["y24"])) + large24
        chosen = {}
        for seed, rows in draws.items():  # the protocol's refit, verbatim in logic (lgbm() in challengers.py)
            pos = np.asarray(rows)
            leaves_d, init_d, yd = lv["y23"][pos], init["y23"][pos], y23.y_small.to_numpy()[pos]
            folds = ch.customer_folds(y23.customer_id.to_numpy()[pos], seed)
            cv = {}
            for d in ch.DECAYS:
                dev = 0.0
                for k in range(5):
                    trm, tem = folds != k, folds == k
                    nv = ch.refit_leaves(leaves_d[trm], values, init_d[trm], yd[trm], d, ch.LGBM["learning_rate"], ch.LGBM["lambda_l2"])
                    mu = np.exp(ch.margin(leaves_d[tem], nv, init_d[tem]))
                    dev += unit_dev(yd[tem], mu * opt_scale(yd[tem], mu)).sum()
                cv[d] = dev
            best = min(cv, key=cv.get)
            nvd = ch.refit_leaves(leaves_d, values, init_d, yd, best, ch.LGBM["learning_rate"], ch.LGBM["lambda_l2"])
            r["draws"][seed][n] = np.exp(ch.margin(lv["y24"], nvd, init["y24"])) + large24
            chosen[seed] = best
        r["decay"][n] = chosen
        print(f"{n} {v}: trees {nt}, 2023 small D2 {r['d23'][n]}, decays {chosen} ({time.time() - t0:.0f}s)", flush=True)

for v, r in results.items():
    src_m = np.concatenate([r["src"][n] for n in LINES])
    arms = [np.concatenate([r["draws"][s][n] for n in LINES]) for s in SEEDS]
    gs = [scr.gain(a) for a in arms]
    lo, hi, p = scr.boot([scr.lm_dev(a) for a in arms], n_boot=1000)
    print(f"== {v}: source gain {scr.gain(src_m):+.5f} lines {[round(scr.line_gain(src_m, i), 5) for i in range(3)]}")
    print(f"   refit arm gain {np.mean(gs):+.5f} (draws {[round(g, 5) for g in gs]}), 95% [{lo:+.5f}, {hi:+.5f}], p {p:.3f}; "
          f"lines {[round(float(np.mean([scr.line_gain(a, i) for a in arms])), 5) for i in range(3)]}; "
          f"NB {np.mean([scr.gain(a, scr.NB) for a in arms]):+.5f}")
    print(f"   trees {r['trees']}, 2023 small D2 {r['d23']}")
