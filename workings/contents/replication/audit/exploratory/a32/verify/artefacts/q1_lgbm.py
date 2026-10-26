"""Post hoc LightGBM diagnostics for A2 (A32 env). Fits on the 2021-22 source only; early stopping on all of 2023 as the
protocol does; 2024 scored rows used only to report. Prints aggregates; writes no row-level file."""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from vhelp import LINES, S, SEEDS, Scorer, ch, opt_scale, scored, unit_dev

which = sys.argv[1] if len(sys.argv) > 1 else "all"
sc = scored()
scr = Scorer(sc)
P = 1.5


def lm_d2_small(y, mu, e):
    """Small-claim level-matched D2 against a constant rate on these rows."""
    base = y.sum() / e.sum() * e
    return 1 - unit_dev(y, mu * opt_scale(y, mu)).sum() / unit_dev(y, base).sum()


data = {}
for n in LINES:
    feats, fr, draws, c0 = ch.load(S, n)
    cols = feats["numeric"] + feats["categorical"]
    trend = {k: c0["slope"] * (f.yr.to_numpy() - c0["centre"]) for k, f in fr.items()}
    rate = fr["src"].y_small.sum() / (fr["src"].exposure * np.exp(trend["src"])).sum()
    init = {k: np.log(f.exposure.to_numpy() * rate) + trend[k] for k, f in fr.items()}
    data[n] = dict(feats=feats, fr=fr, draws=draws, c0=c0, cols=cols, init=init, rate=rate)
    if which in ("all", "cats"):
        src, y24 = fr["src"], fr["y24"]
        card = {c: int(src[c].nunique()) for c in feats["categorical"]}
        unseen = {c: float((~y24[c].isin(set(src[c].dropna().unique())) & y24[c].notna()).mean()) for c in feats["categorical"]}
        print(n, "categorical cardinality (src):", {c: v for c, v in sorted(card.items(), key=lambda x: -x[1])[:8]},
              "| share of 2024 rows with a level unseen in src (max):", max(unseen.items(), key=lambda x: x[1]))


def fit(n, params, rounds=3000, es=True, cols=None, transform=None):
    d = data[n]
    cols = cols or d["cols"]
    fr = d["fr"]
    X = {k: (transform(fr[k][cols]) if transform else fr[k][cols]) for k in ("src", "y23", "y24")}
    cats = [c for c in d["feats"]["categorical"] if c in cols and str(X["src"][c].dtype) == "category"]
    dtrain = lgb.Dataset(X["src"], label=fr["src"].y_small, init_score=d["init"]["src"], categorical_feature=cats, free_raw_data=False)
    dval = lgb.Dataset(X["y23"], label=fr["y23"].y_small, init_score=d["init"]["y23"], reference=dtrain)
    ev = {}
    cb = [lgb.record_evaluation(ev)] + ([lgb.early_stopping(ch.EARLY_STOP, verbose=False)] if es else [])
    bst = lgb.train({**ch.LGBM, **params}, dtrain, rounds, valid_sets=[dval], valid_names=["v"], callbacks=cb)
    return bst, X, ev["v"]["tweedie"]


def scores(n, bst, X, k):
    d = data[n]
    fr = d["fr"]
    mu23 = np.exp(bst.predict(X["y23"], num_iteration=k, raw_score=True) + d["init"]["y23"])
    mu24 = np.exp(bst.predict(X["y24"], num_iteration=k, raw_score=True) + d["init"]["y24"])
    d23 = lm_d2_small(fr["y23"].y_small.to_numpy(), mu23, fr["y23"].exposure_plain.to_numpy())
    ae23 = fr["y23"].y_small.sum() / mu23.sum()
    return d23, ae23, mu24


def gain24(small24: dict):
    """Pooled 2024 gain against C0 of (LightGBM small + C0 large), given each line's small prediction."""
    m = np.concatenate([small24[n] + (sc[n].c0_total - sc[n].c0_small).to_numpy(float) for n in LINES])
    return scr.gain(m), [scr.line_gain(m, i) for i in range(3)]


c0_23 = {n: lm_d2_small(data[n]["fr"]["y23"].y_small.to_numpy(), data[n]["fr"]["y23"].c0_small.to_numpy(),
                        data[n]["fr"]["y23"].exposure_plain.to_numpy()) for n in LINES}
print("C0 small-claim LM D2 on all 2023 rows (C0 buildings' trees early-stopped there too):", {n: round(v, 5) for n, v in c0_23.items()})

# --- 1a. the validation curve of the pre-registered model, run to 1,000 trees without early stopping ------------
if which in ("all", "curve"):
    grid = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 750, 1000]
    curve_small = {k: {} for k in grid}
    for n in LINES:
        t0 = time.time()
        bst, X, nll = fit(n, {}, rounds=1000, es=False)
        nll = np.asarray(nll)
        best = int(np.argmin(nll[: 1000]) + 1)
        # the early-stopping rule: first iteration with no improvement over the next 100
        es_best = next(i + 1 for i in range(len(nll)) if i + 100 < len(nll) and nll[i] <= nll[i + 1: i + 101].min())
        print(f"== {n}: early-stopping rule picks {es_best} (reported {json.loads((S / 'lgbm_meta.json').read_text())[n]['trees']}); "
              f"global min of 2023 NLL within 1,000 at {best}; NLL rel. to its min at 100/200/500/1000: "
              f"{[round(float(nll[k - 1] / nll.min() - 1), 5) for k in (100, 200, 500, 1000)]}")
        for k in grid:
            d23, ae23, mu24 = scores(n, bst, X, k)
            curve_small[k][n] = mu24
            print(f"   trees {k:5d}: 2023 small LM D2 {d23:.5f} (C0 {c0_23[n]:.5f}); 2023 A/E {ae23:.3f}")
        print(f"   ({time.time() - t0:.0f}s)")
    for k in grid:
        g, gl = gain24(curve_small[k])
        print(f"   2024 pooled gain vs C0 at {k:5d} trees (all lines): {g:+.5f}  per line {[round(x, 5) for x in gl]}")

# --- 1b. variants (post hoc), each early-stopped on 2023 exactly as the protocol's model ------------------------
if which in ("all", "variants"):
    def hess_param(n):
        d = data[n]
        f = d["fr"]["src"]
        mu = d["rate"] * f.exposure.to_numpy()
        h = -f.y_small.to_numpy() * (1 - P) * mu ** (1 - P) + (2 - P) * mu ** (2 - P)
        return float(np.median(h))

    def codes(df):
        out = df.copy()
        for c in out.columns:
            if str(out[c].dtype) == "category" and out[c].cat.categories.size > 32:
                out[c] = out[c].cat.codes.replace(-1, np.nan).astype(float)
        return out

    variants = {
        "pre-registered (31 leaves, 200 rows)": ({}, None, None),
        "31 leaves, 2,000 rows": ({"min_data_in_leaf": 2000}, None, None),
        "15 leaves, 2,000 rows": ({"num_leaves": 15, "min_data_in_leaf": 2000}, None, None),
        "7 leaves, 5,000 rows": ({"num_leaves": 7, "min_data_in_leaf": 5000}, None, None),
        "C0-like: depth 3, hessian 20,000 x median": ("hess", None, None),
        "pre-reg, smoothed categoricals": ({"cat_smooth": 100, "cat_l2": 100, "min_data_per_group": 1000, "max_cat_threshold": 8}, None, None),
        "pre-reg, >32-level categoricals as codes": ({}, None, codes),
        "7 leaves, 5,000 rows, >32-level cats as codes": ({"num_leaves": 7, "min_data_in_leaf": 5000}, None, codes),
        "pre-reg, no trend in offset": ({}, "notrend", None),
    }
    out = {}
    for lab, (params, flag, tf) in variants.items():
        small24, info = {}, {}
        for n in LINES:
            pr = params
            if params == "hess":
                pr = {"max_depth": 3, "num_leaves": 8, "min_data_in_leaf": 1, "min_sum_hessian_in_leaf": 20000 * hess_param(n)}
            if flag == "notrend":
                d = data[n]
                saved = d["init"]
                rate0 = d["fr"]["src"].y_small.sum() / d["fr"]["src"].exposure.sum()
                d["init"] = {k: np.log(f.exposure.to_numpy() * rate0) for k, f in d["fr"].items()}
                bst, X, _ = fit(n, pr, transform=tf)
                d23, ae23, mu24 = scores(n, bst, X, bst.best_iteration)
                # back on the trended 2024 basis is irrelevant under level matching; keep as fitted
                d["init"] = saved
            else:
                bst, X, _ = fit(n, pr, transform=tf)
                d23, ae23, mu24 = scores(n, bst, X, bst.best_iteration)
            small24[n] = mu24
            info[n] = (bst.best_iteration, round(d23, 5))
        g, gl = gain24(small24)
        out[lab] = (g, gl, info)
        print(f"{lab:48s} trees/2023 small D2 {info}  | 2024 pooled gain {g:+.5f} lines {[round(x, 5) for x in gl]}", flush=True)
    print("C0 2023 small D2", {n: round(v, 5) for n, v in c0_23.items()})
