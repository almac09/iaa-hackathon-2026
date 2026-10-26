"""A32 Stage 1: GPU and transfer challengers against C0 on the developed-market backtest (audit/exploratory/a32/PROTOCOL.md,
including its pre-score amendments).

Separate processes, because LightGBM, PyTorch and XGBoost bring their own OpenMP runtimes (A20):
  prepare  (pack .venv)  the developed market on the portable features; C0 refitted on 2021-22 with its unchanged
                         procedures; row-level frames for the source years, developed 2023 and a fixed 100,000-row
                         sample of developed 2024 per line (its outcomes in a separate file only `score` reads); five
                         customer draws of 2023, the size of each emerging line, from customers not in the 2024 sample.
  lgbm     (A32 env)     A2: a Tweedie LightGBM on the small-claim target with C0's trend in its offset, fitted on
                         2021-22, its leaves refitted on each draw (refit_leaves).
  tabpfn   (A32 env)     A3: a TabPFN-3.5 claim-occurrence stacker on each draw, on the Apple GPU, from local weights
                         only (offline, no browser); `--smoke` runs the Apple-GPU-against-CPU agreement check instead.
  score    (pack .venv)  the pre-registered metric, paired customer bootstrap, Holm (m = 2) and the pass gate.
Row-level frames go to the scratch folder given on the command line; only aggregates reach audit/evidence/.

Usage (from the pack root):
  python python/challengers.py prepare <scratch>
  <a32 env>/bin/python python/challengers.py lgbm <scratch>
  TABPFN_MODEL_CACHE_DIR=<weights folder> <a32 env>/bin/python python/challengers.py tabpfn <scratch> [mps|cpu] [--smoke]
  python python/challengers.py score <scratch>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P = 1.5
LINES = ("buildings", "contents", "motor")
DRAW_ROWS = {"buildings": 12_580, "contents": 15_834, "motor": 21_680}  # emerging 2023 rows per line (A30)
DRAW_SEEDS = (2032, 2033, 2034, 2035, 2036)
SCORE_ROWS, SCORE_SEED = 100_000, 2032
DECAYS = (0.5, 0.8, 0.9, 0.95, 0.98)
LGBM = {"objective": "tweedie", "tweedie_variance_power": P, "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "seed": 2032, "num_threads": 8, "verbose": -1}
MAX_TREES, EARLY_STOP = 3000, 100
WEIGHTS = (0.25, 0.5, 0.75, 1.0)  # A3 blend weights; 0.5 is the pre-registered primary
TABPFN_WEIGHTS = "tabpfn-v3.5-20260909.safetensors"
TABPFN_ESTIMATORS, CHUNK = 8, 10_000
FAMILIES_M = 2  # Holm's m, fixed by the protocol
PROTOCOL = Path(__file__).resolve().parents[1] / "audit/exploratory/a32/PROTOCOL.md"


def unit_dev(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return 2 * (np.power(y, 2 - P) / ((1 - P) * (2 - P)) - y * np.power(mu, 1 - P) / (1 - P)
                + np.power(mu, 2 - P) / (2 - P))


def opt_scale(y, mu) -> float:
    return float((y * mu ** (1 - P)).sum() / (mu ** (2 - P)).sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- prepare ---------------------------------------------------------------------------------------------------------

def features_for(pcfg: dict, name: str) -> dict:
    """The shared features each arm uses. Contents takes C0's portable GLM inputs (log sum insured, raw elevation and
    river distance), not the within-market quantile bands, so an arm that passes can carry over to the emerging
    market (amendment 7); buildings and motor take the portable tree-model lists."""
    import run_glm_lines as rgl  # noqa: PLC0415
    if name == "contents":
        g = pcfg["glm_lines"]["models"]["contents"]
        return {"numeric": list(g["numeric"]) + rgl.resolve(pcfg, g.get("continuous", [])),
                "categorical": rgl.resolve(pcfg, g["categorical"])}
    lm = pcfg["lines"]["models"][name]
    return {"numeric": list(lm["numeric"]), "categorical": list(lm["categorical"])}


def prepare(out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # noqa: PLC0415
    import emerging as em  # noqa: PLC0415
    from gap_tests import settings  # noqa: PLC0415

    t0 = time.time()
    cfg = common.load_config()
    pcfg = em.portable(cfg)
    rec = common.new_run_record("challengers_prepare", cfg) | {"protocol_sha256": sha256(PROTOCOL)}
    dev, _ = em.load_market(pcfg, "developed", rec)
    summary = pd.read_csv(common.PACK_ROOT / "audit/evidence/glm_lines/ev_g6_summary.csv")
    meta = {"features": {}, "rows": {}, "c0": {}}
    for name in LINES:
        line, s = dev[name], settings(dev[name], summary)
        feats = features_for(pcfg, name)
        line.use_window("tune")  # C0's own exposure basis for the 2021-22 fit (seasonal for contents, D-22)
        trees = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
        m = line.build([2021, 2022], s["threshold"], s["alpha"], trees)
        rows = {"src": line.rows([2021, 2022]), "y23": line.rows([2023])}
        rows["y24"] = line.rows([2024]).sample(n=SCORE_ROWS, random_state=SCORE_SEED)
        frames = {}
        for key, d in rows.items():
            y_small, _ = line.targets(d, s["threshold"])
            f = d[feats["numeric"] + feats["categorical"]].copy()
            f["row_id"], f["customer_id"] = d.row_id.to_numpy(), d.customer_id.astype(str).to_numpy()
            f["yr"], f["exposure"], f["exposure_plain"] = d.yr.to_numpy(), d.exposure.to_numpy(), d.exposure_plain.to_numpy()
            if key != "src":
                yr = int(d.yr.iloc[0])
                f["c0_small"], f["c0_total"] = m.small.predict(d, year=yr), m.predict(d, year=yr)
                f["c0_total_2023"] = m.predict(d, year=2023)  # A3's feature, at one trend year for context and test
            outcomes = pd.DataFrame({"row_id": d.row_id.to_numpy(), "customer_id": d.customer_id.astype(str).to_numpy(),
                                     "y_small": y_small, "y_full": d.y_full.to_numpy(), "n_claims": d.n_claims.to_numpy(),
                                     "new_business": d.new_business.astype(bool).to_numpy(),
                                     "age_band": d[em.BAND[name]].astype("string").to_numpy()})
            if key == "y24":  # 2024 outcomes in their own file, read only by `score` (amendment 9)
                outcomes.reset_index(drop=True).to_parquet(out / f"y24out_{name}.parquet", index=False)
            else:
                for c in ("y_small", "n_claims"):
                    f[c] = outcomes[c].to_numpy()
            for c in f.columns:
                if f[c].dtype == object:
                    f[c] = f[c].astype("string")
            f = f.reset_index(drop=True)  # draws are row positions in the saved frame
            f.to_parquet(out / f"{key}_{name}.parquet", index=False)
            frames[key] = f
        y23 = frames["y23"]
        pool = np.setdiff1d(y23.customer_id.unique(), frames["y24"].customer_id.unique())  # amendment 1
        counts = y23.groupby("customer_id").size()
        draws = {}
        for seed in DRAW_SEEDS:  # whole customers, in random order, until the draw reaches the emerging line's size
            order = np.random.default_rng(seed).permutation(pool)
            n = int(np.searchsorted(np.cumsum(counts.reindex(order).to_numpy()), DRAW_ROWS[name])) + 1
            draws[str(seed)] = np.flatnonzero(y23.customer_id.isin(set(order[:n])).to_numpy()).tolist()
        (out / f"draws_{name}.json").write_text(json.dumps(draws))
        meta["features"][name] = feats
        meta["rows"][name] = {k: len(v) for k, v in frames.items()} | {"draw_rows": [len(v) for v in draws.values()],
                                                                      "draw_pool_customers": int(len(pool))}
        meta["c0"][name] = {"threshold": s["threshold"], "trees": trees[0] if trees else None,
                            "slope": float(m.small.slope), "centre": float(m.small.centre)}
        print(name, meta["rows"][name], meta["c0"][name], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    rec |= {"status": "success", "meta": meta, "elapsed_s": round(time.time() - t0, 1)}
    common.append_run_log(cfg, rec)


# --- shared loading for the arms ------------------------------------------------------------------------------------

def load(out: Path, name: str):
    meta = json.loads((out / "meta.json").read_text())
    feats = meta["features"][name]
    fr = {k: pd.read_parquet(out / f"{k}_{name}.parquet") for k in ("src", "y23", "y24")}
    for c in feats["categorical"]:  # one category list across all frames, so codes agree
        levels = sorted(set().union(*[set(f[c].dropna().astype(str)) for f in fr.values()]))
        for f in fr.values():
            f[c] = pd.Categorical(f[c].astype("string").astype(object).where(f[c].notna(), None), categories=levels)
    for c in feats["numeric"]:  # plain floats: neither model takes pandas' nullable integer or boolean types
        for f in fr.values():
            f[c] = pd.to_numeric(f[c].astype("Float64"), errors="coerce").astype(float)
    draws = {int(k): v for k, v in json.loads((out / f"draws_{name}.json").read_text()).items()}
    return feats, fr, draws, meta["c0"][name]


def customer_folds(customers, seed: int, k: int = 5) -> np.ndarray:
    """Seeded random fold per customer (amendment 3)."""
    codes, uniq = pd.factorize(pd.Series(customers).astype(str))
    return np.random.default_rng(seed).integers(0, k, len(uniq))[codes]


# --- A2: leaf refit ---------------------------------------------------------------------------------------------------

def tweedie_grad_hess(y, f):
    """Gradient and hessian of the Tweedie loss in the raw (log) score, as LightGBM's objective defines them."""
    a, b = np.exp((1 - P) * f), np.exp((2 - P) * f)
    return -y * a + b, -y * (1 - P) * a + (2 - P) * b


def refit_leaves(leaves: np.ndarray, values: list[np.ndarray], init: np.ndarray, y: np.ndarray, decay: float,
                 lr: float, l2: float) -> list[np.ndarray]:
    """Refit each tree's leaves on target rows, in tree order. A leaf with target rows becomes decay x its value
    + (1 - decay) x the Newton step at the current margin (times the learning rate); a leaf with none keeps its
    value; the initial score is kept. `leaves` is (rows, trees) leaf indices; `values` the source leaf values."""
    f = init.astype(float).copy()
    new = []
    for t, v in enumerate(values):
        idx = leaves[:, t]
        g, h = tweedie_grad_hess(y, f)
        n = np.bincount(idx, minlength=len(v))
        step = -np.bincount(idx, weights=g, minlength=len(v)) / (np.bincount(idx, weights=h, minlength=len(v)) + l2) * lr
        nv = np.where(n > 0, decay * v + (1 - decay) * step, v)
        f += nv[idx]
        new.append(nv)
    return new


def margin(leaves: np.ndarray, values: list[np.ndarray], init: np.ndarray) -> np.ndarray:
    f = init.astype(float).copy()
    for t, v in enumerate(values):
        f += v[leaves[:, t]]
    return f


def leaf_values(dump: list[dict]) -> list[np.ndarray]:
    """Each tree's leaf values by leaf index, from LightGBM's model dump (they already include the learning rate)."""
    values = []
    for t in dump:
        v = np.zeros(t["num_leaves"])
        stack = [t["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                v[node["leaf_index"]] = node["leaf_value"]
            elif "split_index" in node:
                stack += [node["left_child"], node["right_child"]]
            else:  # a single-leaf tree
                v[0] = node.get("leaf_value", 0.0)
        values.append(v)
    return values


def lgbm(out: Path) -> None:
    import lightgbm as lgb  # noqa: PLC0415

    res = {"lightgbm": lgb.__version__}
    for name in LINES:
        feats, fr, draws, c0 = load(out, name)
        cols = feats["numeric"] + feats["categorical"]
        src, y23, y24 = fr["src"], fr["y23"], fr["y24"]
        trend = {k: c0["slope"] * (f.yr.to_numpy() - c0["centre"]) for k, f in fr.items()}  # amendment 2
        rate = src.y_small.sum() / (src.exposure * np.exp(trend["src"])).sum()
        init = {k: np.log(f.exposure.to_numpy() * rate) + trend[k] for k, f in fr.items()}
        dtrain = lgb.Dataset(src[cols], label=src.y_small, init_score=init["src"], categorical_feature=feats["categorical"],
                             free_raw_data=False)
        dval = lgb.Dataset(y23[cols], label=y23.y_small, init_score=init["y23"], reference=dtrain)
        bst = lgb.train(LGBM, dtrain, MAX_TREES, valid_sets=[dval], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        n_trees = bst.best_iteration
        values = leaf_values(bst.dump_model(num_iteration=n_trees)["tree_info"])
        lv = {k: bst.predict(f[cols], num_iteration=n_trees, pred_leaf=True).reshape(len(f), -1) for k, f in (("y23", y23), ("y24", y24))}
        check = margin(lv["y24"], values, init["y24"]) - (bst.predict(y24[cols], num_iteration=n_trees, raw_score=True) + init["y24"])
        assert np.abs(check).max() < 1e-6, f"{name}: leaf values do not reproduce LightGBM's score ({np.abs(check).max()})"
        large24 = (y24.c0_total - y24.c0_small).to_numpy()
        pred = {"lgbm_source": np.exp(margin(lv["y24"], values, init["y24"])) + large24}
        chosen = {}
        for seed, rows in draws.items():
            pos = np.asarray(rows)
            leaves_d, init_d, yd = lv["y23"][pos], init["y23"][pos], y23.y_small.to_numpy()[pos]
            folds = customer_folds(y23.customer_id.to_numpy()[pos], seed)
            cv = {}
            for d in DECAYS:
                dev = 0.0
                for k in range(5):
                    tr, te = folds != k, folds == k
                    nv = refit_leaves(leaves_d[tr], values, init_d[tr], yd[tr], d, LGBM["learning_rate"], LGBM["lambda_l2"])
                    mu = np.exp(margin(leaves_d[te], nv, init_d[te]))
                    dev += unit_dev(yd[te], mu * opt_scale(yd[te], mu)).sum()  # level-matched, as the metric
                cv[d] = dev
            best = min(cv, key=cv.get)
            for d in DECAYS:  # every decay is kept as a control; the cross-validated one is the arm
                nvd = refit_leaves(leaves_d, values, init_d, yd, d, LGBM["learning_rate"], LGBM["lambda_l2"])
                pred[f"A2_d{d}_draw{seed}"] = np.exp(margin(lv["y24"], nvd, init["y24"])) + large24
            pred[f"A2_draw{seed}"] = pred[f"A2_d{best}_draw{seed}"]
            chosen[seed] = best
            print(f"  {name} draw {seed}: decay {best} (cv {', '.join(f'{d}: {v:.1f}' for d, v in cv.items())})", flush=True)
        pd.DataFrame({"row_id": y24.row_id.to_numpy(), **pred}).to_parquet(out / f"pred_lgbm_{name}.parquet", index=False)
        res[name] = {"trees": n_trees, "decay_chosen": chosen}
        print(name, res[name], flush=True)
    (out / "lgbm_meta.json").write_text(json.dumps(res, indent=1))


# --- A3: TabPFN-3.5 occurrence stacker ------------------------------------------------------------------------------

def tabpfn_weights() -> Path:
    """The approved local weights file, or stop: nothing may be downloaded or sent anywhere (amendment 6)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"
    folder = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    path = Path(folder) / TABPFN_WEIGHTS if folder else None
    if path is None or not path.is_file():
        raise SystemExit(f"TabPFN weights not found ({path}); set TABPFN_MODEL_CACHE_DIR to the folder holding {TABPFN_WEIGHTS}")
    return path


def tabpfn_fit(design_ctx, occ, device: str, seed: int, weights: Path, precision="auto"):
    from tabpfn import TabPFNClassifier  # noqa: PLC0415
    kw = dict(device=device, model_path=str(weights), n_estimators=TABPFN_ESTIMATORS, random_state=seed,
              inference_precision=precision)
    try:
        clf = TabPFNClassifier(**kw).fit(design_ctx, occ)
        ignored = False
    except ValueError as exc:  # the checkpoint's sample limit below the draw size (amendment 6)
        if "ignore_pretraining_limits" not in str(exc):
            raise
        clf = TabPFNClassifier(**kw, ignore_pretraining_limits=True).fit(design_ctx, occ)
        ignored = True
    assert clf.n_estimators_ == TABPFN_ESTIMATORS and list(clf.classes_) == [0, 1]
    info = {"n_estimators": int(clf.n_estimators_), "use_autocast": bool(clf.use_autocast_),
            "devices": [str(d) for d in clf.devices_], "ignore_pretraining_limits": ignored,
            "max_number_of_samples": int(clf.inference_config_.MAX_NUMBER_OF_SAMPLES)}
    return clf, info


def proba(clf, x) -> np.ndarray:
    return np.concatenate([clf.predict_proba(x.iloc[i:i + CHUNK])[:, 1] for i in range(0, len(x), CHUNK)])


def tabpfn(out: Path, device: str = "mps", smoke: bool = False) -> None:
    weights = tabpfn_weights()
    import tabpfn as tp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from importlib.metadata import version  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    res = {"weights": weights.name, "weights_sha256": sha256(weights), "tabpfn": version("tabpfn"), "torch": torch.__version__,
           "device": device, "lines": {}}
    del tp
    for name in LINES:
        feats, fr, draws, _ = load(out, name)
        cols = feats["numeric"] + feats["categorical"]
        y23, y24 = fr["y23"], fr["y24"]

        def design(f):
            x = f[cols].copy()
            x["log_c0_rate"] = np.log(f.c0_total_2023 / f.exposure)  # one trend year for context and test (amendment 5)
            x["exposure"] = f.exposure.to_numpy()
            return x

        def z(f):
            return np.c_[np.log(f.c0_total_2023 / f.exposure), np.log(f.exposure)]

        if smoke:  # engineering check: Apple GPU against CPU on 2,000 context and 2,000 scored rows
            seed, rows = next(iter(draws.items()))
            dr = y23.iloc[rows].sample(n=2000, random_state=seed)
            occ = (dr.n_claims > 0).astype(int).to_numpy()
            xt = design(y24.iloc[:2000])
            p = {}
            for dev in ("mps", "cpu"):
                for prec in ("float32", "auto"):
                    clf, info = tabpfn_fit(design(dr), occ, dev, seed, weights, torch.float32 if prec == "float32" else "auto")
                    p[(dev, prec)] = proba(clf, xt)
                    res["lines"].setdefault(name, {})[f"{dev}_{prec}"] = info
            rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))  # noqa: E731
            res["lines"][name]["max_rel_diff_float32_mps_vs_cpu"] = rel(p[("mps", "float32")], p[("cpu", "float32")])
            res["lines"][name]["max_rel_diff_default_mps_vs_cpu_float32"] = rel(p[("mps", "auto")], p[("cpu", "float32")])
            print(name, {k: v for k, v in res["lines"][name].items() if k.startswith("max_rel")}, flush=True)
            continue
        x24, z24 = design(y24), z(y24)
        small24, large24 = y24.c0_small.to_numpy(), (y24.c0_total - y24.c0_small).to_numpy()
        pred, infos = {}, {}
        for seed, rows in draws.items():
            t0 = time.time()
            dr = y23.iloc[rows]
            occ = (dr.n_claims > 0).astype(int).to_numpy()
            clf, info = tabpfn_fit(design(dr), occ, device, seed, weights)
            p = proba(clf, x24)
            p0 = LogisticRegression(C=1e6, max_iter=1000).fit(z(dr), occ).predict_proba(z24)[:, 1]
            ratio = np.clip(p, 1e-6, 1) / np.clip(p0, 1e-6, 1)
            for w in WEIGHTS:  # the ratio acts on C0's small-claim part only (amendment 4)
                pred[f"A3_w{w}_draw{seed}"] = small24 * ratio ** w + large24
            infos[seed] = info | {"seconds": round(time.time() - t0, 1), "mean_p": float(p.mean()), "mean_p0": float(p0.mean())}
            print(f"  {name} draw {seed}: {infos[seed]}; ratio 1%-99% {np.quantile(ratio, 0.01):.2f}-{np.quantile(ratio, 0.99):.2f}",
                  flush=True)
        pd.DataFrame({"row_id": y24.row_id.to_numpy(), **pred}).to_parquet(out / f"pred_tabpfn_{name}.parquet", index=False)
        res["lines"][name] = infos
    (out / ("tabpfn_smoke.json" if smoke else "tabpfn_meta.json")).write_text(json.dumps(res, indent=1, default=str))


# --- score ------------------------------------------------------------------------------------------------------------

def score(out: Path, n_boot: int = 2000, seed: int = 2032) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common  # noqa: PLC0415

    t0 = time.time()
    y, e, cust, c0, nb, young, pr = [], [], [], [], [], [], {}
    for name in LINES:
        f = pd.read_parquet(out / f"y24_{name}.parquet")
        o = pd.read_parquet(out / f"y24out_{name}.parquet")
        assert (o.row_id.to_numpy() == f.row_id.to_numpy()).all()
        preds = f[["row_id"]]
        for arm in ("lgbm", "tabpfn"):
            p = pd.read_parquet(out / f"pred_{arm}_{name}.parquet")  # both families are required (amendment 8)
            assert (p.row_id.to_numpy() == f.row_id.to_numpy()).all()
            preds = preds.join(p.drop(columns="row_id"))
        pr[name] = preds
        y.append(o.y_full.to_numpy(float)), e.append(f.exposure_plain.to_numpy(float))
        cust.append(o.customer_id.astype(str).to_numpy())  # one key per customer across lines (the bootstrap cluster)
        c0.append(f.c0_total.to_numpy(float)), nb.append(o.new_business.to_numpy(bool))
        young.append(o.age_band.eq("17-24").fillna(False).to_numpy(bool))
    line_of = np.concatenate([np.full(len(v), i) for i, v in enumerate(y)])
    Y, E, NB, YOUNG, CUST = map(np.concatenate, (y, e, nb, young, cust))
    null = unit_dev(Y, Y.sum() / E.sum() * E)
    cols = [c for c in pr["buildings"].columns if c != "row_id"]
    mu = {c: np.concatenate([pr[n][c].to_numpy(float) for n in LINES]) for c in cols} | {"C0": np.concatenate(c0)}
    families = {"A2": [c for c in cols if c.startswith("A2_draw")], "A3": [c for c in cols if c.startswith("A3_w0.5_draw")]}
    assert all(len(v) == len(DRAW_SEEDS) for v in families.values()), families

    def lm_dev(m_, mask):
        """Row deviances on `mask` with each line at its own deviance-optimal level on those rows."""
        d = np.zeros(len(Y))
        for i in range(len(LINES)):
            k = mask & (line_of == i)
            d[k] = unit_dev(Y[k], m_[k] * opt_scale(Y[k], m_[k]))
        return d

    every = np.ones(len(Y), bool)
    keep = every.copy()
    keep[np.argsort(-Y)[:10]] = False
    dev_all = {c: lm_dev(v, every) for c, v in mu.items()}
    dev_keep = {c: lm_dev(v, keep) for c, v in mu.items()}  # levels re-solved without the ten largest (amendment 8)

    def gain(cs, mask=every, dev=dev_all):
        return float(np.mean([(dev["C0"][mask].sum() - dev[c][mask].sum()) / null[mask].sum() for c in cs]))

    def line_d2(c, i):  # against the line's own constant rate (amendment 8)
        k = line_of == i
        return 1 - dev_all[c][k].sum() / unit_dev(Y[k], Y[k].sum() / E[k].sum() * E[k]).sum()

    raw = {c: 1 - unit_dev(Y, v).sum() / null.sum() for c, v in mu.items()}
    rows = [{"arm": c, "pooled_d2_level_matched": 1 - dev_all[c].sum() / null.sum(), "pooled_d2_raw": raw[c],
             "pooled_gain_level_matched": gain([c]) if c != "C0" else 0.0,
             **{f"{n}_d2_level_matched": line_d2(c, i) for i, n in enumerate(LINES)}} for c in ["C0"] + cols]
    codes, uniq = pd.factorize(CUST)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(n_boot)])
    g = lambda v: np.bincount(codes, weights=v, minlength=len(uniq))  # noqa: E731
    gn, gc = g(null), g(dev_all["C0"])
    results = {}
    for fam, cs in families.items():
        gd = np.mean([g(dev_all[c]) for c in cs], axis=0)
        boot = ((gc - gd) * W).sum(1) / (gn * W).sum(1)
        results[fam] = {"gain": gain(cs), "p_one_sided": float((boot <= 0).mean()), "ci_low_95": float(np.quantile(boot, 0.025)),
                        "ci_high_95": float(np.quantile(boot, 0.975)), "gain_without_top10": gain(cs, keep, dev_keep),
                        "gain_new_business": gain(cs, NB), "gain_age_17_24": gain(cs, YOUNG),
                        "raw_d2_mean": float(np.mean([raw[c] for c in cs])), "draw_gains": [gain([c]) for c in cs],
                        "per_line_d2_gain": {n: float(np.mean([line_d2(c, i) - line_d2("C0", i) for c in cs]))
                                             for i, n in enumerate(LINES)}}
    passed = True
    for rank, fam in enumerate(sorted(results, key=lambda k: results[k]["p_one_sided"])):
        r = results[fam]
        r["holm_threshold"] = 0.025 / (FAMILIES_M - rank)
        passed = passed and r["p_one_sided"] < r["holm_threshold"]
        r["passes_holm"] = passed
        r["passes_gate"] = bool(passed and r["gain"] >= 0.003 and r["gain_without_top10"] > 0
                                and r["gain_new_business"] >= 0 and r["gain_age_17_24"] >= 0)
    cfg = common.load_config()
    rec = common.new_run_record("challengers_score", cfg) | {"protocol_sha256": sha256(PROTOCOL)}
    ev = common.PACK_ROOT / "audit/evidence/challengers"
    ev.mkdir(parents=True, exist_ok=True)
    stamp = {"git_commit": rec["git_commit"], "protocol_sha256": rec["protocol_sha256"]}
    pd.DataFrame(rows).assign(**stamp).to_csv(ev / "ev_a32_arms.csv", index=False)
    pd.DataFrame([{"family": k, **{kk: (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()}, **stamp}
                  for k, v in results.items()]).to_csv(ev / "ev_a32_stage1.csv", index=False)
    rec |= {"status": "success", "metrics": results, "c0_pooled_d2_level_matched": rows[0]["pooled_d2_level_matched"],
            "outputs_sha256": {p.name: sha256(p) for p in sorted(ev.glob("ev_a32_*.csv"))}, "elapsed_s": round(time.time() - t0, 1)}
    common.append_run_log(cfg, rec)
    print(f"C0 pooled level-matched D2 on the scored rows: {rows[0]['pooled_d2_level_matched']:.5f}")
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    step, folder = sys.argv[1], Path(sys.argv[2])
    folder.mkdir(parents=True, exist_ok=True)
    if step == "prepare":
        prepare(folder)
    elif step == "lgbm":
        lgbm(folder)
    elif step == "tabpfn":
        tabpfn(folder, device=next((a for a in sys.argv[3:] if not a.startswith("--")), "mps"), smoke="--smoke" in sys.argv)
    elif step == "score":
        score(folder)
    else:
        raise SystemExit(f"unknown step {step}")
