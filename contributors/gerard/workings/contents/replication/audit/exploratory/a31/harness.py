"""Shared scoring for the A31 model variations (scratch, not committed).

Every variation is scored the same way, on the 2024 emerging explore half (split == "A2024"), having been fitted on
emerging 2023 (split == "fit2023") and, if it wants, the developed market (dev_<line>.parquet, 2021-24). The 2024
half B is held back for confirmation and is not in these files: do not look for it.

The leaderboard scores one file over all three lines with Tweedie D2 (p = 1.5) against a constant-rate baseline,
pooled. `score` reports that pooled D2 twice: raw, and with each line's predictions rescaled to their deviance-
optimal level on the scored rows ("level-matched"), which isolates risk ranking from the level. The 2025 level is
set separately, so level-matched pooled D2 is the headline for a ranking change. The paired customer bootstrap
compares a variation with the portable developed model as it is (C0), level-matched on both sides.

Usage:
    import sys; sys.path.insert(0, "<this folder>")
    import harness as h
    emg = h.load("motor")                     # fit2023, A2024, test2025 rows
    preds = {"buildings": ..., "contents": ..., "motor": ...}   # arrays aligned to h.load(line).query("split == 'A2024'")
    print(h.report(preds, "my variation"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
P = 1.5
LINES = ("buildings", "contents", "motor")
META = json.loads((CACHE / "meta.json").read_text())


def load(line: str) -> pd.DataFrame:
    """Emerging rows: split in {fit2023, A2024, test2025}. y = y_full (developed target, motor on the developed
    reporting pattern), exposure = exposure_plain. c0 = the portable developed model for the row's own year."""
    return pd.read_parquet(CACHE / f"emg_{line}.parquet")


def load_dev(line: str) -> pd.DataFrame:
    """Developed rows 2021-24 (portable features plus raw fields); c0_in_sample = the portable model, in sample."""
    return pd.read_parquet(CACHE / f"dev_{line}.parquet")


def unit_dev(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return 2 * (np.power(y, 2 - P) / ((1 - P) * (2 - P)) - y * np.power(mu, 1 - P) / (1 - P)
                + np.power(mu, 2 - P) / (2 - P))


def opt_scale(y, mu) -> float:
    """Deviance-optimal scale k for predictions mu: sum(y mu^(1-p)) / sum(mu^(2-p))."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return float((y * mu ** (1 - P)).sum() / (mu ** (2 - P)).sum())


def _scored():
    out = {}
    for ln in LINES:
        d = load(ln)
        a = d[d.split == "A2024"]
        out[ln] = {"y": a.y_full.to_numpy(float), "e": a.exposure_plain.to_numpy(float), "c0": a.c0.to_numpy(float),
                   "cust": a.customer_id.astype(str).to_numpy(), "nb": a.new_business.astype(bool).to_numpy()}
    return out


_S = None


def scored():
    global _S
    if _S is None:
        _S = _scored()
    return _S


def score(preds: dict) -> dict:
    """Pooled and per-line D2 on the A2024 rows for {line: predictions}; lines missing from preds use C0."""
    s = scored()
    y = np.concatenate([s[ln]["y"] for ln in LINES])
    e = np.concatenate([s[ln]["e"] for ln in LINES])
    null = unit_dev(y, y.sum() / e.sum() * e).sum()
    out, raw, lm = {}, [], []
    for ln in LINES:
        mu = np.asarray(preds.get(ln, s[ln]["c0"]), float)
        assert mu.shape == s[ln]["y"].shape and np.all(mu > 0), f"{ln}: predictions must align to the A2024 rows"
        k = opt_scale(s[ln]["y"], mu)
        raw.append(unit_dev(s[ln]["y"], mu))
        lm.append(unit_dev(s[ln]["y"], mu * k))
        yl, el = s[ln]["y"], s[ln]["e"]
        nl = unit_dev(yl, yl.sum() / el.sum() * el).sum()
        out[f"{ln}_d2_level_matched"] = 1 - lm[-1].sum() / nl
        out[f"{ln}_scale"] = k
    out["pooled_d2_raw"] = 1 - np.concatenate(raw).sum() / null
    out["pooled_d2_level_matched"] = 1 - np.concatenate(lm).sum() / null
    return out


def bootstrap_gain(preds: dict, n: int = 1000, seed: int = 2031) -> dict:
    """Paired customer bootstrap of the level-matched pooled D2 gain over C0 (scales fixed at the full-sample ones)."""
    s = scored()
    cust = np.concatenate([s[ln]["cust"] for ln in LINES])
    y = np.concatenate([s[ln]["y"] for ln in LINES])
    e = np.concatenate([s[ln]["e"] for ln in LINES])
    d_alt, d_c0 = [], []
    for ln in LINES:
        mu = np.asarray(preds.get(ln, s[ln]["c0"]), float)
        d_alt.append(unit_dev(s[ln]["y"], mu * opt_scale(s[ln]["y"], mu)))
        d_c0.append(unit_dev(s[ln]["y"], s[ln]["c0"] * opt_scale(s[ln]["y"], s[ln]["c0"])))
    d_alt, d_c0 = np.concatenate(d_alt), np.concatenate(d_c0)
    null = unit_dev(y, y.sum() / e.sum() * e)
    codes, uniq = pd.factorize(cust)
    g = lambda v: np.bincount(codes, weights=v, minlength=len(uniq))  # noqa: E731
    ga, gc, gn = g(d_alt), g(d_c0), g(null)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        gains.append(((gc - ga) * w).sum() / (gn * w).sum())
    gains = np.array(gains)
    return {"gain": float((gc - ga).sum() / gn.sum()), "ci_low_95": float(np.quantile(gains, 0.025)),
            "ci_high_95": float(np.quantile(gains, 0.975)), "share_not_better": float((gains <= 0).mean())}


def report(preds: dict, label: str) -> str:
    sc, bg, c0 = score(preds), bootstrap_gain(preds), score({})
    lines = [f"{label}: pooled level-matched D2 {sc['pooled_d2_level_matched']:.5f} (C0 {c0['pooled_d2_level_matched']:.5f}); "
             f"gain {bg['gain']:+.5f} [95% {bg['ci_low_95']:+.5f}, {bg['ci_high_95']:+.5f}], share not better {bg['share_not_better']:.3f}; "
             f"raw pooled {sc['pooled_d2_raw']:.5f} (C0 {c0['pooled_d2_raw']:.5f})"]
    lines.append("  per line level-matched: " + ", ".join(
        f"{ln} {sc[f'{ln}_d2_level_matched']:.5f} (C0 {c0[f'{ln}_d2_level_matched']:.5f})" for ln in LINES))
    return "\n".join(lines)
