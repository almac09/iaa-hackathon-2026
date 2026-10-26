"""Shared helpers for the A32 arm-construction checks (post hoc; read-only on the pack and the scratch frames)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path("<pack>")
S = Path("<scratch>/a32")
sys.path.insert(0, str(PACK / "python"))
import challengers as ch  # noqa: E402

LINES = ch.LINES
SEEDS = ch.DRAW_SEEDS
unit_dev, opt_scale = ch.unit_dev, ch.opt_scale


def scored():
    """2024 scored rows per line: C0 parts, outcomes, flags."""
    out = {}
    for n in LINES:
        f = pd.read_parquet(S / f"y24_{n}.parquet", columns=["row_id", "exposure", "exposure_plain", "c0_small",
                                                               "c0_total", "c0_total_2023"])
        o = pd.read_parquet(S / f"y24out_{n}.parquet")
        assert (o.row_id.to_numpy() == f.row_id.to_numpy()).all()
        out[n] = pd.concat([f, o.drop(columns="row_id")], axis=1)
    return out


class Scorer:
    """The protocol's pooled level-matched metric on the 300,000 scored rows (mirrors challengers.score)."""

    def __init__(self, sc: dict):
        self.Y = np.concatenate([sc[n].y_full.to_numpy(float) for n in LINES])
        self.E = np.concatenate([sc[n].exposure_plain.to_numpy(float) for n in LINES])
        self.line = np.concatenate([np.full(len(sc[n]), i) for i, n in enumerate(LINES)])
        self.NB = np.concatenate([sc[n].new_business.to_numpy(bool) for n in LINES])
        self.CUST = np.concatenate([sc[n].customer_id.astype(str).to_numpy() for n in LINES])
        self.null = unit_dev(self.Y, self.Y.sum() / self.E.sum() * self.E)
        self.c0 = np.concatenate([sc[n].c0_total.to_numpy(float) for n in LINES])
        self.dev_c0 = self.lm_dev(self.c0)
        self.codes, uniq = pd.factorize(self.CUST)
        self.ncust = len(uniq)

    def lm_dev(self, m, mask=None):
        mask = np.ones(len(self.Y), bool) if mask is None else mask
        d = np.zeros(len(self.Y))
        for i in range(len(LINES)):
            k = mask & (self.line == i)
            d[k] = unit_dev(self.Y[k], m[k] * opt_scale(self.Y[k], m[k]))
        return d

    def gain(self, m, mask=None):
        mask = np.ones(len(self.Y), bool) if mask is None else mask
        d = self.lm_dev(m)
        return float((self.dev_c0[mask].sum() - d[mask].sum()) / self.null[mask].sum())

    def d2(self, m):
        return float(1 - self.lm_dev(m).sum() / self.null.sum())

    def line_gain(self, m, i):
        k = self.line == i
        d = self.lm_dev(m)
        base = unit_dev(self.Y[k], self.Y[k].sum() / self.E[k].sum() * self.E[k]).sum()
        return float((self.dev_c0[k].sum() - d[k].sum()) / base)

    def boot(self, devs: list[np.ndarray], n_boot=1000, seed=2032):
        """Paired customer bootstrap of the mean gain over `devs` (row deviance arrays), as score() does."""
        g = lambda v: np.bincount(self.codes, weights=v, minlength=self.ncust)  # noqa: E731
        gn, gc, gd = g(self.null), g(self.dev_c0), np.mean([g(d) for d in devs], axis=0)
        rng = np.random.default_rng(seed)
        out = np.empty(n_boot)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, self.ncust, self.ncust), minlength=self.ncust)
            out[b] = ((gc - gd) * w).sum() / (gn * w).sum()
        return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)), float((out <= 0).mean())
