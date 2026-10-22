"""Paired cluster bootstrap of a D2 difference (audit A20).

Both predictions are scored on the same resample each time, and whole clusters (customers) are resampled, so
within-customer correlation does not narrow the interval. Resamples are drawn as cluster counts and applied as row
weights, which keeps 1,000 resamples of ~200,000 rows fast without copying data.
"""
from __future__ import annotations

import numpy as np

import models


def _weighted_d2(y, mu, w, p):
    ybar = (w * y).sum(1) / w.sum(1)
    null = (w * models.tweedie_deviance(y[None, :], ybar[:, None], p)).sum(1)
    return 1 - (w * models.tweedie_deviance(y, mu, p)[None, :]).sum(1) / null


def paired_cluster_bootstrap(y, mu_a, mu_b, clusters, p, n_boot, seed, batch=50) -> np.ndarray:
    """D2(a) - D2(b) over `n_boot` resamples of whole clusters."""
    codes, n_clusters = np.unique(clusters, return_inverse=True)[1], len(np.unique(clusters))
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, batch):
        k = min(batch, n_boot - start)
        counts = np.stack([np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
                           for _ in range(k)])
        w = counts[:, codes].astype(float)
        out.append(_weighted_d2(y, mu_a, w, p) - _weighted_d2(y, mu_b, w, p))
    return np.concatenate(out)


def summarise(diff: np.ndarray, observed: float) -> dict:
    return {"d2_gain": observed, "boot_mean": diff.mean(), "ci_low_95": np.percentile(diff, 2.5),
            "ci_high_95": np.percentile(diff, 97.5), "share_not_better": (diff <= 0).mean(), "n_boot": len(diff),
            "significant_95": bool(np.percentile(diff, 2.5) > 0)}
