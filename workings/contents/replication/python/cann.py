"""CANN correction for a line's small-claim model (A20): the client side, free of PyTorch.

The networks are trained and evaluated in a separate worker process (python/cann_torch.py run as a script), because
PyTorch and XGBoost cannot share a process on macOS (each loads its own OpenMP runtime; A20). The worker is a plain
subprocess rather than a multiprocessing child: a spawned child re-imports the calling script, which brought
XGBoost's OpenMP runtime into the worker after all and crashed it (A20). Only the feature columns, target and
offset are sent across; the trained ensembles stay in the worker, keyed by an id held here. If the worker dies,
the next call raises at once instead of waiting for a reply that will never come.
"""
from __future__ import annotations

import atexit
import pickle
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

WORKER = Path(__file__).resolve().parent / "cann_torch.py"
_PROC: subprocess.Popen | None = None


def _worker() -> subprocess.Popen:
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        _PROC = subprocess.Popen([sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        atexit.register(_stop)
    return _PROC


def _stop() -> None:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        _PROC.stdin.close()
        _PROC.wait(timeout=60)
    _PROC = None


def _call(name: str, *args):
    proc = _worker()
    try:
        pickle.dump((name, args), proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        proc.stdin.flush()
        ok, value = pickle.load(proc.stdout)
    except (EOFError, BrokenPipeError) as exc:
        code = proc.wait()
        raise RuntimeError(f"CANN worker exited with code {code} during {name}") from exc
    if not ok:
        raise RuntimeError(f"CANN worker failed in {name}:\n{value}")
    return value


class CANNEnsemble:
    """Mean log correction of `n_seeds` networks trained on top of a fixed offset, rebalanced on the fitted rows."""

    def __init__(self, numeric: list[str], categorical: list[str], spec: dict, p: float):
        self.numeric, self.categorical, self.spec, self.p = numeric, categorical, spec, p
        self.key = uuid.uuid4().hex

    def _cols(self, d: pd.DataFrame) -> pd.DataFrame:
        return d[self.numeric + self.categorical].reset_index(drop=True)

    def fit(self, d: pd.DataFrame, y: np.ndarray, offset_pred: np.ndarray) -> "CANNEnsemble":
        _call("fit_remote", self.key, self.numeric, self.categorical, self.spec, self.p, self._cols(d),
              np.asarray(y, dtype=float), np.asarray(offset_pred, dtype=float))
        return self

    def correction(self, d: pd.DataFrame) -> np.ndarray:
        return _call("correction_remote", self.key, self._cols(d))


def self_check() -> dict:
    return _call("self_check")


def worker_modules() -> list[str]:
    """Names of the extension libraries loaded in the worker (for the OpenMP isolation test)."""
    return _call("loaded_modules")
