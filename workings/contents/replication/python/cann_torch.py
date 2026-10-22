"""PyTorch side of the CANN correction (A20). Runs ONLY as the worker script started by cann.py.

On macOS, PyTorch and XGBoost each ship their own OpenMP runtime (libomp.dylib); loading both in one process made
XGBoost segfault or deadlock in A20. The pipeline therefore never imports torch itself: cann.py starts this file as
a separate process, sends it feature data over a pipe, and it trains the ensembles on the Apple GPU (MPS) and returns
corrections.

Method: the incumbent model's log prediction is a fixed offset and the network's output layer starts at zero, so
training begins exactly at the incumbent (Wuthrich & Merz 2019). Categoricals enter as entity embeddings. The
correction is the mean over networks that differ only in their seed (the nagging predictor of Richman & Wuthrich),
rebalanced so the fitted rows keep the incumbent's total.

GPU use: the seeds are not trained one after another. All of them are held as one batched network (weights stacked
along a leading seed axis, applied with batched matrix products), so each optimiser step trains every seed on its
own mini-batch in a single pass. Seeds stay independent: each has its own initialisation, batch order, gradient
clipping and early stopping, exactly as if trained alone.
"""
from __future__ import annotations

import os
import pickle
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

_ENSEMBLES: dict[str, "Ensemble"] = {}
EVAL_CHUNK = 65536


def device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def tweedie_loss(y: torch.Tensor, log_mu: torch.Tensor, p: float) -> torch.Tensor:
    """Mean Tweedie deviance over the last axis, up to terms that do not depend on mu (1 < p < 2)."""
    return (-y * torch.exp((1 - p) * log_mu) / (1 - p) + torch.exp((2 - p) * log_mu) / (2 - p)).mean(-1)


def _emb_dim(k: int) -> int:
    return min(16, int(np.ceil(np.sqrt(k))) + 1)


class Encoder:
    """Standardised numerics with missing flags, and categorical codes (0 = unseen or missing), fixed on the
    training rows."""

    def __init__(self, train: pd.DataFrame, numeric: list[str], categorical: list[str]):
        self.numeric, self.categorical = numeric, categorical
        x = train[numeric].to_numpy(dtype=float)
        self.mean, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
        self.levels = {c: sorted(train[c].astype("string").dropna().unique()) for c in categorical}

    def encode(self, df: pd.DataFrame, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        x = df[self.numeric].to_numpy(dtype=float)
        miss = np.isnan(x)
        num = np.c_[np.where(miss, 0.0, (x - self.mean) / self.sd), miss]
        def code(c):
            s = df[c].astype("string")
            return pd.Categorical(s.where(s.isin(self.levels[c])), categories=self.levels[c]).codes + 1
        codes = (np.stack([code(c) for c in self.categorical], 1) if self.categorical
                 else np.zeros((len(df), 0)))
        return (torch.tensor(num, dtype=torch.float32, device=dev), torch.tensor(codes, dtype=torch.long, device=dev))


class BatchedNet(nn.Module):
    """`n` networks of identical shape (embeddings, SiLU hidden layers with dropout, zero-initialised head), held as
    stacked weights so that all of them run in one batched pass. Initialisation matches nn.Embedding/nn.Linear."""

    def __init__(self, n: int, n_num: int, cardinalities: list[int], hidden: list[int], dropout: float):
        super().__init__()
        self.n, self.dropout = n, dropout
        self.emb = nn.ParameterList([nn.Parameter(torch.randn(n, k + 1, _emb_dim(k))) for k in cardinalities])
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        width = n_num + sum(_emb_dim(k) for k in cardinalities)
        for h in hidden + [1]:
            bound = 1 / np.sqrt(width)
            self.weights.append(nn.Parameter(torch.empty(n, width, h).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n, 1, h).uniform_(-bound, bound)))
            width = h
        nn.init.zeros_(self.weights[-1])
        nn.init.zeros_(self.biases[-1])  # the correction starts at exactly zero

    def forward(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        """xn [n, B, numeric], xc [n, B, categorical] -> log correction [n, B]."""
        seeds = torch.arange(self.n, device=xn.device)[:, None]
        h = torch.cat([xn] + [e[seeds, xc[:, :, j]] for j, e in enumerate(self.emb)], 2)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = F.dropout(F.silu(torch.baddbmm(b, h, w)), self.dropout, self.training)
        return torch.baddbmm(self.biases[-1], h, self.weights[-1]).squeeze(2)

    def shared(self, xn: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        """Every network on the same rows, in chunks: [n, rows]."""
        return torch.cat([self(xn[None, i:i + EVAL_CHUNK].expand(self.n, -1, -1),
                               xc[None, i:i + EVAL_CHUNK].expand(self.n, -1, -1))
                          for i in range(0, len(xn), EVAL_CHUNK)], 1)


def _clip_per_seed(params, max_norm: float) -> None:
    """clip_grad_norm_ applied to each seed's own gradient, so one seed's step never rescales another's."""
    sq = sum(q.grad.pow(2).flatten(1).sum(1) for q in params)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for q in params:
        q.grad.mul_(scale.view(-1, *[1] * (q.grad.dim() - 1)))


def _train(xn, xc, y, offset, val_mask, p, spec, dev) -> BatchedNet:
    n, bs = spec["n_seeds"], spec["batch_size"]
    torch.manual_seed(spec["seed"])
    net = BatchedNet(n, xn.shape[1], [len(v) for v in spec["_levels"]], spec["hidden"], spec["dropout"]).to(dev)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=spec["learning_rate"])
    train_idx, val_idx = torch.nonzero(~val_mask).squeeze(1), torch.nonzero(val_mask).squeeze(1)
    gen = torch.Generator(device="cpu").manual_seed(spec["seed"])
    best = torch.full((n,), np.inf, device=dev)
    stale = torch.zeros(n, dtype=torch.long, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    best_state = [q.detach().clone() for q in params]
    for _ in range(spec["max_epochs"]):
        net.train()
        order = train_idx[torch.argsort(torch.rand(n, len(train_idx), generator=gen), 1).to(dev)]  # own order per seed
        for i in range(0, order.shape[1], bs):
            b = order[:, i:i + bs]
            opt.zero_grad(set_to_none=True)
            tweedie_loss(y[b], offset[b] + net(xn[b], xc[b]), p).sum().backward()  # seeds' losses are separable
            _clip_per_seed(params, 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            loss = tweedie_loss(y[val_idx], offset[val_idx] + net.shared(xn[val_idx], xc[val_idx]), p)
            improved = active & (loss < best - 1e-7)
            best = torch.where(improved, loss, best)
            for kept, q in zip(best_state, params):
                kept[improved] = q.detach()[improved]
            stale = torch.where(improved, 0, stale + 1)
            active &= stale < spec["patience"]  # a stopped seed keeps its best state; later epochs are ignored
        if not bool(active.any()):
            break
    with torch.no_grad():
        for kept, q in zip(best_state, params):
            q.copy_(kept)
    return net


class Ensemble:
    """Mean log correction of `n_seeds` networks trained on top of a fixed offset, rebalanced on the fitted rows."""

    def __init__(self, numeric: list[str], categorical: list[str], spec: dict, p: float):
        self.numeric, self.categorical, self.spec, self.p = numeric, categorical, spec, p

    def fit(self, d: pd.DataFrame, y: np.ndarray, offset_pred: np.ndarray) -> "Ensemble":
        dev = device()
        self.encoder = Encoder(d, self.numeric, self.categorical)
        spec = {**self.spec, "_levels": [self.encoder.levels[c] for c in self.categorical]}
        xn, xc = self.encoder.encode(d, dev)
        yt = torch.tensor(y, dtype=torch.float32, device=dev)
        off = torch.tensor(np.log(offset_pred), dtype=torch.float32, device=dev)
        val_mask = torch.tensor(np.random.default_rng(spec["seed"]).random(len(d)) < spec["val_share"], device=dev)
        self.net = _train(xn, xc, yt, off, val_mask, self.p, spec, dev)
        self.rebalance = 1.0
        corrected = offset_pred * np.exp(self.correction(d))
        self.rebalance = offset_pred.sum() / corrected.sum()
        return self

    def correction(self, d: pd.DataFrame) -> np.ndarray:
        """Mean log correction over the ensemble (log of the rebalancing factor included)."""
        xn, xc = self.encoder.encode(d, device())
        self.net.eval()
        with torch.no_grad():
            out = self.net.shared(xn, xc).mean(0).cpu().numpy().astype(float)
        return out + np.log(self.rebalance)


# Worker entry points (called through cann.py).
def fit_remote(key: str, numeric, categorical, spec, p, d, y, offset_pred) -> None:
    _ENSEMBLES[key] = Ensemble(numeric, categorical, spec, p).fit(d, y, offset_pred)


def correction_remote(key: str, d: pd.DataFrame) -> np.ndarray:
    return _ENSEMBLES[key].correction(d)


def release_remote(key: str) -> None:
    _ENSEMBLES.pop(key, None)


def loaded_modules() -> list[str]:
    return sorted(sys.modules)


def self_check() -> dict:
    """Facts the unit tests need from the torch side: zero-initialised correction, the loss's minimiser, unseen
    categories coded as missing, and seeds that are trained independently."""
    net = BatchedNet(2, 3, [4], [8], 0.0)
    zero = bool(torch.all(net(torch.randn(2, 5, 3), torch.ones(2, 5, 1, dtype=torch.long)) == 0))
    y = torch.tensor([0.0, 0.0, 3.0, 1.0])
    losses = {m: tweedie_loss(y, torch.full((4,), float(np.log(m))), 1.5).item() for m in (0.5, 1.0, 1.5)}
    enc = Encoder(pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]}), ["x"], ["c"])
    _, codes = enc.encode(pd.DataFrame({"x": [np.nan], "c": ["zzz"]}), torch.device("cpu"))
    torch.manual_seed(0)  # a loss on seed 0 alone must leave seed 1's gradient at zero
    net = BatchedNet(2, 3, [4], [8], 0.0)
    nn.init.normal_(net.weights[-1])
    net(torch.randn(2, 5, 3), torch.ones(2, 5, 1, dtype=torch.long))[0].sum().backward()
    separable = all(float(q.grad[1].abs().sum()) == 0 for q in net.parameters() if q.grad is not None)
    return {"zero_init": zero, "loss_argmin": min(losses, key=losses.get), "unseen_code": codes.tolist()[0][0],
            "seeds_separable": separable, "device": str(device())}


ENTRY_POINTS = {f.__name__: f for f in (fit_remote, correction_remote, release_remote, loaded_modules, self_check)}


def serve() -> None:
    """Answer (name, args) requests on stdin with (ok, value) replies on stdout until stdin closes."""
    reply_stream = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything a library prints goes to stderr, keeping the reply stream clean
    sys.stdout = sys.stderr
    torch.set_num_threads(4)
    while True:
        try:
            name, args = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            reply = (True, ENTRY_POINTS[name](*args))
        except Exception:
            reply = (False, traceback.format_exc())
        pickle.dump(reply, reply_stream, protocol=pickle.HIGHEST_PROTOCOL)
        reply_stream.flush()


if __name__ == "__main__":
    serve()
