"""A31 variation v1: written_premium as a labelled diagnostic for owner decision D-05 (scratch). D-05 keeps
written_premium out of the technical model; nothing here is to be adopted without the owner.

Signal. x = log(written_premium / annual C0 rate), the annual C0 rate being c0 / exposure_plain. written_premium is the
annual premium whatever the row's period in both markets (mean log premium is flat across exposure bands), so x
compares like with like. x is centred within each data set by year x channel x new_business cell (DEMEAN_CELLS), so
the incumbent's commercial differentials (new-business and comparison-site discounts, present in both markets) and the
year-on-year drift between premium inflation and C0's trend are not read as risk; the 2025 book is 77% new business
against 8% in training, so this matters for the transfer. x is then clipped to the training quantiles CLIP_Q.

Forms (per line, CONFIG):
  none      : C0 unchanged
  linear    : log(mu) = log(c0) + a_year + b * x                    (one parameter; a geometric blend weight)
  linear_nb : as linear with separate b for renewals and new business (an unpenalised NB level term is fitted
              and then dropped, so C0's new-business relativity is kept)
  pwl       : piecewise-linear in x, knots at training quantiles, ridge
  band      : quantile bands of x, ridge
  xgb       : XGBoost Tweedie residual with base_margin log(c0 * k_year), depth 2, heavily regularised
Fitting uses total claim cost (optionally capped at `cap`, applied to the full c0) with offset log(c0), unweighted:
the Tweedie p = 1.5 deviance the leaderboard scores (equal to the rate model with weights exposure^(2-p)). Year
intercepts absorb each training year's level so the slope is not confounded with the 2023/2024 A/E swing.

Level. The variation changes ranking only. On the score rows the relativity is normalised within each year x
channel x new_business cell so that every cell's total equals C0's (cells under MIN_CELL rows use the whole score
set), hence each line's mean prediction equals mean C0 and the 2025 level is left to be set separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P = 1.5
LINES = ("buildings", "contents", "motor")

# ---- frozen configuration: chosen on 2023 cross-validation grouped by customer_id (before any A2024 scoring) ----
DEMEAN_CELLS = True
MIN_CELL = 30
CLIP_Q = (0.005, 0.995)
CONFIG = {
    "buildings": {"form": "none"},
    "contents": {"form": "none"},
    "motor": {"form": "linear_nb", "cap": 50000.0, "ridge": 0.0},
}
XGB_PARAMS = {"objective": "reg:tweedie", "tweedie_variance_power": P, "eta": 0.03, "max_depth": 2,
              "subsample": 0.8, "colsample_bytree": 1.0, "nthread": 2, "seed": 2031, "verbosity": 0}


def _dev(y, mu):
    return 2 * (np.power(y, 2 - P) / ((1 - P) * (2 - P)) - y * np.power(mu, 1 - P) / (1 - P) + np.power(mu, 2 - P) / (2 - P))


def _cells(df: pd.DataFrame, demean_cells: bool = DEMEAN_CELLS) -> np.ndarray:
    yr = df.yr.astype(int).astype(str).to_numpy()
    if not demean_cells:
        return yr
    return (pd.Series(yr) + "|" + df.channel.astype(str).to_numpy() + "|"
            + df.new_business.astype(bool).astype(str).to_numpy()).to_numpy()


def _cell_mean(v: np.ndarray, w: np.ndarray, cell: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """w-weighted mean of v within cell, or within the fallback group when the cell has fewer than MIN_CELL rows."""
    g = pd.DataFrame({"vw": v * w, "w": w, "c": cell, "f": fallback})
    cm = (g.groupby("c").vw.transform("sum") / g.groupby("c").w.transform("sum")).to_numpy()
    fm = (g.groupby("f").vw.transform("sum") / g.groupby("f").w.transform("sum")).to_numpy()
    return np.where(g.groupby("c").w.transform("size").to_numpy() >= MIN_CELL, cm, fm)


def signal(df: pd.DataFrame, demean_cells: bool = DEMEAN_CELLS) -> np.ndarray:
    """x = log(written_premium / annual C0 rate), exposure-weighted-centred by year (x channel x new_business)."""
    wp = df.written_premium.to_numpy(float)
    e = df.exposure_plain.to_numpy(float)
    c0 = df.c0.to_numpy(float)
    if not (np.all(np.isfinite(wp)) and np.all(wp > 0) and np.all(e > 0) and np.all(c0 > 0)):
        raise ValueError("written_premium, exposure_plain and c0 must be present and positive on every row")
    x = np.log(wp * e / c0)
    return x - _cell_mean(x, e, _cells(df, demean_cells), df.yr.astype(int).to_numpy())


def fit_glm(X, y, offset, pen, iters: int = 60):
    """Newton-Raphson for log(mu) = offset + X beta under the Tweedie p deviance with ridge penalties `pen`."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(offset + X @ beta)
        grad = X.T @ (2 * (mu ** (2 - P) - y * mu ** (1 - P))) + 2 * pen * beta
        w = 2 * ((2 - P) * mu ** (2 - P) - (1 - P) * y * mu ** (1 - P))
        step = np.linalg.solve(X.T @ (X * w[:, None]) + np.diag(2 * pen) + 1e-9 * np.eye(len(beta)), grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _year_dummies(yr, years):
    return np.column_stack([(yr == v).astype(float) for v in years])


class LineModel:
    def __init__(self, form="linear", ridge=0.0, cap=None, demean_cells=DEMEAN_CELLS, n_knots=3, n_bands=10,
                 xgb_features="x", xgb_rounds=1, xgb_min_rows=1500.0, xgb_lambda_rows=1500.0):
        self.form, self.ridge, self.cap, self.demean_cells = form, ridge, cap, demean_cells
        self.n_knots, self.n_bands = n_knots, n_bands
        self.xgb_features, self.xgb_rounds = xgb_features, xgb_rounds
        self.xgb_min_rows, self.xgb_lambda_rows = xgb_min_rows, xgb_lambda_rows

    # -- design
    def _x(self, df):
        return np.clip(signal(df, self.demean_cells), *self.clip)

    def _basis(self, df):
        x = self._x(df)
        if self.form == "linear":
            return x[:, None]
        if self.form == "linear_nb":
            nb = df.new_business.astype(bool).to_numpy().astype(float)
            return np.column_stack([x * (1 - nb), x * nb])
        if self.form == "pwl":
            return np.column_stack([x] + [np.maximum(x - k, 0.0) for k in self.knots])
        if self.form == "band":
            b = np.searchsorted(self.knots, x, side="right")
            return np.column_stack([(b == j).astype(float) for j in range(len(self.knots) + 1)])
        raise ValueError(self.form)

    def _xgb_X(self, df):
        cols = [self._x(df)]
        if "lr" in self.xgb_features:
            lr = np.log(df.c0.to_numpy(float) / df.exposure_plain.to_numpy(float))
            cols.append(lr - pd.Series(lr).groupby(df.yr.to_numpy()).transform("mean").to_numpy())
        if "nb" in self.xgb_features:
            cols.append(df.new_business.astype(bool).to_numpy().astype(float))
        return np.column_stack(cols)

    # -- fit
    def fit(self, df):
        if self.form == "none":
            return self
        y, c0, yr = df.y_full.to_numpy(float), df.c0.to_numpy(float), df.yr.astype(int).to_numpy()
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError("training outcomes must be present and non-negative")
        if self.cap is not None:
            y = np.minimum(y, self.cap)
        self.clip = tuple(np.quantile(signal(df, self.demean_cells), CLIP_Q))
        self.years = sorted(set(yr))
        if self.form == "xgb":
            import xgboost as xgb
            k = {v: float((y[yr == v] * c0[yr == v] ** (1 - P)).sum() / (c0[yr == v] ** (2 - P)).sum()) for v in self.years}
            margin = np.log(c0 * np.array([k[v] for v in yr]))
            h0 = float(np.mean(np.exp(margin) ** (2 - P)))  # rough Tweedie hessian per row, to express regularisation in rows
            prm = dict(XGB_PARAMS, min_child_weight=self.xgb_min_rows * h0, reg_lambda=self.xgb_lambda_rows * h0)
            self.booster = xgb.train(prm, xgb.DMatrix(self._xgb_X(df), label=y, base_margin=margin),
                                     num_boost_round=self.xgb_rounds)
            return self
        x = self._x(df)
        if self.form == "pwl":
            self.knots = np.quantile(x, np.linspace(0, 1, self.n_knots + 2)[1:-1])
        elif self.form == "band":
            self.knots = np.quantile(x, np.linspace(0, 1, self.n_bands + 1)[1:-1])
        B = self._basis(df)
        free = [_year_dummies(yr, self.years)]
        if self.form == "linear_nb":
            free.append(df.new_business.astype(bool).to_numpy().astype(float)[:, None])
        F = np.column_stack(free)
        X = np.column_stack([F, B])
        pen = np.r_[np.zeros(F.shape[1]), np.full(B.shape[1], self.ridge)]
        if self.form == "linear_nb":  # ridge shrinks the NB slope towards the renewal slope, not towards zero
            X = np.column_stack([F, B.sum(axis=1), B[:, 1]])
            pen = np.r_[np.zeros(F.shape[1] + 1), self.ridge]
            beta = fit_glm(X, y, np.log(c0), pen)
            b, d = beta[F.shape[1]:]
            self.coef = np.array([b, b + d])
        else:
            beta = fit_glm(X, y, np.log(c0), pen)
            self.coef = beta[F.shape[1]:]
        return self

    # -- predict
    def relativity(self, df):
        """Multiplicative correction on c0 before normalisation."""
        if self.form == "none":
            return np.ones(len(df))
        if self.form == "xgb":
            import xgboost as xgb
            dm = xgb.DMatrix(self._xgb_X(df), base_margin=np.zeros(len(df)))
            return np.exp(self.booster.predict(dm, output_margin=True))
        return np.exp(self._basis(df) @ self.coef)

    def predict(self, df):
        c0 = df.c0.to_numpy(float)
        f = self.relativity(df)
        norm = _cell_mean(f, c0, _cells(df, self.demean_cells), np.zeros(len(df), int))
        return c0 * f / norm


def model_for(line: str) -> LineModel:
    cfg = dict(CONFIG[line])
    return LineModel(form=cfg.pop("form"), **cfg)


def predict(train: dict, score: dict, dev: dict, year: int) -> dict:
    """train/score/dev: line -> DataFrame. Returns line -> positive predictions aligned to the score rows, at C0's
    level on every year x channel x new_business cell. `dev` is not used by the frozen configuration."""
    out = {}
    for line in LINES:
        s = score[line]
        if CONFIG[line]["form"] == "none":
            out[line] = s.c0.to_numpy(float)
            continue
        if not np.all(s.yr.astype(int).to_numpy() == year):
            raise ValueError(f"{line}: score rows are not all year {year}")
        p = model_for(line).fit(train[line]).predict(s)
        if not (np.all(np.isfinite(p)) and np.all(p > 0)):
            raise ValueError(f"{line}: invalid predictions")
        out[line] = p
    return out
