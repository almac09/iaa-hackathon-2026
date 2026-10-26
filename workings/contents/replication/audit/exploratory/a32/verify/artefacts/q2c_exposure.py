"""Post hoc: occurrence per exposure by exposure band in the source, the draw pool, the rest of 2023 and the scored 2024
rows; and the A3 ratio by new business."""
import json

import numpy as np
import pandas as pd

from vhelp import LINES, S, SEEDS, scored

sc = scored()
tp = {n: pd.read_parquet(S / f"pred_tabpfn_{n}.parquet") for n in LINES}
BANDS = [0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.01]
for n in LINES:
    src = pd.read_parquet(S / f"src_{n}.parquet", columns=["exposure_plain", "n_claims", "y_small", "customer_id"])
    y23 = pd.read_parquet(S / f"y23_{n}.parquet", columns=["exposure_plain", "n_claims", "y_small", "customer_id", "c0_small"])
    draws = {int(k): v for k, v in json.loads((S / f"draws_{n}.json").read_text()).items()}
    drawn = np.zeros(len(y23), bool)
    for v in draws.values():
        drawn[np.asarray(v)] = True
    pool = ~y23.customer_id.isin(set(sc[n].customer_id)).to_numpy()
    d24 = sc[n]
    groups = {"src 21-22": src, "2023 drawn": y23[drawn], "2023 pool": y23[pool], "2023 not pool": y23[~pool], "2024 scored": d24}
    print(f"== {n}: claims per unit exposure (occurrence rows / exposure) by exposure band {BANDS[1:]}; row share in brackets")
    for lab, g in groups.items():
        b = pd.cut(g.exposure_plain, BANDS, labels=False)
        t = pd.DataFrame({"b": b, "o": (g.n_claims > 0).astype(int), "e": g.exposure_plain}).groupby("b").agg(o=("o", "sum"), e=("e", "sum"), n=("o", "size"))
        rate = (t.o / t.e).round(3).tolist()
        share = (t.n / t.n.sum()).round(2).tolist()
        print(f"  {lab:14s} rate {rate}  share {share}")
    nb = d24.new_business.to_numpy(bool)
    small, large = d24.c0_small.to_numpy(float), (d24.c0_total - d24.c0_small).to_numpy(float)
    lr = [np.log((tp[n][f"A3_w1.0_draw{s}"].to_numpy(float) - large) / small) for s in SEEDS]
    m_nb = np.mean([np.exp(x[nb].mean()) for x in lr])
    m_rn = np.mean([np.exp(x[~nb].mean()) for x in lr])
    ys = d24.y_small.to_numpy(float)
    print(f"  scored NB: geometric-mean ratio NB {m_nb:.3f} vs renewal {m_rn:.3f};  small A/E vs C0 NB {ys[nb].sum() / small[nb].sum():.3f}"
          f" vs renewal {ys[~nb].sum() / small[~nb].sum():.3f}; NB part-year share {(d24.exposure_plain[nb] < 0.99).mean():.3f}"
          f" vs renewal {(d24.exposure_plain[~nb] < 0.99).mean():.3f}")
