"""Scratch (A31): predicted public score of candidate segment rescales of the as-is file, across every solution the
six-segment model finds consistent with the three public scores (the leaderboard reviewer's family())."""
import json
import sys

import numpy as np
import pandas as pd

LB = "<a31>/review/leaderboard_levels"
sys.path.insert(0, LB)
import os  # noqa: E402
os.chdir(LB)
from common_ll import SCORES, rows_2024  # noqa: E402
import s5e_predictions as s5  # noqa: E402  (runs its own report on import; ignored)

segs, B0 = s5.segs, s5.B0
d = rows_2024(include_2023=True)
EVID = "<pack>/audit/evidence/emerging/"
k = pd.read_csv(EVID + "ev_w4_level_2025.csv").set_index("line").level_k
nb = pd.read_csv(EVID + "ev_w5_new_business.csv")
f_emg = nb[nb.source == "emerging 2023-24"].set_index("line").nb_ae_relative_to_renewal
f_bl = nb[nb.source == "blended"].set_index("line").nb_ae_relative_to_renewal


def opt(y, m):
    return (y * m ** -0.5).sum() / (m ** 0.5).sum()


def with_loading(line, f):
    """Renewal and new-business scales with loading f on new business, the level re-solved on 2023-24 (as A30 did)."""
    g = d[d.line == line]
    nbf = g.nb_flag.to_numpy()
    kr = opt(g.y_full.to_numpy(), g.c0.to_numpy() * np.where(nbf, f, 1.0))
    return kr * f, kr


def vec(scales):
    return np.array([scales[s] for s in segs])


base = {f"{ln}_{s}": k[ln] for ln in k.index for s in ("nb", "ren")}
cands = {"as is": {s: 1.0 for s in segs}, "level reset (scored 0.15021)": dict(base)}
m_nb, m_ren = with_loading("motor", f_emg["motor"])
cands[f"level reset + motor NB loading x{f_emg['motor']:.3f} (emerging 2023-24)"] = base | {"motor_nb": m_nb, "motor_ren": m_ren}
m_nb2, m_ren2 = with_loading("motor", f_bl["motor"])
cands[f"level reset + motor NB loading x{f_bl['motor']:.3f} (blended)"] = base | {"motor_nb": m_nb2, "motor_ren": m_ren2}
half = {s: np.sqrt(cands[list(cands)[2]][s]) for s in segs}  # geometric midpoint between as is and the emerging-loading file
cands["midpoint: as is and level reset + emerging motor NB loading"] = half
for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = s5.family(lo, hi, B0, SCORES)
    print(f"\nbox [{lo}, {hi}]: predicted public score across consistent solutions (min / median / max; share beating as is)")
    for name, sc in cands.items():
        t = vec(sc)
        v = np.concatenate([s5.S_of(t, cn, beta, K, B0) for cn, beta, K in fam])
        print(f"  {name:70s} {v.min():.5f} / {np.median(v):.5f} / {v.max():.5f}; {(v > SCORES['asis'] + 1e-9).mean():.2f}")
        if lo == 0.7:
            print("      scales:", {s: round(float(x), 3) for s, x in zip(segs, t)})

# Evidence (aggregates) and the halfway candidate file (row-level, work/ only).
rows = []
for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = s5.family(lo, hi, B0, SCORES)
    for name, sc in cands.items():
        v = np.concatenate([s5.S_of(vec(sc), cn, beta, K, B0) for cn, beta, K in fam])
        rows.append({"candidate": name, "scale_box": f"[{lo}, {hi}]", **{f"scale_{s}": round(float(sc[s]), 4) for s in segs},
                     "predicted_min": v.min(), "predicted_median": float(np.median(v)), "predicted_max": v.max(),
                     "share_beating_as_is": float((v > SCORES["asis"] + 1e-9).mean()), "n_solutions": len(v)})
PACK = "<pack>/"
pd.DataFrame(rows).to_csv(PACK + "audit/evidence/emerging/ev_a31_level_candidates.csv", index=False)
asis = pd.read_csv(PACK + "work/submissions/submission_emerging_as_is_20261026.csv")
t = __import__("common_ll").test_rows().set_index("row_id")
seg = t.seg.reindex(asis.row_id).to_numpy()
mid = cands["midpoint: as is and level reset + emerging motor NB loading"]
out = asis.assign(prediction=asis.prediction.to_numpy() * np.vectorize(mid.get)(seg))
assert len(out) == 209063 and out.prediction.gt(0).all() and out.row_id.equals(asis.row_id)
path = PACK + "work/submissions/submission_emerging_level_midpoint_20261026.csv"
out.to_csv(path, index=False)
print("wrote", path.split("replication/")[1], "mean", round(out.prediction.mean(), 1), "vs as is", round(asis.prediction.mean(), 1))
