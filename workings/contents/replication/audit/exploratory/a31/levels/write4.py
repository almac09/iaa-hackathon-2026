"""Scratch (A31): evidence for the four-score refit, and the halfway file with motor new business at x1.05."""
import io, contextlib, sys  # noqa: E401
sys.argv = ["x"]
with contextlib.redirect_stdout(io.StringIO()):
    import update4 as u  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
PACK = "<pack>/"
base = u.cd.cands["midpoint: as is and level reset + emerging motor NB loading"]
cands = {"halfway file (scored 0.15742)": base, "halfway file, motor new business x1.05": dict(base) | {"motor_nb": 1.05}}
rows = []
for lo, hi in ((0.7, 1.4), (0.6, 1.5)):
    fam = u.family(lo, hi)
    K = np.concatenate([k for _, _, k in fam], axis=1)
    for i, s in enumerate(u.segs):
        rows.append({"kind": "segment level consistent with the four scores", "item": s, "scale_box": f"[{lo}, {hi}]",
                     "min": K[i].min(), "median": float(np.median(K[i])), "max": K[i].max(), "n_solutions": K.shape[1]})
    gain = np.concatenate([beta * (u.B0[:, None] * (np.sqrt(k) - 1) ** 2).sum(0) for cn, beta, k in fam])
    rows.append({"kind": "public gain over as is at the true segment levels", "item": "all", "scale_box": f"[{lo}, {hi}]",
                 "min": gain.min(), "median": float(np.median(gain)), "max": gain.max(), "n_solutions": len(gain)})
    for name, sc in cands.items():
        v = np.concatenate([u.S_of(u.cd.vec(sc), cn, beta, k) for cn, beta, k in fam])
        rows.append({"kind": "predicted public score", "item": name, "scale_box": f"[{lo}, {hi}]", "min": v.min(),
                     "median": float(np.median(v)), "max": v.max(), "n_solutions": len(v)})
pd.DataFrame(rows).to_csv(PACK + "audit/evidence/emerging/ev_a31_level_four_scores.csv", index=False)
mid = pd.read_csv(PACK + "work/submissions/submission_emerging_level_midpoint_20261026.csv")
t = __import__("common_ll").test_rows().set_index("row_id")
seg = t.seg.reindex(mid.row_id).to_numpy()
out = mid.assign(prediction=np.where(seg == "motor_nb", mid.prediction * 1.05 / base["motor_nb"], mid.prediction))
assert len(out) == 209063 and out.prediction.gt(0).all()
path = PACK + "work/submissions/submission_emerging_level_midpoint_motornb105_20261026.csv"
out.to_csv(path, index=False)
print("wrote", path.split("replication/")[1], "mean", round(out.prediction.mean(), 1))
