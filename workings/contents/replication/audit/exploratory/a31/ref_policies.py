"""Scratch (A31): the 20 reference policies priced by C0 (production) and two XGBoost sense checks, with the office
premium build-up. Writes a row-level table to work/ (not committed) and prints aggregates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "python")
import confirm as cf  # noqa: E402
import harness as h  # noqa: E402
import office_premium as op  # noqa: E402

ref = pd.read_parquet("data/emerging/reference_policies.parquet")
r = op.rates(pd.read_parquet("data/emerging/expenses.parquet"), "emerging")
emg = {ln: h.load(ln) for ln in h.LINES}
train = {ln: pd.concat([emg[ln][emg[ln].split.isin(["fit2023", "A2024"])], cf.load_b(ln)]) for ln in h.LINES}
score = {ln: emg[ln][emg[ln].row_id.isin(ref.row_id)] for ln in h.LINES}
dev = {ln: h.load_dev(ln) for ln in h.LINES}
models = {"C0 (production)": {ln: score[ln].c0.to_numpy() for ln in h.LINES}}
for key, label in (("v8_emerging_gbm_benchmark", "XGB emerging only"), ("v3_joint_gbm", "XGB joint markets")):
    models[label] = cf.module(key).predict(train, score, dev, 2025)
rows = []
for ln in h.LINES:
    s = score[ln]
    for label, preds in models.items():
        b = op.build_up(preds[ln], s.exposure_plain.to_numpy(), np.full(len(s), ln), s.channel.astype(str).to_numpy(), r)
        rows.append(pd.DataFrame({"row_id": s.row_id.to_numpy(), "line": ln, "model": label,
                                  "channel": s.channel.astype(str).to_numpy(), "new_business": s.new_business.to_numpy(),
                                  "exposure": s.exposure_plain.to_numpy(), "risk_premium_period": preds[ln],
                                  "written_premium": s.written_premium.to_numpy()}).join(b))
out = pd.concat(rows, ignore_index=True)
assert out.row_id.nunique() == len(ref) == 20 and (out.risk_premium_period > 0).all()
out.to_csv("work/reference_policies_2025.csv", index=False)
wide = out.pivot_table(index=["line", "row_id"], columns="model", values="risk_premium_annual")
c0 = wide["C0 (production)"]
print("ratio of annual risk premium to C0 by line (median, min, max):")
for m in wide.columns.drop("C0 (production)"):
    q = (wide[m] / c0).groupby(level=0).agg(["median", "min", "max"]).round(2)
    print(" ", m, q.to_dict("index"))
print("rank correlation with C0 within line:", {m: wide.groupby(level=0).apply(lambda g: g[m].corr(g["C0 (production)"], method="spearman")).round(2).to_dict() for m in wide.columns.drop("C0 (production)")})
p0 = out[out.model == "C0 (production)"]
print("C0 office premium against written premium by line (median ratio):", (p0.office_premium / p0.written_premium).groupby(p0.line).median().round(2).to_dict())
print("C0 risk premium against written premium by line (median ratio):", (p0.risk_premium_annual / p0.written_premium).groupby(p0.line).median().round(2).to_dict())
