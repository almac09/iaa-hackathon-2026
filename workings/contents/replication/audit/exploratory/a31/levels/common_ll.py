"""Shared loaders for the A31 leaderboard-levels review (scratch)."""
import numpy as np, pandas as pd
A = "<a31>"
SUB = "<pack>/work/submissions/"
FILES = {"asis": "submission_emerging_as_is_20261026.csv",
         "nb": "submission_emerging_level_reset_nb_scenario_20261026.csv",
         "lr": "submission_emerging_level_reset_20261026.csv"}
SCORES = {"asis": 0.15601, "nb": 0.15451, "lr": 0.15021}
LEADER = 0.17303
LINES = ("buildings", "contents", "motor")
P = 1.5

def unit_dev(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    return 2 * (np.power(y, 2 - P) / ((1 - P) * (2 - P)) - y * np.power(mu, 1 - P) / (1 - P) + np.power(mu, 2 - P) / (2 - P))

def test_rows():
    t = []
    for ln in LINES:
        d = pd.read_parquet(f"{A}/cache/emg_{ln}.parquet")
        d = d[d.split == "test2025"][["row_id", "c0", "new_business", "exposure_plain", "exposure"]].copy()
        d["line"] = ln
        t.append(d)
    t = pd.concat(t, ignore_index=True)
    for k, f in FILES.items():
        s = pd.read_csv(SUB + f).rename(columns={"prediction": k})
        t = t.merge(s, on="row_id", how="left")
    t["nb_flag"] = t.new_business.astype(bool)
    t["seg"] = t.line + np.where(t.nb_flag, "_nb", "_ren")
    return t

def rows_2024(include_2023=False):
    out = []
    for ln in LINES:
        a = pd.read_parquet(f"{A}/cache/emg_{ln}.parquet")
        b = pd.read_parquet(f"{A}/cache/holdout_B/emg_{ln}_B.parquet")
        keep = ["A2024", "fit2023"] if include_2023 else ["A2024"]
        d = pd.concat([a[a.split.isin(keep)], b], ignore_index=True)
        d = d[["row_id", "customer_id", "split", "yr", "y_full", "exposure_plain", "c0", "new_business"]].copy()
        d["line"] = ln
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["customer_id"] = d.customer_id.astype(str)
    d["nb_flag"] = d.new_business.astype(bool)
    d["seg"] = d.line + np.where(d.nb_flag, "_nb", "_ren")
    return d
