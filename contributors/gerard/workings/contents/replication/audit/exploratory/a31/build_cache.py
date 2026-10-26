"""Scratch (A31 exploration): build both markets once and cache row-level frames for the model variations.

Writes, per line: emg_<line>.parquet (emerging 2023 fit rows, 2024 explore half A, 2025 test rows), the held-back
2024 half B separately (holdout_B/), and dev_<line>.parquet (developed 2021-24). Each row carries the portable
developed model's prediction for its own year (C0; 2025 for test rows), the target, exposure and every raw field.
Row-level data: delete the cache after use."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "python")
import common  # noqa: E402
import emerging as em  # noqa: E402

OUT = Path(sys.argv[1])
cfg = common.load_config()
rec = common.new_run_record("scratch_a31_cache", cfg)
ctx = em.Context(cfg, rec)

RAW = {"emerging": Path("data/emerging"), "developed": Path("data")}
TABLE = {"buildings": "buildings", "contents": "contents", "motor": "motor"}


def extras(market: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Raw fields the Line frame lacks, joined by row, customer, address and (address, year)."""
    raw = RAW[market]
    pol = pd.concat([pd.read_parquet(raw / f"policies_{TABLE[name]}.parquet"),
                     pd.read_parquet(raw / f"test_{TABLE[name]}.parquet")], ignore_index=True)
    add = [c for c in pol.columns if c not in df.columns and c != "row_id"]
    out = df.merge(pol[["row_id"] + add].drop_duplicates("row_id"), on="row_id", how="left")
    cust = pd.read_parquet(raw / "customers.parquet")
    cust["customer_id"] = cust.customer_id.astype(str)
    addc = [c for c in cust.columns if c not in out.columns and c not in ("customer_id", "surname")]
    out = out.merge(cust[["customer_id"] + addc].drop_duplicates("customer_id"), on="customer_id", how="left")
    if "address_id" in out:
        adr = pd.read_parquet(raw / "addresses.parquet")
        adda = [c for c in adr.columns if c not in out.columns and c != "address_id"]
        out = out.merge(adr[["address_id"] + adda].drop_duplicates("address_id"), on="address_id", how="left")
    assert len(out) == len(df)
    return out


# One customer split for the 2024 emerging rows across all lines: half A to explore, half B held back to confirm.
cust24 = sorted(set().union(*[set(ctx.line(n).rows([2024]).customer_id.astype(str)) for n in ("buildings", "contents", "motor")]))
rng = np.random.default_rng(2031)
half_b = set(np.array(cust24)[rng.permutation(len(cust24))[: len(cust24) // 2]])
meta = {"P": em.P, "features": {}, "rows": {}}
asis = pd.read_csv("work/submissions/submission_emerging_as_is_20261026.csv").set_index("row_id").prediction
for name in ("buildings", "contents", "motor"):
    line = ctx.line(name)  # motor on the developed reporting pattern, as E3 chose
    parts = []
    for yr in (2023, 2024):
        d = line.rows([yr]).copy()
        d["c0"] = ctx.base(name, d, yr)
        parts.append(d)
    t = line.test.copy()
    t["c0"] = ctx.base(name, t, 2025)
    parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df["customer_id"] = df.customer_id.astype(str)
    df = extras("emerging", name, df)
    df["split"] = np.select([df.src.eq("test"), df.yr.eq(2023), df.customer_id.isin(half_b)],
                            ["test2025", "fit2023", "B2024"], "A2024")
    gap = (df[df.split == "test2025"].set_index("row_id").c0 / asis.reindex(df[df.split == "test2025"].row_id)).sub(1).abs().max()
    assert gap < 1e-9, f"{name}: cache C0 does not reproduce the as-is submission ({gap})"
    for c in df.columns:  # parquet needs one type per object column
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df[df.split != "B2024"].to_parquet(OUT / f"emg_{name}.parquet", index=False)
    df[df.split == "B2024"].to_parquet(OUT / "holdout_B" / f"emg_{name}_B.parquet", index=False)
    dl = ctx.dev[name]
    dd = dl.rows([2021, 2022, 2023, 2024]).copy()
    dd["c0_in_sample"] = ctx.models[name].predict(dd)
    dd["customer_id"] = dd.customer_id.astype(str)
    dd = extras("developed", name, dd)
    for c in dd.columns:
        if dd[c].dtype == object:
            dd[c] = dd[c].astype("string")
    dd.to_parquet(OUT / f"dev_{name}.parquet", index=False)
    pc = ctx.pcfg
    meta["features"][name] = {"glm": {k: pc["glm_lines"]["models"][name].get(k) for k in ("numeric", "categorical", "continuous")},
                              "xgb": {k: pc["lines"]["models"][name][k] for k in ("numeric", "categorical")},
                              "excess_threshold": ctx.settings[name]["threshold"], "trend_slope": float(ctx.models[name].small.slope)}
    meta["rows"][name] = df.split.value_counts().to_dict() | {"developed": len(dd)}
    print(name, meta["rows"][name], flush=True)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
