"""A submission with one line rescaled to a target mean prediction per row, for a one-line A/B on the leaderboard
(audit A24). Every other row is unchanged, so the score difference isolates that line's level.

Usage (from the pack root):
  python python/level_variant.py --source work/submissions/<file>.csv --prefix M --target-mean 424.91 --label motor_v014
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RUNNER = "level_variant"


def rescale(sub: pd.DataFrame, prefix: str, target_mean: float) -> tuple[pd.DataFrame, float]:
    """`sub` with the rows whose row_id starts with `prefix` scaled so their mean equals `target_mean`."""
    rows = sub.iloc[:, 0].astype(str).str.startswith(prefix)
    if not rows.any():
        raise ValueError(f"no rows with prefix {prefix!r}")
    factor = target_mean / sub.loc[rows].iloc[:, 1].mean()
    out = sub.copy()
    out.loc[rows, out.columns[1]] = out.loc[rows, out.columns[1]] * factor
    return out, float(factor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--target-mean", type=float, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_known_args()[0]  # --run-label is read by common.new_run_record
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    try:
        src = common.PACK_ROOT / args.source
        sub = pd.read_csv(src, dtype={0: str})
        out, factor = rescale(sub, args.prefix, args.target_mean)
        v = out.iloc[:, 1]
        if v.isna().any() or not (np.isfinite(v) & (v > 0)).all():
            raise ValueError("rescaled submission has missing, infinite or non-positive values")
        path = src.with_name(f"{src.stem}_{args.label}.csv")
        out.round(6).to_csv(path, index=False)
        rec["inputs"] = {"source": src.relative_to(common.PACK_ROOT).as_posix(), "source_sha256": common.sha256_file(src)}
        rec["outputs_sha256"] = {path.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(path)}
        rec["metrics"] = {"prefix": args.prefix, "target_mean": args.target_mean, "factor": factor, "rows": len(out)}
        rec["status"] = "success"
        print(f"{path.relative_to(common.PACK_ROOT)}: {args.prefix} rows x {factor:.4f} (mean {args.target_mean})")
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
