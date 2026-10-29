"""Copy the aggregate EDA evidence tables the site uses into site/data/eda/.

AI-assisted (Databricks Genie Code), issue #183.

The site is public, and competition rules section 5b forbid publishing the
competition data. So this script only takes tables that are already
aggregated (Gerard's audit/evidence CSVs, produced by his logged runs). Each
file must pass two guards before it is copied:

* at most MAX_ROWS rows (aggregates are small; row-level data is not);
* no identifier column (row_id, policy_id, customer_id, address_id, claim_id).

If either guard fails, the build stops.

Usage:  python site/scripts/build_eda.py [--out site/data/eda]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "contributors/gerard/workings/contents/replication/audit/evidence"
MAX_ROWS = 200
ID_COLUMNS = {"row_id", "policy_id", "customer_id", "address_id", "claim_id"}

# output name -> source file under audit/evidence
TABLES = {
    "row_counts.csv": "python/ev_01_row_counts.csv",
    "claims_by_line_year.csv": "python/ev_01_claims_by_line_year.csv",
    "exposure_units.csv": "python/ev_02_exposure_units.csv",
    "loss_ratio_by_line_year.csv": "python/ev_04_loss_ratio_by_line_year.csv",
    "severity_by_peril_contents.csv": "python/ev_05_severity_by_peril.csv",
    "cost_concentration_contents.csv": "python/ev_05_cost_concentration.csv",
    "contacts_by_claim_status.csv": "python/ev_07_contacts_by_claim_status.csv",
    "target_by_line_year.csv": "glm_lines/ev_11_target_by_line_year.csv",
    "report_lag_by_peril.csv": "emerging/ev_w3_report_lag_patterns.csv",
    "new_business_emerging.csv": "emerging/ev_w4_new_business.csv",
}


def check(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        n = sum(1 for _ in reader)
    bad = ID_COLUMNS & {h.strip().lower() for h in header}
    if bad:
        sys.exit(f"refusing {path}: identifier column(s) {sorted(bad)}")
    if n > MAX_ROWS:
        sys.exit(f"refusing {path}: {n} rows > {MAX_ROWS}; not an aggregate table")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "site/data/eda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, src in TABLES.items():
        p = EVIDENCE / src
        n = check(p)
        shutil.copyfile(p, a.out / name)
        print(f"{name:36s} {n:4d} rows  <- {src}")


if __name__ == "__main__":
    main()
