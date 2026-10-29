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
MAX_ROWS = 500  # the largest table, contents one-way factor x level, has 394 rows
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
    # added for the extra EDA charts, the data-cleaning page and the model page (issue #185)
    "oneway_contents.csv": "python/ev_m1_oneway_tables.csv",
    "size_band_mix_contents.csv": "python/ev_05_size_band_mix.csv",
    "claim_persistence_contents.csv": "python/ev_06_transition_matrix.csv",
    "importance_xgb.csv": "lines/ev_l5_importance.csv",
    "missing_values_contents.csv": "python/ev_02_contents_missing_values.csv",
    "exposure_before_after_contents.csv": "python/ev_03_exposure_before_after.csv",
    "sum_insured_before_after_contents.csv": "python/ev_03_sum_insured_before_after.csv",
    "claim_reconciliation_contents.csv": "python/ev_03_claim_reconciliation.csv",
    "join_completeness_contents.csv": "python/ev_03_join_completeness.csv",
    "motor_id_padding.csv": "python/ev_02_motor_customer_id_padding.csv",
    "fixes_by_line.csv": "glm_lines/ev_09_fixes.csv",
    "backtest_2024.csv": "glm_lines/ev_g3_backtest_2024.csv",
    "large_claim_tail.csv": "glm_lines/ev_g4_large_claim_tail.csv",
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
