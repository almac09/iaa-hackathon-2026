"""Office premium built up from the risk premium and the market's expense table (audit A31).

The competition scores the risk premium, the expected claim cost. The premium charged adds, per policy-year:

    office = (risk x (1 + cat reinsurance, property lines only) + fixed expense)
             / (1 - variable expense - commission for the channel - premium tax - profit margin)

with every rate from expenses.parquet ("share of premium", except cat reinsurance, a share of the property risk
premium, and the fixed expense, per policy). The risk premium is annualised first (the row's expected cost over its
exposure in years), so policies on cover for part of the year are quoted on the same annual basis as the written
premium, which is annual (A12).

The emerging table has no comparison-site commission although 5% of the emerging books sell through comparison
sites. The developed market's rate stands in for it (COMPARISON_SITE_FALLBACK): an assumption for the owner.

Usage (from the pack root): price the emerging reference policies from a submission file. The row-level table goes
to work/ (not committed); the summary by line is printed.
  python python/office_premium.py work/submissions/submission_emerging_as_is_20261026.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROPERTY = {"buildings", "contents"}
COMMISSION = {"direct": "commission_direct", "broker": "commission_broker", "comparison_site": "commission_comparison_site"}
COMPARISON_SITE_FALLBACK = "developed"  # the market whose comparison-site commission fills a gap


def rates(expenses: pd.DataFrame, market: str) -> dict:
    """The market's expense items as a dict, with a missing comparison-site commission filled (see module docstring)."""
    r = expenses.set_index("item")[market].to_dict()
    if pd.isna(r.get("commission_comparison_site")):
        r["commission_comparison_site"] = float(expenses.set_index("item").loc["commission_comparison_site",
                                                                                  COMPARISON_SITE_FALLBACK])
    missing = [k for k, v in r.items() if pd.isna(v)]
    if missing:
        raise ValueError(f"expense items missing for {market}: {missing}")
    return r


def build_up(risk: np.ndarray, exposure: np.ndarray, line: np.ndarray, channel: np.ndarray, r: dict) -> pd.DataFrame:
    """Annual office premium and its components for each row. `risk` is the expected claim cost for the row's period."""
    risk, exposure = np.asarray(risk, float), np.asarray(exposure, float)
    if np.any(exposure <= 0):
        raise ValueError("exposure must be positive to annualise the risk premium")
    unknown = set(channel) - set(COMMISSION)
    if unknown:
        raise ValueError(f"no commission rate for channel(s) {sorted(unknown)}")
    annual = risk / exposure
    cat = np.where(np.isin(line, list(PROPERTY)), r["cat_reinsurance"], 0.0) * annual
    commission = np.array([r[COMMISSION[c]] for c in channel])
    loading = r["variable_expense"] + commission + r["premium_tax"] + r["target_profit_margin"]
    if np.any(loading >= 1):
        raise ValueError("premium-proportional loadings reach 100% of premium")
    office = (annual + cat + r["fixed_expense_per_policy"]) / (1 - loading)
    return pd.DataFrame({"risk_premium_annual": annual, "cat_reinsurance": cat,
                         "fixed_expense": r["fixed_expense_per_policy"],
                         "variable_expense": office * r["variable_expense"], "commission": office * commission,
                         "premium_tax": office * r["premium_tax"], "profit_margin": office * r["target_profit_margin"],
                         "office_premium": office})


def price_reference(submission: Path, data_dir: Path = Path("data/emerging")) -> pd.DataFrame:
    """The market's reference policies priced from a submission file's risk premiums."""
    ref = pd.read_parquet(data_dir / "reference_policies.parquet")
    sub = pd.read_csv(submission, float_precision="round_trip").set_index("row_id").prediction  # exact, as written
    missing = set(ref.row_id) - set(sub.index)
    if missing:
        raise ValueError(f"{len(missing)} reference policies are not in {submission.name}")
    risk = sub.reindex(ref.row_id).to_numpy()
    b = build_up(risk, ref.exposure.to_numpy(), ref.line.to_numpy(), ref.channel.astype(str).to_numpy(),
                 rates(pd.read_parquet(data_dir / "expenses.parquet"), "emerging"))
    return pd.concat([ref[["row_id", "line", "channel", "new_business", "exposure", "written_premium"]]
                      .reset_index(drop=True).assign(risk_premium_period=risk), b], axis=1)


if __name__ == "__main__":
    out = price_reference(Path(sys.argv[1]))
    Path("work").mkdir(exist_ok=True)
    out.to_csv("work/reference_policies_office_premium.csv", index=False)
    summary = out.assign(office_over_written=out.office_premium / out.written_premium,
                         risk_over_written=out.risk_premium_annual / out.written_premium) \
        .groupby("line")[["risk_premium_annual", "office_premium", "written_premium", "office_over_written",
                          "risk_over_written"]].median().round(2)
    print(summary.to_string())
