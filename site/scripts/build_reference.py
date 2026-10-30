"""Build site/data/reference_policies.json from the Part 2 reference policies.

AI-assisted (Databricks Genie Code), calculator follow-up to issue #181.

Source: iaa-ai-assisted-hackathon-2026-part-2/emerging/reference_policies.parquet
(20 rows: motor, buildings, contents). Each row gets idx 0..n-1 in file order and
keeps its own row_id. Missing rating factors are joined in:
  * addresses (address_id): flood zone, area type, property type, elevation, distance to river
  * customers (customer_id): customer segment, prior claims (5 yrs)
Contents factors are derived as in Gerard's sql/08 (input_mappings.csv):
lvl(x) = coalesce(CAST(x AS VARCHAR), 'MISSING'), counts capped at 3 and 3 = OTHER.
Exposure = (period_end - period_start + 1) / 365 (D-01).
Buildings rows get their large-claim GLM inputs (weather joined on address and
policy year, construction codes and D-02/D-27 fixes as Gerard's pipeline); the
small-claim model is XGBoost, so no full buildings risk or office premium is quoted.

Usage:  python site/scripts/build_reference.py [--model site/data/model.json] [--out FILE]
Exits 1 if any value is not a model level, outside a slider range or off its step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EM = REPO / "iaa-ai-assisted-hackathon-2026-part-2/emerging"
# Gerard's office premium module (audit A31): the same expense rates, incl. the comparison-site fallback
sys.path.insert(0, str(REPO / "contributors/gerard/workings/contents/replication/python"))
import office_premium as op  # noqa: E402

BRIEF = ("reference_policies (emerging folder): Twenty 2025 emerging-market policies taken from the test files "
         "(same columns, plus line). For each one, show in your written report how you build the office premium from "
         "your predicted risk premium using the expenses assumptions. This is assessed by the judging panel and does "
         "not affect the Kaggle score.")

DISPLAY = {
    "motor": ["driver_age_band", "years_licensed", "ncb_years", "vehicle_value_group", "vehicle_age", "make_model",
              "cover_type", "vehicle_use", "channel", "payment_method", "voluntary_excess", "prior_claims_5y",
              "missed_payments_12m", "marketing_segment", "urban_density"],
    "contents": ["policyholder_age_band", "sum_insured", "tenure", "security_level", "high_value_items",
                 "accidental_damage_cover", "bicycle_cover", "works_from_home", "lodgers", "voluntary_excess",
                 "prior_claims_5y", "missed_payments_12m", "marketing_segment", "flood_zone", "urban_density",
                 "property_type", "elevation_m", "distance_to_river_km"],
    "buildings": ["policyholder_age_band", "sum_insured", "construction_type", "occupancy", "year_built",
                  "property_type", "flood_zone", "distance_to_river_km", "elevation_m", "soil_type",
                  "dry_summer_index", "rainfall_mm", "urban_density",
                  "accidental_damage_cover", "voluntary_excess", "prior_claims_5y", "missed_payments_12m",
                  "marketing_segment", "channel", "payment_method"],
}


CONSTRUCTION = {"BRK": "brick", "STN": "stone", "WOD": "timber", "MIX": "non_standard"}  # conforming view
YEAR_BUILT_1900 = {"A": 2008, "B": 1993, "C": 1972, "D": 1951, "E": 1929, "F": 1902, "G": 1880, None: 1961}  # D-27


def isna(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def lvl(v, cap: int | None = None) -> str:
    if isna(v):
        return "MISSING"
    if isinstance(v, bool) or str(v) in ("True", "False"):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if cap is not None:
        v = min(int(v), cap)
    return str(v)


def num(v, nd: int | None = None):
    if isna(v):
        return None
    v = float(v)
    if nd is not None:
        v = round(v, nd)
    return int(v) if v.is_integer() else v


def shown(v):
    if isna(v):
        return "Unknown"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v.item() if hasattr(v, "item") else v


def load() -> pd.DataFrame:
    ref = pd.read_parquet(EM / "reference_policies.parquet")
    ref.insert(0, "idx", range(len(ref)))
    addr = pd.read_parquet(EM / "addresses.parquet")
    cust = pd.read_parquet(EM / "customers.parquet")
    weather = pd.read_parquet(EM / "weather.parquet", columns=["address_id", "year", "dry_summer_index", "rainfall_mm"])
    ref["year"] = pd.to_datetime(ref["period_start"]).dt.year  # sql/11: weather on address and policy year
    full = (ref.merge(addr, on="address_id", how="left", suffixes=("", "_addr"), validate="m:1", indicator="_a")
               .merge(cust, on="customer_id", how="left", suffixes=("", "_cust"), validate="m:1", indicator="_c")
               .merge(weather, on=["address_id", "year"], how="left", validate="m:1"))
    bad = full[(full["_a"] != "both") | (full["_c"] != "both")]
    if len(bad):
        raise SystemExit(f"unmatched address/customer for rows {bad['row_id'].tolist()}")
    return full


def policy_inputs(r, line: str) -> dict:
    days = (pd.Timestamp(r["period_end"]) - pd.Timestamp(r["period_start"])).days + 1
    p = {"exposure": days / 365, "new_business": bool(r["new_business"])}
    if line == "contents":
        p.update(f_flood_zone=lvl(r["flood_zone"]), f_tenure=lvl(r["tenure"]), f_segment=lvl(r["marketing_segment"]),
                 f_prior_claims=lvl(r["prior_claims_5y"], 3), f_missed_pay=lvl(r["missed_payments_12m"], 3),
                 f_vol_xs=lvl(r["voluntary_excess"]), f_security=lvl(r["security_level"]),
                 f_hvi=lvl(r["high_value_items"]), f_ad_cover=lvl(r["accidental_damage_cover"]),
                 f_bike=lvl(r["bicycle_cover"]), f_wfh=lvl(r["works_from_home"]), f_lodgers=lvl(r["lodgers"]),
                 f_urban=lvl(r["urban_density"]), f_prop_type=lvl(r["property_type"]),
                 policyholder_age_band=lvl(r["policyholder_age_band"]), elevation_m=num(r["elevation_m"], 1),
                 distance_to_river_km=num(r["distance_to_river_km"], 2), sum_insured=num(r["sum_insured"]))
        for k in ("f_prior_claims", "f_missed_pay"):
            if p[k] == "3":
                p[k] = "OTHER"  # merged level (input_mappings.csv)
    elif line == "motor":
        p.update(marketing_segment=lvl(r["marketing_segment"]), cover_type=lvl(r["cover_type"]),
                 urban_density=lvl(r["urban_density"]), channel=lvl(r["channel"]), vehicle_use=lvl(r["vehicle_use"]),
                 payment_method=lvl(r["payment_method"]), driver_age_band=lvl(r["driver_age_band"]),
                 prior_claims_5y=num(r["prior_claims_5y"]), missed_payments_12m=num(r["missed_payments_12m"]),
                 ncb_years=num(r["ncb_years"]), years_licensed=num(r["years_licensed"]),
                 vehicle_age=num(r["vehicle_age"]), vehicle_value_group=num(r["vehicle_value_group"]),
                 voluntary_excess=num(r["voluntary_excess"]))
    elif line == "buildings":
        # large-claim GLM inputs only (the small-claim XGBoost isn't in the calculator)
        si = num(r["sum_insured"])
        if si is not None and si >= 1e7:
            si = si / 1000  # D-02 (buildings)
        yb = num(r["year_built"])
        if yb == 1900:  # D-27: placeholder year -> emerging mean year built for the energy rating
            yb = YEAR_BUILT_1900.get(None if isna(r["energy_rating"]) else str(r["energy_rating"]), YEAR_BUILT_1900[None])
        ct = None if isna(r["construction_type"]) else CONSTRUCTION.get(str(r["construction_type"]).upper(), str(r["construction_type"]))
        p.update(soil_type=lvl(r["soil_type"]), construction_type=lvl(ct), flood_zone=num(r["flood_zone"]),
                 sum_insured=si, elevation_m=num(r["elevation_m"], 1), year_built=yb,
                 distance_to_river_km=num(r["distance_to_river_km"], 2),
                 dry_summer_index=num(r["dry_summer_index"], 3), rainfall_mm=num(r["rainfall_mm"]))
    return p


def validate(model: dict, line: str, name: str, p: dict) -> list[str]:
    if line not in model["lines"]:
        return []
    errs = []
    for k, inp in model["lines"][line]["inputs"].items():
        v = p.get(k, "MISSING_KEY")
        if inp["type"] == "select":
            if str(v) not in [o["value"] for o in inp["options"]]:
                errs.append(f"{name} {k}={v!r}: not a model level")
        elif v is None:
            if not inp.get("allow_missing"):
                errs.append(f"{name} {k}: missing but the model has no Unknown level")
        elif not inp["min"] <= v <= inp["max"]:
            errs.append(f"{name} {k}={v}: outside {inp['min']}-{inp['max']}")
        elif abs(round((v - inp["min"]) / inp["step"]) * inp["step"] + inp["min"] - v) > 1e-9:
            errs.append(f"{name} {k}={v}: not on slider step {inp['step']}")
    return errs


def build(model: dict) -> tuple[dict, list[str]]:
    full = load()
    exp_raw = pd.read_parquet(EM / "expenses.parquet")
    rates = {k: float(v) for k, v in op.rates(exp_raw, "emerging").items()}
    out = {"_note": "The reference policies from the Part 2 dataset (emerging/reference_policies.parquet), idx 0..n-1 in "
                    "file order. Built by site/scripts/build_reference.py. Buildings rows are shown, not priced.",
           "brief": BRIEF,
           "expenses": {"rates_emerging": rates,
                        "comparison_site_fallback": op.COMPARISON_SITE_FALLBACK,
                        "table": [{"item": r.item, "developed": None if isna(r.developed) else float(r.developed),
                                   "emerging": None if isna(r.emerging) else float(r.emerging), "unit": r.unit}
                                  for r in exp_raw.itertuples()]},
           "policies": []}
    errs = []
    for _, r in full.iterrows():
        line, idx = r["line"], int(r["idx"])
        for col in ("row_id", "policy_id"):
            prefix = {"M": "motor", "C": "contents", "B": "buildings"}.get(str(r[col])[0])
            if prefix != line:
                errs.append(f"{r[col]}: {col} prefix says {prefix} but line is {line}")
        name = f"{idx}. {r['row_id']}"
        p = policy_inputs(r, line)
        errs += validate(model, line, name, p)
        out["policies"].append({
            "id": f"ref_{idx:02d}", "idx": idx, "row_id": r["row_id"], "policy_id": str(r["policy_id"]),
            "type": str(r["row_id"])[0], "line": line, "name": name,
            # priced = a full risk premium can be quoted; partial = only some components (buildings: GLM large part)
            "priced": line in model["lines"] and not model["lines"][line].get("missing_components"),
            "partial": line in model["lines"] and bool(model["lines"][line].get("missing_components")),
            "channel": str(r["channel"]), "written_premium": float(r["written_premium"]),
            "period": f"{r['period_start']} to {r['period_end']}",
            "description": f"{line.capitalize()}, {'new business' if r['new_business'] else 'renewal'}, "
                           f"{r['period_start']} to {r['period_end']} (exposure {p['exposure']:.3f}).",
            "display": {c.replace("_", " "): shown(r.get(c)) for c in DISPLAY[line]},
            "policy": p,
        })
    return out, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "site/data/model.json")
    ap.add_argument("--out", type=Path, default=REPO / "site/data/reference_policies.json")
    a = ap.parse_args()
    model = json.loads(a.model.read_text())
    out, errs = build(model)
    for e in errs:
        print("ERROR", e)
    if errs:
        sys.exit(1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    by = pd.Series([p["line"] for p in out["policies"]]).value_counts().to_dict()
    print(f"wrote {a.out} ({len(out['policies'])} policies: {by})")


if __name__ == "__main__":
    main()
