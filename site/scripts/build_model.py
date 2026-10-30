"""Build site/data/model.json from Gerard's Week 2 (C0) model export.

AI-assisted (Databricks Genie Code), issue #180.

Source: contributors/gerard/workings/contents/replication/export/excel/
(copied from commit a56b201 on contents/gerard-week2-emerging). Only model
coefficients are read; no competition rows are used or published.

Scope: contents (one Tweedie GLM) and motor (small-claim GLM + large-claim GLM).
Buildings is left out because its small-claim model is XGBoost.

Usage:  python site/scripts/build_model.py [--src DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "contributors/gerard/workings/contents/replication/export/excel"
DEFAULT_OUT = REPO / "site/data/model.json"
SOURCE_COMMIT = "a56b2016297ecd76d7da964164342ca3dd4593b1"
LINES = ["contents", "motor"]

# Friendly labels and help text for the calculator. Anything not listed falls
# back to the raw term name.
LABELS = {
    "policyholder_age_band": ("Policyholder age", "Age band of the policyholder."),
    "driver_age_band": ("Driver age", "Age band of the main driver."),
    "f_flood_zone": ("Flood zone", "1 = lowest flood risk, 5 = highest."),
    "f_tenure": ("Tenure", "Owner-occupier or renter."),
    "f_segment": ("Customer segment", "Marketing segment: life stage x affluence."),
    "marketing_segment": ("Customer segment", "Marketing segment: life stage x affluence."),
    "f_prior_claims": ("Prior claims (5 yrs)", "Number of claims in the last five years (3 or more grouped)."),
    "f_missed_pay": ("Missed payments (12 m)", "Missed premium payments in the last 12 months (3 or more grouped)."),
    "f_vol_xs": ("Voluntary excess", "Extra excess the customer chose to take."),
    "f_security": ("Home security", "Alarm and locks fitted."),
    "f_hvi": ("High-value items", "Number of specified high-value items."),
    "f_ad_cover": ("Accidental damage cover", "Optional accidental damage add-on."),
    "f_bike": ("Bicycle cover", "Optional bicycle add-on."),
    "f_wfh": ("Works from home", "Someone works from home (home occupied more)."),
    "f_lodgers": ("Lodgers", "Lodgers live at the property."),
    "f_urban": ("Area type", "Urban density of the address."),
    "urban_density": ("Area type", "Urban density of the address."),
    "f_prop_type": ("Property type", "Type of building."),
    "elevation_m": ("Elevation (m)", "Height of the address above sea level."),
    "distance_to_river_km": ("Distance to river (km)", "Distance from the address to the nearest river."),
    "sum_insured": ("Sum insured", "Value of the contents covered (CU)."),
    "cover_type": ("Cover type", "Comprehensive, third party fire & theft, or third party only."),
    "channel": ("Sales channel", "Where the policy was bought."),
    "vehicle_use": ("Vehicle use", "What the car is used for."),
    "payment_method": ("Payment method", "Annual or monthly instalments."),
    "prior_claims_5y": ("Prior claims (5 yrs)", "Number of claims in the last five years."),
    "missed_payments_12m": ("Missed payments (12 m)", "Missed premium payments in the last 12 months."),
    "ncb_years": ("No-claims bonus (yrs)", "Years of no-claims bonus."),
    "years_licensed": ("Years licensed", "Years the driver has held a licence."),
    "vehicle_age": ("Vehicle age (yrs)", "Age of the car."),
    "vehicle_value_group": ("Vehicle value group", "1 = cheapest cars, 20 = most expensive."),
    "voluntary_excess": ("Voluntary excess", "Extra excess the customer chose to take."),
}

# Display labels for model levels (value -> label). Values are the model levels.
LEVEL_LABELS = {
    "f_prior_claims": {"OTHER": "3 or more"},
    "f_missed_pay": {"OTHER": "3 or more"},
    "f_flood_zone": {"MISSING": "Unknown"},
    "f_hvi": {"MISSING": "Unknown"},
    "f_security": {"MISSING": "Unknown"},
}

# Numeric inputs that are not splines need an explicit slider range.
NUMERIC_RANGES = {
    "sum_insured": dict(min=5000, max=150000, step=100),  # 100 so reference policies (e.g. 22,100) load exactly
    "prior_claims_5y": dict(min=0, max=5, step=1),
    "missed_payments_12m": dict(min=0, max=3, step=1),
}


def fnum(x: str):
    if x is None or x == "":
        return None
    v = float(x)
    if math.isinf(v):
        return None  # JSON has no infinity: null bound = open-ended
    return v


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(term: str):
    return LABELS.get(term, (term.replace("_", " ").capitalize(), ""))


def build(src: Path) -> dict:
    terms = read_csv(src / "terms.csv")
    pieces = read_csv(src / "spline_pieces.csv")
    scalars = read_csv(src / "scalars.csv")

    model = OrderedDict()
    model["meta"] = {
        "name": "Week 2 model C0 (Gerard), contents and motor",
        "source_commit": SOURCE_COMMIT,
        "source_dir": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
        "source_sha256": {n: sha256(src / n) for n in ("terms.csv", "spline_pieces.csv", "scalars.csv")},
        "note": "Coefficients only. Final = (small + large) x level step. Built by site/scripts/build_model.py.",
    }
    model["lines"] = OrderedDict()

    for line in LINES:
        L = OrderedDict(components=OrderedDict(), inputs=OrderedDict(), segments={})
        rows = [r for r in terms if r["line"] == line]
        comps = list(OrderedDict.fromkeys(r["component"] for r in rows))
        for comp in comps:
            crow = [r for r in rows if r["component"] == comp]
            C = OrderedDict(intercept=None, trend_log_2025=None, trend_multiplier_2025=None,
                            scale=1.0, scale_log=0.0, terms=[])
            by_term = OrderedDict()
            for r in crow:
                by_term.setdefault(r["term"], []).append(r)
            for term, trs in by_term.items():
                kinds = {r["kind"] for r in trs}
                if "intercept" in kinds:
                    C["intercept"] = float(trs[0]["contribution"])
                elif "trend" in kinds:
                    for r in trs:
                        if r["level"].startswith("log contribution in 2025"):
                            C["trend_log_2025"] = float(r["contribution"])
                            C["trend_multiplier_2025"] = math.exp(float(r["contribution"]))
                        elif r["level"].startswith("coefficient"):
                            C["trend_slope"] = float(r["contribution"])
                        elif r["level"].startswith("origin"):
                            C["trend_origin"] = float(r["value"])
                elif "offset" in kinds:
                    continue  # ln(exposure), applied by the engine
                elif "scale" in kinds:
                    C["scale"] = float(trs[0]["value"])
                    C["scale_log"] = float(trs[0]["contribution"])
                elif kinds == {"categorical"}:
                    levels = OrderedDict()
                    base = None
                    for r in trs:
                        if r["level"] == "(any other level)":
                            continue
                        levels[r["level"]] = float(r["contribution"])
                        if r["note"] == "base level":
                            base = r["level"]
                    C["terms"].append(dict(term=term, kind="categorical", input=term, levels=levels, base=base))
                    _add_select(L, term, levels, base)
                elif kinds == {"band"}:
                    bands, missing = [], None
                    for r in trs:
                        if r["level"] == "MISSING":
                            missing = float(r["contribution"])
                        else:
                            bands.append(dict(level=r["level"], lower=fnum(r["band_lower"]),
                                              upper=fnum(r["band_upper"]), contribution=float(r["contribution"])))
                    C["terms"].append(dict(term=term, kind="band", input=term, bands=bands, missing=missing))
                    _add_number(L, term)
                elif "spline value" in kinds:
                    grid = [float(r["value"]) for r in trs if r["kind"] == "spline value"]
                    miss = [float(r["contribution"]) for r in trs if r["kind"] == "spline missing"]
                    pcs = [dict(lower=fnum(p["lower"]), lower_included=p["lower_included"] == "True",
                                upper=fnum(p["upper"]), upper_included=p["upper_included"] == "True",
                                origin=fnum(p["origin"]) or 0.0,
                                a=[float(p[k] or 0) for k in ("a0", "a1", "a2", "a3")])
                           for p in pieces if (p["line"], p["component"], p["term"]) == (line, comp, term)]
                    step = min(b - a for a, b in zip(grid, grid[1:])) if len(grid) > 1 else 1
                    C["terms"].append(dict(term=term, kind="spline", input=term, pieces=pcs,
                                           missing=miss[0] if miss else 0.0,
                                           grid_min=min(grid), grid_max=max(grid)))
                    _add_number(L, term, dict(min=min(grid), max=max(grid), step=round(step, 6)))
                    # Offer "Unknown" only where the fit learned a missing level. A missing
                    # coefficient of exactly 0 means the fit saw no missing rows, so pricing
                    # "unknown" there would just switch the curve off (e.g. years licensed:
                    # x0.03-0.07). Issue #181 follow-up from Cian's test.
                    if miss and miss[0] != 0.0:
                        L["inputs"][term]["allow_missing"] = True
                elif kinds == {"continuous"}:
                    r = trs[0]
                    if term == "log_si":
                        C["terms"].append(dict(term=term, kind="log_si", input="sum_insured",
                                               coefficient=float(r["contribution"]),
                                               note="ln(max(sum insured, 1)); sum insured / 1,000 at or above 1m (D-02)"))
                        _add_number(L, "sum_insured")
                    else:
                        raise ValueError(f"unsupported continuous term {line}/{comp}/{term}")
                elif kinds == {"spline missing"}:
                    raise ValueError(f"spline without grid {line}/{comp}/{term}")
                else:
                    raise ValueError(f"unhandled kinds {kinds} for {line}/{comp}/{term}")
            L["components"][comp] = C

        # level steps by segment, applied in this order
        for seg in ("nb", "ren"):
            steps = []
            for r in scalars:
                if r["line"] == line and r["segment"] == f"{line}_{seg}" and r["applied"] == "True":
                    if r["item"].startswith("level step 1"):
                        s1 = float(r["value"])
                        steps.append(dict(op="times", value=s1, label="Level step 1: halfway scale"))
                    elif r["item"].startswith("level step 2"):
                        steps.append(dict(op="times", value=float(r["value"]), label="Level step 2: x1.05"))
                        steps.append(dict(op="divide", value=s1, label="Level step 2: divide by halfway scale"))
            L["segments"][seg] = steps
        model["lines"][line] = L
    return model


def _add_select(L, term, levels, base):
    if term in L["inputs"]:
        return
    lab, help_ = label(term)
    names = LEVEL_LABELS.get(term, {})
    def order(v):  # numbers in numeric order, then text, then OTHER / MISSING last
        try:
            return (0, float(v), "")
        except ValueError:
            return (2, 0.0, v) if v in ("OTHER", "MISSING") else (1, 0.0, v)
    opts = [dict(value=v, label=names.get(v, v.replace("_", " "))) for v in sorted(levels, key=order)]
    L["inputs"][term] = dict(label=lab, help=help_, type="select", options=opts, default=base)


def _add_number(L, term, rng=None):
    lab, help_ = label(term)
    rng = rng or NUMERIC_RANGES.get(term)
    cur = L["inputs"].get(term)
    if cur is None:
        L["inputs"][term] = dict(label=lab, help=help_, type="number", **(rng or {}))
    elif rng and "min" in rng and "min" not in cur:
        cur.update(rng)  # a spline grid gives a better range than a band
    if "min" not in L["inputs"][term]:
        raise ValueError(f"no range for numeric input {term}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    model = build(a.src.resolve())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(model, indent=1), encoding="utf-8")
    n = sum(len(c["terms"]) for L in model["lines"].values() for c in L["components"].values())
    print(f"wrote {a.out} ({n} terms)")


if __name__ == "__main__":
    main()
