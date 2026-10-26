"""One-way analysis of home contents rating factors.

Each factor is banded, then scored out of time: level loss-cost rates are fitted on
2021-23 and used to predict 2024 row cost (rate x exposure). The score is the gain in
Tweedie D2 (power 1.5, the competition metric) over an exposure-only model.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score

df = pd.read_parquet(sys.argv[1])
df = df[df.src == "train"].copy()

# Banding for continuous factors; categorical factors are used as-is.
q = lambda s, n: pd.qcut(s, n, duplicates="drop").astype(str)
df["age_band"] = pd.cut(df.policyholder_age, [0, 25, 30, 35, 40, 50, 60, 70, 80, 120]).astype(str)
df["si_band"] = q(df.sum_insured, 10)
df["hvi"] = df.high_value_items.astype("Int64").astype(str)
df["security"] = df.security_level.fillna("MISSING")
df["missed_pay"] = df.missed_payments_12m.clip(upper=3).astype(str)
df["renewals"] = pd.cut(df.renewal_count, [-1, 0, 1, 3, 6, 10, 99]).astype(str)
df["quote_days"] = pd.cut(df.days_quote_to_inception, [-1, 0, 3, 7, 14, 21, 30]).astype(str)
df["cs_contacts"] = df.customer_service_contacts.clip(upper=4).astype(str)
df["wp_band"] = q(df.written_premium, 10)
df["household"] = df.household_size.clip(upper=6).astype(str)
df["prior_claims"] = df.prior_claims_5y.clip(upper=3).astype(str)
df["yrs_at_addr"] = pd.cut(df.years_at_address, [-1, 1, 3, 5, 10, 20, 99]).astype(str)
df["cust_since"] = df.customer_since_year.astype(str)
df["river_km"] = q(df.distance_to_river_km, 8)
df["elev"] = q(df.elevation_m, 8)
df["built"] = pd.cut(df.year_built, [0, 1900, 1945, 1970, 1990, 2005, 2100]).astype(str)
df["bedrooms_b"] = df.bedrooms.clip(upper=6).astype(str)
for w in ["hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
          "wildfire_index", "rainfall_mm", "mean_temperature_c"]:
    df[w + "_b"] = q(df[w], 8)
df["licence_pts"] = df.licence_points.clip(upper=6).astype(str)

FACTORS = {
    "policy": ["age_band", "si_band", "tenure", "security", "hvi", "accidental_damage_cover",
               "works_from_home", "lodgers", "bicycle_cover", "voluntary_excess", "channel",
               "payment_method", "missed_pay", "new_business", "renewals", "quote_days",
               "discount_applied", "cs_contacts", "wp_band"],
    "customer": ["occupation", "marital_status", "household", "credit_band", "prior_claims",
                 "licence_pts", "yrs_at_addr", "cust_since", "email_domain", "marketing_consent",
                 "document_preference", "previous_insurer", "campaign_code", "branch_id",
                 "marketing_segment"],
    "address": ["region", "urban_density", "flood_zone", "river_km", "elev", "soil_type", "built",
                "property_type", "roof_type", "basement", "trees_within_5m", "listed_building",
                "bedrooms_b", "energy_rating", "solar_panels"],
    "weather": [w + "_b" for w in ["hail_days", "wind_exposure_index", "freezing_days",
                                   "dry_summer_index", "wildfire_index", "rainfall_mm",
                                   "mean_temperature_c"]],
}

fit = df[df.yr <= 2023]
val = df[df.yr == 2024]
y = val.claim_cost.values
base_rate = fit.claim_cost.sum() / fit.exposure.sum()
d2_const = 0.0  # by definition: constant prediction equal to the mean
d2_expo = d2_tweedie_score(y, base_rate * val.exposure.values, power=1.5)


def oneway(col):
    g = df.groupby(col, dropna=False).agg(expo=("exposure", "sum"), n=("n_claims", "sum"),
                                         cost=("claim_cost", "sum"))
    g["freq_k"] = 1000 * g.n / g.expo
    g["sev"] = g.cost / g.n
    g["lc"] = g.cost / g.expo
    g["rel"] = g.lc / (df.claim_cost.sum() / df.exposure.sum())
    g["expo_pct"] = 100 * g.expo / g.expo.sum()
    # stability: relativity in 2021-22 vs 2023-24
    for lab, part in [("rel_2122", df[df.yr <= 2022]), ("rel_2324", df[df.yr >= 2023])]:
        p = part.groupby(col, dropna=False).agg(e=("exposure", "sum"), c=("claim_cost", "sum"))
        g[lab] = (p.c / p.e) / (part.claim_cost.sum() / part.exposure.sum())
    return g


def score(col):
    r = fit.groupby(col, dropna=False).agg(e=("exposure", "sum"), c=("claim_cost", "sum"))
    rate = (r.c / r.e).rename("rate")
    pred = val[[col]].join(rate, on=col)["rate"].fillna(base_rate).values * val.exposure.values
    return d2_tweedie_score(y, pred, power=1.5) - d2_expo


print(f"Out-of-time D2 (fit 2021-23, test 2024): constant={d2_const:.4f}  exposure-only={d2_expo:.4f}\n")
rows = []
for grp, cols in FACTORS.items():
    for c in cols:
        g = oneway(c)
        rows.append(dict(group=grp, factor=c, levels=len(g), d2_gain=score(c),
                         rel_min=g.rel.min(), rel_max=g.rel.max()))
rank = pd.DataFrame(rows).sort_values("d2_gain", ascending=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
print(rank.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

show = sys.argv[2].split(",") if len(sys.argv) > 2 else rank.factor.head(14).tolist()
for c in show:
    print(f"\n=== {c} ===")
    print(oneway(c)[["expo_pct", "n", "freq_k", "sev", "lc", "rel", "rel_2122", "rel_2324"]]
          .to_string(float_format=lambda v: f"{v:,.2f}"))
