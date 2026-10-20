"""Does splitting contents claims into separate components improve the expected-cost prediction?

Compares four GLM structures with identical rating factors, fitted on 2021-23 and scored on
2024 with Tweedie D2 (power 1.5, the competition metric):
  A  single Tweedie GLM on total claim cost
  B  sum of four per-peril Tweedie GLMs
  C  sum of four per-size-band Tweedie GLMs (<1k, 1k-5k, 5k-15k, 15k+)
  D  per-peril frequency (Poisson) x severity (Gamma), summed over perils
Excludes customer_service_contacts (D-04) and written_premium (D-05), as proposed.
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import d2_tweedie_score

work, claims, policies = sys.argv[1:4]

df = duckdb.sql(f"SELECT * FROM '{work}' WHERE src = 'train'").df()
clm = duckdb.sql(f"""
    SELECT p.row_id, c.peril, c.incurred_to_date - c.recoveries AS x
    FROM '{claims}' c JOIN '{policies}' p
      ON c.policy_id = p.policy_id AND c.loss_date BETWEEN p.period_start AND p.period_end
    WHERE c.line = 'contents' AND c.status <> 'repudiated'""").df()

BANDS = [("b1_lt1k", 0, 1_000), ("b2_1k5k", 1_000, 5_000), ("b3_5k15k", 5_000, 15_000),
         ("b4_15kplus", 15_000, np.inf)]
for name, lo, hi in BANDS:
    s = clm[(clm.x >= lo) & (clm.x < hi)].groupby("row_id").x.sum()
    df[name] = df.row_id.map(s).fillna(0.0)
PERILS = {"theft": "cost_theft", "escape_of_water": "cost_eow",
          "accidental_damage": "cost_ad", "flood": "cost_flood"}
for peril in PERILS:
    df["n_" + peril] = df.row_id.map(clm[clm.peril == peril].groupby("row_id").size()).fillna(0)

# Rating factors (categorical, missing kept as its own level - D-06).
cat = lambda s: s.astype("string").fillna("MISSING")
X = pd.DataFrame({
    "flood_zone": cat(df.flood_zone.astype("Int64")),
    "elev": cat(pd.qcut(df.elevation_m, 8, duplicates="drop").astype(str)),
    "river": cat(pd.qcut(df.distance_to_river_km, 8, duplicates="drop").astype(str).replace("nan", None)),
    "tenure": cat(df.tenure), "region": cat(df.region), "segment": cat(df.marketing_segment),
    "prior": cat(df.prior_claims_5y.clip(upper=3)), "missed": cat(df.missed_payments_12m.clip(upper=3)),
    "vol_xs": cat(df.voluntary_excess), "credit": cat(df.credit_band),
    "age": cat(pd.cut(df.policyholder_age, [0, 25, 30, 35, 40, 50, 60, 70, 80, 120]).astype(str)),
    "security": cat(df.security_level), "hvi": cat(df.high_value_items.astype("Int64")),
    "ad_cover": cat(df.accidental_damage_cover), "bike": cat(df.bicycle_cover),
    "wfh": cat(df.works_from_home), "lodgers": cat(df.lodgers),
    "urban": cat(df.urban_density), "prop_type": cat(df.property_type),
})
D = pd.get_dummies(X, drop_first=True, dtype=float)
D["log_si"] = np.log(df.sum_insured)
D = sm.add_constant(D)
off = np.log(df.exposure.values)
fit_m, val_m = (df.yr <= 2023).values, (df.yr == 2024).values
y_val = df.claim_cost.values[val_m]


def tweedie(target):
    m = sm.GLM(df[target].values[fit_m], D.values[fit_m], offset=off[fit_m],
               family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()))
    r = m.fit(maxiter=100)
    return np.exp(D.values[val_m] @ r.params + off[val_m])


def freq_sev(peril, cost_col):
    n = df["n_" + peril].values
    f = sm.GLM(n[fit_m], D.values[fit_m], offset=off[fit_m],
               family=sm.families.Poisson()).fit(maxiter=100)
    has = fit_m & (n > 0)
    sev = df[cost_col].values[has] / n[has]
    s = sm.GLM(sev, D.values[has], var_weights=n[has],
               family=sm.families.Gamma(link=sm.families.links.Log())).fit(maxiter=100)
    return np.exp(D.values[val_m] @ f.params + off[val_m]) * np.exp(D.values[val_m] @ s.params)


score = lambda p: d2_tweedie_score(y_val, p, power=1.5)
base = df.claim_cost.values[fit_m].sum() / df.exposure.values[fit_m].sum()
results = {"exposure only": base * df.exposure.values[val_m]}
results["A single Tweedie"] = tweedie("claim_cost")
results["B per-peril Tweedie"] = sum(tweedie(c) for c in PERILS.values())
results["C per-size-band Tweedie"] = sum(tweedie(b[0]) for b in BANDS)
results["D per-peril freq x sev"] = sum(freq_sev(p, c) for p, c in PERILS.items())

print(f"Fit 2021-23 ({fit_m.sum():,} rows), score 2024 ({val_m.sum():,} rows), {D.shape[1]} params per GLM\n")
print(f"{'model':<26}{'D2':>8}{'pred/actual':>13}")
for k, p in results.items():
    print(f"{k:<26}{score(p):>8.4f}{p.sum() / y_val.sum():>13.3f}")

# Do factors act differently on small and large claims? Compare per-band Tweedie relativities.
print("\nFactor relativities by size band (exp(coef)), selected levels:")
rel = {}
for name, _, _ in BANDS:
    r = sm.GLM(df[name].values[fit_m], D.values[fit_m], offset=off[fit_m],
               family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())).fit(maxiter=100)
    rel[name] = pd.Series(np.exp(r.params), index=D.columns)
rel = pd.DataFrame(rel)
keep = [c for c in D.columns if c.startswith(("flood_zone_", "tenure_", "vol_xs_", "prior_", "security_",
                                              "ad_cover_", "hvi_"))] + ["log_si"]
print(rel.loc[keep].round(2).to_string())
