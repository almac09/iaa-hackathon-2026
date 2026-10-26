"""Week 2: pricing the emerging market (audit A30 onwards).

The emerging market has two policy years (2023-24), a 2025 book that is mostly a broker portfolio of new business,
fewer rating factors (age bands instead of ages; no mileage, telematics or credit band; gender added), its own
regions and construction codes, and less data. The same SQL as Part 1 runs on both markets through a thin view
layer that gives each market the other's columns (CONFORM): the developed market gains age bands cut exactly as
the emerging ones, and the emerging market gains the columns it lacks as missing, with its construction codes
mapped to the developed ones. Week 1's cleaning rules (exposure from dates, sum-insured scale, declined claims
excluded, net cost) then apply unchanged.

  E1  Transfer test. Each line's model is refitted on the developed market (2021-24) with only the factors both
      markets share (the portable model, PORTABLE_DROP) and predicts the emerging 2023 and 2024 rows. Actual
      against expected by line, year and factor, and D2 against a constant-rate baseline, show what carries over
      as it is. The cost of portability is measured on the developed 2024 backtest.
  E2  Adaptation, fitted on emerging 2023 and scored on emerging 2024 (the one out-of-time year), line by line:
        C0   the portable developed model as it is;
        C1   C0 with the line's level reset to the deviance-optimal scale on 2023;
        C2a  C0 as an offset plus corrections for the factors that are new or shifted in this market (region, new
             business, age band, channel): a ridge-penalised Tweedie GLM, so each correction shrinks towards the
             developed relativity, with the penalty chosen by five-fold cross-validation on 2023, grouped by customer;
        C2b  the same with the line's main rating factors corrected too (numerics in five bands);
        C3   an emerging-only GLM on the C2b factors, with no developed prior (2024 carried forward by the developed
             trend);
        C4   C1 with claim-history credibility within the emerging market (the developed k per group);
        and, as a diagnostic only, C2a with gender.
      Rule, fixed before the run: C1 is the base, on the premise that the emerging level differs from the developed
      one (E1: A/E 0.71-0.80 for contents and motor in both years; buildings 1.19 then 0.70, so the premise holds
      for contents and motor only). C2a, C2b, C3 or C4 replaces it only if its 2024 gain over C1 is significant
      (paired customer bootstrap, 95% lower bound above zero) after Holm across them for the line. Amended after
      the first run (A30): cross-validation chose the largest penalty offered every time, so the grid was extended
      to 10, and C4 was added (which also tightens Holm's first threshold from 0.025/3 to 0.025/4); no decision
      changed. Gender is never adopted here: whether it may be used in this market is the owner's decision.
  E3  Development: which reporting pattern to use for the emerging target. The emerging market's own pattern rests
      on two loss years; the developed pattern on four. The 2023 claims are rebuilt as they stood at 2023-12-31
      (claims reported by then, valuation 2023-12-31) and developed with each pattern, then compared with the
      2024-12-31 view of 2023. Rule, fixed before the run: use the pattern whose projection of each line's 2023
      cost has the smaller absolute log error; keep the market's own where the two errors are within 2%. The case
      reserves are a single 2024-12-31 snapshot, so only reporting (IBNR) is tested, as in Part 1 (A17).
  E5  Three checks prompted by the data audit, run before the final prediction (A30):
      - Missing values: about 12% of several emerging fields are missing at random, against 2.5% in the developed
        market, so the developed model's missing-level relativities do not fit them. Each line's fields with at
        least 5% missing in emerging and under 5% in developed (flood zone excepted, whose missingness carries
        information) are filled with the developed median or mode before the developed model predicts. Rule, fixed
        before the run: fill a line's fields if, with the level re-estimated each way, D2 improves both fitting
        2023 and scoring 2024 and the reverse, and the mean gain over the two is significant.
      - The level reset scored in both directions, as a record (no decision rests on it).
      - New business: its actual-to-expected relative to renewals in the developed market (in sample, 2021-24, and
        by window) and in the emerging market (2023-24), blended by credibility Z = n / (n + 1,000) on emerging
        new-business claims (K = 1,000 is an assumption), for a scenario file. Not adopted by this runner: the
        2024 holdout is 9% new business and cannot test it, so whether to load the 77% new-business 2025 book is
        the owner's decision.
  E4  The 2025 prediction, as E2, E3 and E5 decided: each line's portable developed model, projected to 2025 with
      the developed trend, with missing values filled where E5 adopted it, times the line's level, the
      deviance-optimal scale of the emerging 2023 and 2024 actuals on the target developed with the pattern E3
      chose (EMERGING_PATTERN). Pooling the two years is a judgement for buildings, whose 2023 carried a
      fire-frequency excess concentrated in region E02 (49 fires against 21 in 2024): it weights such a year as
      about one in two. The level with 2023 fire at the 2024 rate is reported beside it. No factor corrections and
      no claim history, which E2 rejected. Reported beside it:
      - the single-year scales and a 90% range for each line's 2025 level, combining as independent: the customer-
        bootstrap error of the pooled scale; year noise, both inside the two-year estimate and in 2025 itself, at
        the larger of the developed market's year-to-year noise and the emerging between-year dispersion; the
        developed Mack development error for 2023 and 2024, scaled up for the smaller emerging book by the square
        root of the ratio of claim counts (conservative: it scales the pattern error too); and trend error, taken
        as half the developed trend a year over the 1.5 years from the middle of 2023-24 (an assumption). New
        business is a named sensitivity, not part of the range;
      - the new-business sensitivity, weighted by expected cost, not rows.
      Three files: the adopted prediction; the portable model as it is, for a leaderboard comparison; and the
      adopted prediction with E5's blended new-business loading, the level re-solved on 2023-24, as a scenario.

Usage (from the pack root, with the emerging files in data/emerging/):
  python python/emerging.py --tests E1 --run-label "A30 emerging: E1 transfer test"
  E4 reads E5's decisions from its evidence, so E5 runs first.
"""
from __future__ import annotations

import copy
import datetime as dt
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import glm_lines  # noqa: E402
import models  # noqa: E402
import run_glm_lines as rgl  # noqa: E402
import significance  # noqa: E402
from gap_tests import settings  # noqa: E402
from run_xgb import write_csv  # noqa: E402

RUNNER = "emerging"
OUT = common.PACK_ROOT / "audit" / "evidence" / "emerging"
P = 1.5
EMERGING_WINDOWS = {"tune": ([2023], 2024), "final": ([2023, 2024], 2025)}
MARKETS = {
    "developed": {"data_dir": "data", "expected_inputs": "audit/expected_inputs.sha256", "windows": rgl.WINDOWS},
    "emerging": {"data_dir": "data/emerging", "expected_inputs": "audit/expected_inputs_emerging.sha256",
                 "windows": EMERGING_WINDOWS},
}
# Only the SQL that builds the modelling tables; sql/01, 02 and 04-07 write evidence for the contents GLM pack.
SQL_FILES = ["sql/03_build_contents.sql", "sql/08_features.sql"]
AGE_BAND = ("CASE WHEN {x} IS NULL THEN NULL WHEN {x} <= 24 THEN '17-24' WHEN {x} <= 34 THEN '25-34' "
            "WHEN {x} <= 49 THEN '35-49' WHEN {x} <= 64 THEN '50-64' ELSE '65+' END")
# Emerging construction codes against the developed ones; the shares agree (brick 70% against 74%, mixed 10% against
# non-standard 11%, stone 8% against 9%, wood 12% against timber 13%).
CONSTRUCTION = "CASE upper(construction_type) WHEN 'BRK' THEN 'brick' WHEN 'STN' THEN 'stone' " \
               "WHEN 'WOD' THEN 'timber' WHEN 'MIX' THEN 'non_standard' ELSE construction_type END"
CONFORM = {
    "developed": {
        "policies_motor": f"*, {AGE_BAND.format(x='driver_age')} AS driver_age_band",
        "test_motor": f"*, {AGE_BAND.format(x='driver_age')} AS driver_age_band",
        "policies_buildings": f"*, {AGE_BAND.format(x='policyholder_age')} AS policyholder_age_band",
        "test_buildings": f"*, {AGE_BAND.format(x='policyholder_age')} AS policyholder_age_band",
        "policies_contents": f"*, {AGE_BAND.format(x='policyholder_age')} AS policyholder_age_band",
        "test_contents": f"*, {AGE_BAND.format(x='policyholder_age')} AS policyholder_age_band",
        "customers": "*, NULL::VARCHAR AS gender",
    },
    "emerging": {
        "policies_motor": "*, NULL::BIGINT AS driver_age, NULL::DOUBLE AS annual_mileage, "
                          "NULL::BOOLEAN AS telematics_opt_in, NULL::DOUBLE AS telematics_score",
        "test_motor": "*, NULL::BIGINT AS driver_age, NULL::DOUBLE AS annual_mileage, "
                      "NULL::BOOLEAN AS telematics_opt_in, NULL::DOUBLE AS telematics_score",
        "policies_buildings": f"* REPLACE ({CONSTRUCTION} AS construction_type), NULL::BIGINT AS policyholder_age",
        "test_buildings": f"* REPLACE ({CONSTRUCTION} AS construction_type), NULL::BIGINT AS policyholder_age",
        "policies_contents": "*, NULL::BIGINT AS policyholder_age",
        "test_contents": "*, NULL::BIGINT AS policyholder_age",
        "customers": "*, NULL::VARCHAR AS credit_band, NULL::DATE AS date_of_birth",
    },
}
# Factors the portable model cannot use: absent from the emerging market, or whose levels do not carry over (under
# 90% of emerging rows have a level the developed market has: region 0%, occupation 56%).
PORTABLE_DROP = {"region", "credit_band", "occupation", "policyholder_age", "driver_age", "annual_mileage",
                 "telematics_score", "telematics_opt_in", "f_region", "f_credit", "f_age_band"}
BAND = {"buildings": "policyholder_age_band", "contents": "policyholder_age_band", "motor": "driver_age_band"}


def portable(cfg: dict) -> dict:
    """The production feature lists without PORTABLE_DROP, each line's age band added, and contents' standardised
    log sum insured (standardised within each market by sql/08) replaced by the plain log."""
    out = copy.deepcopy(cfg)
    keep = lambda cols: [c for c in rgl.resolve(out, cols) if c not in PORTABLE_DROP]  # noqa: E731
    for name, lm in out["lines"]["models"].items():
        lm["numeric"], lm["categorical"] = keep(lm["numeric"]), keep(lm["categorical"]) + [BAND[name]]
    for name, m in out["glm_lines"]["models"].items():
        if m["small"] == "glm":
            m["numeric"], m["categorical"] = keep(m["numeric"]), keep(m["categorical"]) + [BAND[name]]
        if "excess" in m:
            m["excess"]["numeric"] = keep(m["excess"]["numeric"])
            m["excess"]["categorical"] = keep(m["excess"]["categorical"]) + ([BAND[name]] if name == "motor" else [])
        if isinstance(m.get("freq"), dict):
            for key in ("numeric", "categorical"):
                m["freq"][key] = keep(m["freq"][key])
    out["glm_lines"]["models"]["contents"]["continuous"] = ["log_si"]
    return out


def market_cfg(cfg: dict, market: str, external_pattern: bool = False) -> dict:
    out = copy.deepcopy(cfg)
    out["paths"]["data_dir"] = MARKETS[market]["data_dir"]
    out["paths"]["expected_inputs"] = MARKETS[market]["expected_inputs"]
    out["sql_files"] = SQL_FILES
    out["sql_variables"]["use_external_dev_pattern"] = external_pattern
    return out


def conform(con, cfg: dict, market: str) -> None:
    """Replace the raw table views with the market's conforming views (CONFORM)."""
    data_dir = common.PACK_ROOT / cfg["paths"]["data_dir"]
    for table, select in CONFORM[market].items():
        path = (data_dir / f"{table}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT {select} FROM read_parquet('{path}')")


def load_market(cfg: dict, market: str, rec: dict, external_pattern: pd.DataFrame | None = None):
    """Every Line of a market, built by the production SQL on the conforming views, with the market's windows.
    `external_pattern` is the reporting pattern to use instead of the market's own (sql/10)."""
    mcfg = market_cfg(cfg, market, external_pattern is not None)
    rec[f"inputs_sha256_{market}"] = common.verify_inputs(mcfg)
    con = common.open_duckdb(mcfg)
    conform(con, mcfg, market)
    if external_pattern is not None:
        con.register("external_dev_pattern", external_pattern)
    lines = rgl.prepare(con, mcfg, rec.setdefault(f"provenance_{market}", {}), windows=MARKETS[market]["windows"])[0]
    c = lines["contents"]
    # sql/03 selects the contents columns explicitly, so the age band and gender are joined on here.
    band = con.execute("""SELECT row_id, policyholder_age_band FROM policies_contents
                          UNION ALL SELECT row_id, policyholder_age_band FROM test_contents""").df()
    gender = con.execute("SELECT customer_id, gender FROM customers").df().drop_duplicates("customer_id")
    df = c.df.merge(band, on="row_id", how="left").merge(gender, on="customer_id", how="left")
    assert len(df) == len(c.df) and (df.row_id.to_numpy() == c.df.row_id.to_numpy()).all()
    c.df = df.assign(log_si=np.log(df.sum_insured.clip(lower=1)))
    for line in lines.values():
        line.train, line.test = line.df[line.df.src == "train"], line.df[line.df.src == "test"]
    return lines, con


def d2_rate(y: np.ndarray, mu: np.ndarray, e: np.ndarray) -> float:
    """Tweedie D2 against a constant-rate baseline (the leaderboard's), on these rows."""
    base = y.sum() / e.sum() * e
    return 1 - models.tweedie_deviance(y, mu, P).sum() / models.tweedie_deviance(y, base, P).sum()


def transfer_rows(name: str, y, mu, e, year: int, label: str) -> dict:
    scaled = mu * y.sum() / mu.sum()
    return {"line": name, "year": year, "model": label, "rows": len(y), "claims_rows": int((y > 0).sum()),
            "actual": y.sum(), "expected": mu.sum(), "ae": y.sum() / mu.sum(), "d2_rate": d2_rate(y, mu, e),
            "d2_rate_level_matched": d2_rate(y, scaled, e)}


FACTORS = {"buildings": ["new_business", "policyholder_age_band", "region", "construction_type", "flood_zone",
                         "channel", "urban_density", "soil_type", "property_type"],
           "contents": ["new_business", "policyholder_age_band", "region", "f_flood_zone", "channel", "f_urban",
                        "f_tenure", "f_prop_type", "f_segment"],
           "motor": ["new_business", "driver_age_band", "region", "cover_type", "channel", "urban_density",
                     "vehicle_use", "marketing_segment", "payment_method"]}


def by_factor(name: str, v: pd.DataFrame, y, mu, year: int) -> list[dict]:
    """Actual against expected by level, with the expected rescaled to the year's actual total (shape only)."""
    m = mu * y.sum() / mu.sum()
    out = []
    for f in FACTORS[name]:
        if f not in v:
            continue
        g = pd.DataFrame({"level": v[f].astype("string").fillna("MISSING").to_numpy(), "a": y, "e": m,
                          "claims": (y > 0).astype(int), "exposure": v.exposure_plain.to_numpy()}).groupby("level").sum()
        for lv, r in g.iterrows():
            out.append({"line": name, "year": year, "factor": f, "level": lv, "exposure": r.exposure,
                        "claims": int(r.claims), "actual": r.a, "expected": r.e, "ae": r.a / r.e if r.e else np.nan})
    return out


class Context:
    """Both markets built once per run: the developed market and the emerging market on the portable feature set,
    and each line's portable model fitted on all developed years."""

    def __init__(self, cfg: dict, rec: dict):
        self.cfg, self.rec, self.pcfg = cfg, rec, portable(cfg)
        self.summary = pd.read_csv(common.PACK_ROOT / "audit/evidence/glm_lines/ev_g6_summary.csv")
        self.dev, self.dev_con = load_market(self.pcfg, "developed", rec)
        self.emg, self.emg_con = load_market(self.pcfg, "emerging", rec)
        self.settings, self.trees, self.models, self._fill, self._emg_devpat = {}, {}, {}, {}, None
        for name, line in self.dev.items():
            s = settings(line, self.summary)
            self.settings[name] = s
            self.trees[name] = line.tune_trees(s["threshold"]) if line.spec["small"] == "xgb" else None
            line.use_window("final")
            self.models[name] = line.build(rgl.WINDOWS["final"][0], s["threshold"], s["alpha"], self.trees[name])

    def base(self, name: str, d: pd.DataFrame, year: int, fill: bool = False) -> np.ndarray:
        """The portable developed model's prediction for emerging rows of `year` (C0), with the E5 missing values
        filled first if `fill`."""
        return self.models[name].predict(self.filled(name, d) if fill else d, year=year)

    def line(self, name: str) -> rgl.Line:
        """The emerging line on the target developed with the reporting pattern E3 chose (EMERGING_PATTERN)."""
        if EMERGING_PATTERN[name] == "own":
            return self.emg[name]
        if self._emg_devpat is None:
            pattern = self.dev_con.execute("SELECT * FROM dev_pattern").df()
            self._emg_devpat = load_market(self.pcfg, "emerging", self.rec, external_pattern=pattern)[0]
        return self._emg_devpat[name]

    def fill_values(self, name: str) -> dict:
        """E5: fields the line's portable models use that are missing in at least 5% of emerging training rows but
        under 5% of developed ones (flood zone excepted), with the developed median or mode to fill them with."""
        if name not in self._fill:
            dev, emg, out = self.dev[name].train, self.emg[name].train, {}
            for col in features_used(self.pcfg, name):
                if col not in dev or col not in emg or "flood_zone" in col:
                    continue
                miss = lambda s: (s.isna() | (s.astype("string") == "MISSING")).mean()  # noqa: E731
                if miss(emg[col]) >= 0.05 and miss(dev[col]) < 0.05:
                    known = dev[col][~(dev[col].isna() | (dev[col].astype("string") == "MISSING"))]
                    out[col] = fill_value(known)
            self._fill[name] = out
        return self._fill[name]

    def filled(self, name: str, d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for col, value in self.fill_values(name).items():
            gap = d[col].isna() | (d[col].astype("string") == "MISSING")
            d.loc[gap, col] = value
        return d


def e1(cfg: dict, rec: dict, ctx: Context) -> dict:
    """Transfer test (see the module docstring)."""
    full, _ = load_market(cfg, "developed", rec)
    rows, factors, cost = [], [], []
    for name in ("buildings", "contents", "motor"):
        line, s = ctx.dev[name], ctx.settings[name]
        no_hist = {g: np.inf for g in s["ks"]}
        # Cost of portability on the developed 2024 backtest (fit 2021-23), both without claim history.
        bt_years, bt_year = rgl.WINDOWS["backtest"]
        for label, ln in (("production features", full[name]), ("portable features", line)):
            ln.use_window("backtest")
            tr = ln.tune_trees(s["threshold"]) if ln.spec["small"] == "xgb" else None
            m = ln.build(bt_years, s["threshold"], s["alpha"], tr)
            v = ln.rows([bt_year])
            mu = ln.predict(m, "backtest", no_hist, v, year=bt_year)
            cost.append(transfer_rows(name, v.y_full.to_numpy(), mu, v.exposure_plain.to_numpy(), bt_year, label)
                        | {"market": "developed", "trees": tr[0] if tr else None})
        for year in (2023, 2024):
            v = ctx.emg[name].rows([year])
            mu = ctx.base(name, v, year)
            y, e = v.y_full.to_numpy(), v.exposure_plain.to_numpy()
            rows.append(transfer_rows(name, y, mu, e, year, "portable developed model, as is") | {"market": "emerging"})
            factors += by_factor(name, v, y, mu, year)
        print(f"  {name}: " + "; ".join(f"{r['year']} A/E {r['ae']:.3f}, D2 {r['d2_rate']:.4f} "
                                         f"(level-matched {r['d2_rate_level_matched']:.4f})"
                                         for r in rows if r["line"] == name), flush=True)
    return {"ev_w1_transfer.csv": pd.DataFrame(rows), "ev_w1_transfer_by_factor.csv": pd.DataFrame(factors),
            "ev_w1_portability_cost.csv": pd.DataFrame(cost)}


TARGETED = {"buildings": ["region", "new_business", "policyholder_age_band", "channel"],
            "contents": ["region", "new_business", "policyholder_age_band", "channel"],
            "motor": ["region", "new_business", "driver_age_band", "channel"]}
MAIN = {  # main rating factors (numeric, categorical), corrected in C2b and used alone in C3
    "buildings": (["flood_zone", "sum_insured", "elevation_m", "year_built", "distance_to_river_km",
                   "dry_summer_index", "freezing_days", "rainfall_mm"],
                  ["construction_type", "soil_type", "occupancy", "energy_rating", "urban_density"]),
    "contents": (["elevation_m", "distance_to_river_km", "log_si"],
                 ["f_flood_zone", "f_tenure", "f_prior_claims", "f_vol_xs", "f_security", "f_hvi", "f_ad_cover",
                  "f_urban", "f_prop_type"]),
    "motor": (["ncb_years", "years_licensed", "vehicle_age", "vehicle_value_group", "voluntary_excess",
               "prior_claims_5y"],
              ["cover_type", "urban_density", "vehicle_use", "payment_method", "marketing_segment"]),
}
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FOLDS = 5


class Correction:
    """A Tweedie GLM for log(actual / base) on banded factors: with base the developed prediction it is a set of
    credibility corrections that shrink towards the developed relativities as the ridge penalty grows; with base
    the exposure it is a stand-alone rate model."""

    def __init__(self, numeric: list[str], categorical: list[str], alpha: float):
        self.numeric, self.categorical, self.alpha = numeric, categorical, alpha

    def _x(self, d: pd.DataFrame) -> np.ndarray:
        f = self.spec.factors(d)
        return models.design_matrix(f, self.levels, [])[0]

    def fit(self, d: pd.DataFrame, y: np.ndarray, base: np.ndarray) -> "Correction":
        self.spec = glm_lines.FactorSpec(d, self.numeric, self.categorical, n_bands=5)
        self.levels = models.factor_levels(self.spec.factors(d), self.numeric + self.categorical)
        self.model = models.fit_tweedie(self._x(d), y, base, P, self.alpha)
        return self

    def predict(self, d: pd.DataFrame, base: np.ndarray) -> np.ndarray:
        return self.model.predict(self._x(d)) * base


def folds_by_customer(customers, k: int = FOLDS, seed: int = 2026) -> np.ndarray:
    codes, uniq = pd.factorize(pd.Series(customers))
    return np.random.default_rng(seed).permutation(len(uniq))[codes] % k


def choose_alpha(d: pd.DataFrame, y: np.ndarray, base: np.ndarray, numeric, categorical) -> tuple[float, list]:
    """The ridge penalty with the smallest out-of-fold deviance, folds grouped by customer."""
    fold, cv = folds_by_customer(d.customer_id.to_numpy()), []
    for a in ALPHAS:
        dev = 0.0
        for k in range(FOLDS):
            tr, te = fold != k, fold == k
            m = Correction(numeric, categorical, a).fit(d[tr], y[tr], base[tr])
            dev += float(models.tweedie_deviance(y[te], m.predict(d[te], base[te]), P).sum())
        cv.append(dev)
    return ALPHAS[int(np.argmin(cv))], cv


def deviance_scale(y: np.ndarray, mu: np.ndarray) -> float:
    return float((y * mu ** (1 - P)).sum() / (mu ** (2 - P)).sum())


def e2(cfg: dict, rec: dict, ctx: Context) -> dict:
    """Adaptation candidates fitted on 2023 and scored on 2024 (see the module docstring)."""
    bs = cfg["glm_lines"]["bootstrap"]
    rows = []
    for name in ("buildings", "contents", "motor"):
        line = ctx.emg[name]
        d23, d24 = line.rows([2023]), line.rows([2024])
        y23, y24, e24 = d23.y_full.to_numpy(), d24.y_full.to_numpy(), d24.exposure_plain.to_numpy()
        b23, b24 = ctx.base(name, d23, 2023), ctx.base(name, d24, 2024)
        trend = float(np.exp(ctx.models[name].small.slope))
        k = deviance_scale(y23, b23)
        num, cat = MAIN[name]
        specs = {"C2a": ([], TARGETED[name]), "C2b": (num, cat + TARGETED[name]),
                 "C2a + gender (diagnostic)": ([], TARGETED[name] + ["gender"])}
        preds, alphas = {"C0": b24, "C1": b24 * k}, {}
        for label, (n_, c_) in specs.items():
            a, _ = choose_alpha(d23, y23, b23, n_, c_)
            preds[label], alphas[label] = Correction(n_, c_, a).fit(d23, y23, b23).predict(d24, b24), a
        e23 = d23.exposure_plain.to_numpy()
        a, _ = choose_alpha(d23, y23, e23, num, cat + TARGETED[name])
        preds["C3"], alphas["C3"] = Correction(num, cat + TARGETED[name], a).fit(d23, y23, e23).predict(d24, e24) * trend, a
        # C4: the developed model with emerging claim history (E from this market's 2023 frequency model), rebalanced
        # to keep the 2023 total, then the C1 level.
        preds["C4"] = line.predict(ctx.models[name], "tune", ctx.settings[name]["ks"], d24, year=2024) * k
        clusters = d24[bs["cluster"]].to_numpy()
        tested = []
        for label, mu in preds.items():
            r = {"line": name, "candidate": label, "alpha": alphas.get(label), "level_k_2023": k if label == "C1" else None,
                 "ae_2024": y24.sum() / mu.sum(), "d2_rate_2024": d2_rate(y24, mu, e24)}
            if label != "C1":
                diff = significance.paired_cluster_bootstrap(y24, mu, preds["C1"], clusters, P, bs["n"], bs["seed"])
                r |= {f"vs_C1_{k_}": v for k_, v in significance.summarise(
                    diff, models.d2_tweedie(y24, mu, P) - models.d2_tweedie(y24, preds["C1"], P)).items()}
            rows.append(r)
            if label in ("C2a", "C2b", "C3", "C4"):
                tested.append(r)
        # Holm across C2a, C2b, C3 and C4 at one-sided 2.5%, on the bootstrap share of resamples not better than C1.
        passed = True
        for rank, r in enumerate(sorted(tested, key=lambda r: r["vs_C1_share_not_better"])):
            passed = passed and r["vs_C1_share_not_better"] < 0.025 / (len(tested) - rank)
            r["passes_holm"] = passed
            r["adopt"] = bool(passed and r["vs_C1_ci_low_95"] > 0)
        print(f"  {name}: " + "; ".join(f"{r['candidate']} D2 {r['d2_rate_2024']:.4f} A/E {r['ae_2024']:.3f}"
                                         for r in rows if r["line"] == name), flush=True)
    return {"ev_w2_candidates.csv": pd.DataFrame(rows)}


LINES_SQL = ["sql/09_build_lines.sql", "sql/10_development.sql"]


def developed_cost(cfg: dict, market: str, valuation: str, pattern: pd.DataFrame | None) -> pd.DataFrame:
    """Each line's and peril's developed cost by loss year, with claims as reported by `valuation` and the
    valuation date set to it; `pattern` replaces the market's own reporting pattern."""
    mcfg = market_cfg(cfg, market, pattern is not None)
    mcfg["sql_variables"]["valuation_date"] = valuation
    con = common.open_duckdb(mcfg)
    conform(con, mcfg, market)
    path = (common.PACK_ROOT / mcfg["paths"]["data_dir"] / "claims.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE VIEW claims AS SELECT * FROM read_parquet('{path}') "
                f"WHERE report_date <= DATE '{valuation}'")
    if pattern is not None:
        con.register("external_dev_pattern", pattern)
    common.run_sql_files(con, mcfg)
    for rel in LINES_SQL:
        con.execute((common.PACK_ROOT / rel).read_text())
    return con.execute("""SELECT line, peril, year(loss_date) AS loss_year, sum(net_cost) AS reported,
                                 sum(net_cost_dev) AS developed FROM line_claims_dev GROUP BY ALL""").df()


def e3(cfg: dict, rec: dict, ctx: Context) -> dict:
    """Development test (see the module docstring)."""
    pattern = ctx.dev_con.execute("SELECT * FROM dev_pattern").df()
    later = developed_cost(ctx.pcfg, "emerging", "2024-12-31", None)
    runs = {"own": developed_cost(ctx.pcfg, "emerging", "2023-12-31", None),
            "developed": developed_cost(ctx.pcfg, "emerging", "2023-12-31", pattern)}
    rows = []
    for level in (["line"], ["line", "peril"]):
        ref = later[later.loss_year == 2023].groupby(level)[["reported", "developed"]].sum()
        for name, df in runs.items():
            g = df[df.loss_year == 2023].groupby(level)[["reported", "developed"]].sum()
            j = g.join(ref, rsuffix="_2024_view")
            j["pattern"], j["ratio_to_2024_view"] = name, j.developed / j.developed_2024_view
            rows += j.reset_index().assign(level="+".join(level)).to_dict("records")
    out = pd.DataFrame(rows)
    by_line = out[out.level == "line"].pivot(index="line", columns="pattern", values="ratio_to_2024_view")
    choice = {}
    for line_name, r in by_line.iterrows():
        err = {k: abs(np.log(v)) for k, v in r.items()}
        choice[line_name] = "own" if abs(err["own"] - err["developed"]) < np.log(1.02) else min(err, key=err.get)
    out["chosen_for_line"] = out.line.map(choice)
    lags = []
    for market, con in (("developed", ctx.dev_con), ("emerging", ctx.emg_con)):
        lags.append(con.execute("SELECT * FROM ev_10_report_lag_pattern").df().assign(market=market))
    print("  as-at 2023-12-31 projection of 2023 against the 2024-12-31 view: "
          + "; ".join(f"{k}: own {v['own']:.3f}, developed {v['developed']:.3f} -> {choice[k]}"
                      for k, v in by_line.iterrows()), flush=True)
    return {"ev_w3_development_asat.csv": out, "ev_w3_report_lag_patterns.csv": pd.concat(lags)}


EMERGING_PATTERN = {"buildings": "own", "contents": "own", "motor": "developed"}  # E3, A30
SUBMISSIONS = common.PACK_ROOT / "work" / "submissions"
NB_K = 1000.0  # E5: credibility constant for new business, in emerging new-business claims (an assumption)


def fill_value(known: pd.Series):
    """The value to fill a gap with: the median of a numeric field, the most common value of any other (booleans
    included, which pandas counts as numeric)."""
    if pd.api.types.is_bool_dtype(known):
        return bool(known.astype(bool).mode().iloc[0])
    if pd.api.types.is_numeric_dtype(known):
        return float(known.median())
    return known.astype("string").mode().iloc[0]


def features_used(cfg: dict, name: str) -> list[str]:
    """Every input column of a line's portable models: the small-claim model (GLM or XGBoost) and the large-claim
    GLM."""
    g = cfg["glm_lines"]["models"][name]
    cols = []
    if g["small"] == "glm":
        cols += rgl.resolve(cfg, g["numeric"]) + rgl.resolve(cfg, g["categorical"]) + rgl.resolve(cfg, g.get("continuous", []))
    else:
        lm = cfg["lines"]["models"][name]
        cols += lm["numeric"] + lm["categorical"]
    if "excess" in g:
        e = g["excess"]
        cols += e["numeric"] + rgl.resolve(cfg, e["categorical"]) + rgl.resolve(cfg, e.get("continuous", []))
    return list(dict.fromkeys(cols))


def boot_d2_gain(y, alt, base, e, clusters, bs) -> dict:
    diff = significance.paired_cluster_bootstrap(y, alt, base, clusters, P, bs["n"], bs["seed"])
    return significance.summarise(diff, models.d2_tweedie(y, alt, P) - models.d2_tweedie(y, base, P))


def nb_ratio(y, m, nb, w=None) -> float:
    """Actual-to-expected of new business relative to renewals."""
    w = np.ones(len(y)) if w is None else w
    return ((w * y)[nb].sum() / (w * m)[nb].sum()) / ((w * y)[~nb].sum() / (w * m)[~nb].sum())


def boot_ratio(y, m, nb, customers, n, rng) -> list[float]:
    codes, uniq = pd.factorize(pd.Series(customers))
    return [nb_ratio(y, m, nb, np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes])
            for _ in range(n)]


def e5(cfg: dict, rec: dict, ctx: Context) -> dict:
    """Checks prompted by the data audit (see the module docstring)."""
    bs, rng = cfg["glm_lines"]["bootstrap"], np.random.default_rng(cfg["glm_lines"]["bootstrap"]["seed"])
    imp, rev, nbr, fills = [], [], [], []
    for name in ("buildings", "contents", "motor"):
        line = ctx.line(name)
        fills += [{"line": name, "field": c, "fill_value": v} for c, v in ctx.fill_values(name).items()]
        gains = []
        for fit_y, score_y in ((2023, 2024), (2024, 2023)):
            df, ds = line.rows([fit_y]), line.rows([score_y])
            yf, ys, es = df.y_full.to_numpy(), ds.y_full.to_numpy(), ds.exposure_plain.to_numpy()
            mu = {}
            for variant, fill in (("as is", False), ("filled", True)):
                k = deviance_scale(yf, ctx.base(name, df, fit_y, fill))
                mu[variant] = ctx.base(name, ds, score_y, fill) * k
                if variant == "as is":
                    b0 = ctx.base(name, ds, score_y)
                    rev.append({"line": name, "fit_year": fit_y, "score_year": score_y, "level_k": k,
                                "d2_rate_as_is": d2_rate(ys, b0, es), "d2_rate_level_reset": d2_rate(ys, mu[variant], es),
                                **{f"reset_vs_as_is_{a}": b for a, b in boot_d2_gain(
                                    ys, mu[variant], b0, es, ds[bs["cluster"]].to_numpy(), bs).items()}})
            g = boot_d2_gain(ys, mu["filled"], mu["as is"], es, ds[bs["cluster"]].to_numpy(), bs)
            gains.append(g)
            imp.append({"line": name, "fit_year": fit_y, "score_year": score_y, "fields": len(ctx.fill_values(name)),
                        "d2_rate_as_is": d2_rate(ys, mu["as is"], es), "d2_rate_filled": d2_rate(ys, mu["filled"], es),
                        **{f"filled_vs_as_is_{a}": b for a, b in g.items()}})
        mean, lo, hi = significance.two_year_interval([g["d2_gain"] for g in gains], [g["ci_low_95"] for g in gains],
                                                      [g["ci_high_95"] for g in gains])
        adopt = bool(all(g["d2_gain"] > 0 for g in gains) and lo > 0)
        for r in imp:
            if r["line"] == name:
                r.update({"two_way_mean": mean, "two_way_ci_low_95": lo, "two_way_ci_high_95": hi, "adopt": adopt})
        # New business: developed in sample (2021-24 and by window), emerging 2023-24 under the level reset.
        dl = ctx.dev[name]
        rows = {}
        for label, years in (("developed 2021-24", [2021, 2022, 2023, 2024]), ("developed 2021-22", [2021, 2022]),
                             ("developed 2023-24", [2023, 2024])):
            d = dl.rows(years)
            m = ctx.models[name].predict(d)
            nb = d.new_business.astype(bool).to_numpy()
            b = boot_ratio(d.y_full.to_numpy(), m, nb, d.customer_id.to_numpy(), bs["n"] // 5, rng)
            rows[label] = (nb_ratio(d.y_full.to_numpy(), m, nb), np.percentile(b, 2.5), np.percentile(b, 97.5),
                           int((d.y_full.to_numpy()[nb] > 0).sum()))
        fit = pd.concat([line.rows([2023]), line.rows([2024])])
        mu0 = np.concatenate([ctx.base(name, line.rows([yr]), yr, adopt) for yr in (2023, 2024)])
        y, nb = fit.y_full.to_numpy(), fit.new_business.astype(bool).to_numpy()
        b = boot_ratio(y, mu0, nb, fit.customer_id.to_numpy(), bs["n"], rng)
        r_emg, n_emg = nb_ratio(y, mu0, nb), int((y[nb] > 0).sum())
        rows["emerging 2023-24"] = (r_emg, np.percentile(b, 2.5), np.percentile(b, 97.5), n_emg)
        z = n_emg / (n_emg + NB_K)
        blended = float(np.exp(z * np.log(r_emg) + (1 - z) * np.log(rows["developed 2021-24"][0])))
        for label, (r, lo_, hi_, n) in rows.items():
            nbr.append({"line": name, "source": label, "nb_ae_relative_to_renewal": r, "ci_low_95": lo_,
                        "ci_high_95": hi_, "nb_claims": n})
        nbr.append({"line": name, "source": "blended", "nb_ae_relative_to_renewal": blended, "credibility_z": z,
                    "credibility_k": NB_K, "nb_claims": n_emg})
        print(f"  {name}: fill {len(ctx.fill_values(name))} fields -> {'adopt' if adopt else 'keep as is'} "
              f"(two-way mean {mean:+.5f}, {lo:+.5f} to {hi:+.5f}); new business developed "
              f"{rows['developed 2021-24'][0]:.3f}, emerging {r_emg:.3f}, blended {blended:.3f} (Z {z:.2f})", flush=True)
    return {"ev_w5_missing_values.csv": pd.DataFrame(imp), "ev_w5_fill_values.csv": pd.DataFrame(fills),
            "ev_w5_level_reset_both_ways.csv": pd.DataFrame(rev), "ev_w5_new_business.csv": pd.DataFrame(nbr)}


def e4(cfg: dict, rec: dict, ctx: Context) -> dict:
    """The 2025 prediction (see the module docstring)."""
    bs = cfg["glm_lines"]["bootstrap"]
    w5_imp = pd.read_csv(OUT / "ev_w5_missing_values.csv").groupby("line").adopt.first().to_dict()
    w5_nb = pd.read_csv(OUT / "ev_w5_new_business.csv")
    rec["e5_inputs_sha256"] = {f: common.sha256_file(OUT / f) for f in ("ev_w5_missing_values.csv", "ev_w5_new_business.csv")}
    mack = pd.read_csv(common.PACK_ROOT / cfg["glm_lines"]["level"]["mack_evidence"])
    sigma_dev = pd.read_csv(common.PACK_ROOT / "audit/evidence/glm_lines/ev_g10_level_uncertainty.csv").set_index("line")
    summary = ctx.summary.set_index("line")
    rng = np.random.default_rng(bs["seed"])
    level_rows, nb_rows, preds = [], [], {}
    for name in ("buildings", "contents", "motor"):
        line, fill = ctx.line(name), bool(w5_imp[name])
        years = (2023, 2024)
        parts = {yr: line.rows([yr]) for yr in years}
        mus = {yr: ctx.base(name, parts[yr], yr, fill) for yr in years}
        fit, mu0 = pd.concat(parts.values()), np.concatenate(list(mus.values()))
        y = fit.y_full.to_numpy()
        k = deviance_scale(y, mu0)
        codes, uniq = pd.factorize(fit.customer_id)

        def boot_k(yy, mm, cc, nn):
            out = []
            for _ in range(bs["n"]):
                w = np.bincount(rng.integers(0, nn, nn), minlength=nn)[cc]
                out.append(float((w * yy * mm ** (1 - P)).sum() / (w * mm ** (2 - P)).sum()))
            return np.array(out)
        ks = boot_k(y, mu0, codes, len(uniq))
        cv_k = float(ks.std() / ks.mean())
        k_year, cv_year = {}, {}
        for yr in years:
            yy, c_, u_ = parts[yr].y_full.to_numpy(), *pd.factorize(parts[yr].customer_id)
            k_year[yr] = deviance_scale(yy, mus[yr])
            b = boot_k(yy, mus[yr], c_, len(u_))
            cv_year[yr] = float(b.std() / b.mean())
        w = {yr: float((mus[yr] ** (2 - P)).sum() / (mu0 ** (2 - P)).sum()) for yr in years}
        sig_dev_year = float(sigma_dev.loc[name, "sigma_year"])
        gap = np.log(k_year[2023] / k_year[2024])
        sig_emg_year = float(np.sqrt(max(0.0, (gap ** 2 - cv_year[2023] ** 2 - cv_year[2024] ** 2) / 2)))
        sig_year = max(sig_dev_year, sig_emg_year)
        dev_claims = {yr: int((ctx.dev[name].rows([yr]).y_full > 0).sum()) for yr in years}
        emg_claims = {yr: int((parts[yr].y_full > 0).sum()) for yr in years}
        cv_dev = {yr: float(mack[(mack.line == name) & (mack.ay == yr)].mack_cv_of_ultimate.iloc[0])
                  * np.sqrt(dev_claims[yr] / emg_claims[yr]) for yr in years}
        slope = float(ctx.models[name].small.slope)
        sig_trend = 1.5 * abs(slope) / 2
        sigma = float(np.sqrt(cv_k ** 2 + (1 + w[2023] ** 2 + w[2024] ** 2) * sig_year ** 2
                              + sum((w[yr] * cv_dev[yr]) ** 2 for yr in years) + sig_trend ** 2))
        test = line.test
        base25 = ctx.base(name, test, 2025, fill)
        e25 = test.exposure_plain.to_numpy()
        r24 = parts[2024]
        e24 = r24.exposure_plain.to_numpy()
        rate24 = r24.y_full.sum() / e24.sum()
        at24 = ctx.base(name, r24, 2024, fill).sum() / e24.sum()
        at25 = ctx.base(name, r24, 2025, fill).sum() / e24.sum()
        # New business: the blended loading from E5, the level re-solved on 2023-24 with it, as a scenario.
        nb_fit, nb25 = fit.new_business.astype(bool).to_numpy(), test.new_business.astype(bool).to_numpy()
        f_nb = float(w5_nb[(w5_nb.line == name) & (w5_nb.source == "blended")].nb_ae_relative_to_renewal.iloc[0])
        k_nb = deviance_scale(y, mu0 * np.where(nb_fit, f_nb, 1.0))
        scen25 = base25 * k_nb * np.where(nb25, f_nb, 1.0)
        preds[name] = pd.DataFrame({"row_id": test.row_id.to_numpy(), "adopted": base25 * k, "as_is": base25,
                                    "nb_scenario": scen25})
        row = {"line": name, "pattern": EMERGING_PATTERN[name], "missing_values_filled": fill, "level_k": k,
               "k_2023": k_year[2023], "k_2024": k_year[2024], "weight_2023": w[2023], "weight_2024": w[2024],
               "cv_k_bootstrap": cv_k, "sigma_year_used": sig_year, "sigma_year_developed": sig_dev_year,
               "sigma_year_emerging_between_years": sig_emg_year, "dev_cv_2023_scaled": cv_dev[2023],
               "dev_cv_2024_scaled": cv_dev[2024], "sigma_trend": sig_trend, "sigma_total": sigma,
               "k_low_90": k * np.exp(-1.645 * sigma), "k_high_90": k * np.exp(1.645 * sigma),
               "rows_2025": len(test), "rate_2025": (base25 * k).sum() / e25.sum(),
               "rate_2025_low_90": (base25 * k).sum() / e25.sum() * np.exp(-1.645 * sigma),
               "rate_2025_high_90": (base25 * k).sum() / e25.sum() * np.exp(1.645 * sigma),
               "rate_2025_as_is": base25.sum() / e25.sum(), "rate_2025_nb_scenario": scen25.sum() / e25.sum(),
               "developed_rate_2024_emerging": rate24, "level_2024_ae_as_is": float(rate24 / at24),
               "trend_2024_to_2025": at25 / at24, "mix_2025_book_vs_2024": (base25.sum() / e25.sum()) / at25,
               "developed_market_rate_2024": float(summary.loc[name, "developed_rate_2024"]),
               "mean_prediction_2025": float((base25 * k).mean()), "claims_2023_24": int((y > 0).sum())}
        if name == "buildings":
            # The level with 2023 fire at the 2024 rate: the ratio-of-totals change applied to the pooled scale.
            fire = ctx.emg_con.execute("""SELECT year(loss_date) AS yr, sum(net_cost_dev) AS cost FROM line_claims_dev
                                          WHERE line = 'buildings' AND peril = 'fire' GROUP BY 1""").df().set_index("yr").cost
            expo = {yr: parts[yr].exposure_plain.sum() for yr in years}
            excess = fire.get(2023, 0.0) - fire.get(2024, 0.0) / expo[2024] * expo[2023]
            row["level_k_2023_fire_at_2024_rate"] = k * (y.sum() - excess) / y.sum()
        level_rows.append(row)
        # The new-business sensitivity, weighted by expected cost.
        m = mu0 * k
        s_fit, s25 = float(m[nb_fit].sum() / m.sum()), float(base25[nb25].sum() / base25.sum())
        for label, r in (("emerging 2023-24", float(w5_nb[(w5_nb.line == name) & (w5_nb.source == "emerging 2023-24")]
                                                         .nb_ae_relative_to_renewal.iloc[0])), ("blended", f_nb)):
            nb_rows.append({"line": name, "loading": label, "nb_ae_relative_to_renewal": r,
                            "nb_cost_share_2023_24": s_fit, "nb_cost_share_2025": s25,
                            "nb_row_share_2023_24": float(nb_fit.mean()), "nb_row_share_2025": float(nb25.mean()),
                            "level_effect_2025": (s25 * r + 1 - s25) / (s_fit * r + 1 - s_fit)})
        print(f"  {name}: level x{k:.3f} (single years {k_year[2023]:.3f}, {k_year[2024]:.3f}; 90% "
              f"x{row['k_low_90']:.3f}-{row['k_high_90']:.3f}), 2025 rate {row['rate_2025']:.1f} against 2024 "
              f"{rate24:.1f}; with new-business loading x{f_nb:.3f}: {row['rate_2025_nb_scenario']:.1f}", flush=True)
    sample_path = common.PACK_ROOT / MARKETS["emerging"]["data_dir"] / "sample_submission.parquet"
    expected = common.read_expected_hashes(common.PACK_ROOT / MARKETS["emerging"]["expected_inputs"])
    actual = common.sha256_file(sample_path)
    if expected.get("sample_submission.parquet") != actual:
        raise RuntimeError(f"Input hash mismatch for sample_submission.parquet: got {actual}")
    rec["sample_sha256_emerging"] = actual
    sample = pd.read_parquet(sample_path)
    allp = pd.concat(preds.values())
    assert allp.row_id.is_unique and set(sample.row_id) == set(allp.row_id)
    files = {}
    for col, label in (("adopted", "level_reset"), ("as_is", "as_is"), ("nb_scenario", "level_reset_nb_scenario")):
        sub = sample[["row_id"]].merge(allp[["row_id", col]], on="row_id", how="left").rename(columns={col: "prediction"})
        assert len(sub) == len(sample) and sub.prediction.notna().all() and (sub.prediction > 0).all()
        assert (sub.row_id.to_numpy() == sample.row_id.to_numpy()).all()
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        path = SUBMISSIONS / f"submission_emerging_{label}_{dt.date.today():%Y%m%d}.csv"
        sub.to_csv(path, index=False)
        files[label] = {"path": path.relative_to(common.PACK_ROOT).as_posix(), "sha256": common.sha256_file(path),
                        "rows": len(sub), "min": float(sub.prediction.min()), "mean": float(sub.prediction.mean())}
    rec["submissions"] = files
    print(f"  wrote {', '.join(f['path'] for f in files.values())}", flush=True)
    return {"ev_w4_level_2025.csv": pd.DataFrame(level_rows), "ev_w4_new_business.csv": pd.DataFrame(nb_rows)}


def main() -> None:
    args = sys.argv[1:]
    which = args[args.index("--tests") + 1].split(",") if "--tests" in args else ["E1", "E2", "E3", "E5", "E4"]
    unknown = set(which) - {"E1", "E2", "E3", "E4", "E5"}
    if unknown:
        raise SystemExit(f"unknown tests: {sorted(unknown)}")
    t0 = time.time()
    cfg = common.load_config()
    rec = common.new_run_record(RUNNER, cfg)
    rec["tests"] = which
    try:
        ctx = Context(cfg, rec)
        frames = {}
        for test, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E5", e5), ("E4", e4)):
            if test in which and test == "E4" and "E5" in which:
                outputs_e5 = [write_csv(df, OUT / n) for n, df in frames.items() if n.startswith("ev_w5")]  # E4 reads them
                rec["outputs_written_before_e4"] = [o.name for o in outputs_e5]
            if test in which:
                frames.update(fn(cfg, rec, ctx))
        OUT.mkdir(parents=True, exist_ok=True)
        outputs = [write_csv(df, OUT / name) for name, df in frames.items()]
        rec["outputs_sha256"] = {o.relative_to(common.PACK_ROOT).as_posix(): common.sha256_file(o) for o in outputs}
        rec["metrics"] = {name: df.round(6).to_dict("records") for name, df in frames.items()
                          if "by_factor" not in name}
        rec["status"] = "success"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["versions"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                           **{pkg: metadata.version(pkg) for pkg in ("duckdb", "scikit-learn", "xgboost")}}
        rec["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        rec["elapsed_s"] = round(time.time() - t0, 1)
        common.append_run_log(cfg, rec)


if __name__ == "__main__":
    main()
